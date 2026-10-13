# 当前默认路径与 vLLM 的性能画像（2026-10-13）

本轮先 profile，再选择优化。当前主要差距在长输入的 GDN prefill：同一 511-token 输入，
native 记录到 24,576 次标量递推 kernel、110.17 ms；vLLM 的 FlashInfer CP 分块路径为
384 次 kernel、7.93 ms。单请求 decode 的两类大矩阵 kernel 总时间接近。下一轮优先定位
GDN 的串行依赖与状态流量，再处理 prefill attention；本轮没有修改生产算法或添加用户开关。

机器可读结果见 [冻结汇总](artifacts/cuda-vllm-current-profile-2026-10-13.json)。
这是诊断记录，不能作为同精度加速证明或五组独立配对验收。

## 测量条件与实际路径

- native 来源：干净 `9244acb85e0cffbc21c9255eb813a77ae46f428d`；测量前
  [10 项 CI](https://github.com/1994/inference.rs/actions/runs/38414797672) 全部成功。
  重新构建 CUDA release，二进制 SHA256 为
  `824cc07324b4ec5bf13e4f50b1142993d5e8728db28d910a6395210e0d918848`。
- vLLM：固定官方 `3fe917abb13160d60bea5330399f5c09ca282348` 的
  `0.31.1rc1.dev289+g3fe917abb` wheel；Torch `2.13.0+cu130`、FlashInfer `0.7.0.post1`。
  wheel 和实际 Python/site 身份保存在汇总与原始 invocation 中。
- 同一 RTX 5090、同一 Qwen3.8-27B-NVFP4 checkpoint、同一输入 token 数组、MTP 0、
  greedy、每请求完整输出 64 token；同一服务上限：显存利用率 0.88、context 65,536、
  最多 16 序列。每 case 一次排除的预热和三次测量。
- native 回读 prefill 宽度及 scheduler token budget 均为 256，窄图 64；vLLM
  `max_num_batched_tokens=256`，实际 long trace 也为 `256 + 255` 两次 prefill。
  native 单请求走 private decode；batch4 主体走四槽 pooled decode，pool 未禁用。
- 两端 prefix cache 开启，short/long/batch4 测量中的 prefix reuse 都是 0。
  hot_long 的实际 reuse 是 native 3,584、vLLM 3,136；剩余工作不同，单列诊断。
  vLLM 日志中的 attention page 是 1,568 token，native cache block 是 32，缓存布局不同。
- native 保留 F32 激活与 recurrent state；vLLM 模型 dtype 为 BF16，FlashInfer 调用把
  initial state、g、beta 转为 F32。这仍有激活、归一化和求值顺序差异，不能宣称同数值契约。
- 所有模型/分析重任务经 `safe-run.sh`：32 GiB 主机上限、零 swap、独占锁；每秒检查
  GPU 进程及 cgroup，保留桌面后台进程。所有七个有效 run 均完成，没有外来 compute 进程。

第一次试跑复用了旧调参表，并显式禁止新调参；旧 kernel fingerprint 失效后落到保守
GEMM。这组结果和后台 watcher 的权限错误保留为 `preliminary-stale-cache-invalid.json`，
不计为默认基线。之后使用默认自动调参补齐缓存，基础计时和 profiler 均命中 46 项
缓存选择，没有新测量、预算耗尽或缓存写入。冻结表 SHA256 为
`ed6747b0514451a2944b9291ec673a2c674f20feed33b053be1ff5231bcb0b28`。

## 未插桩服务结果

native 数字为 profiler 前后两个独立进程的各 case 中位数再取中位数；vLLM 为一个
独立进程的三次测量中位数。时间单位 ms，均为客户端观测。

| case | native TTFT | vLLM TTFT | native TPOT | vLLM TPOT | native group makespan | vLLM group makespan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short，51 token | 44.76 | 46.53 | 15.62 | 14.82 | 1,031.72 | 979.95 |
| long，511 token | 240.37 | 96.77 | 15.72 | 14.86 | 1,231.46 | 1,033.34 |
| batch4，每请求 52 token | 172.05 | 90.16 | 18.89 | 16.27 | 1,369.18 | 1,115.95 |
| hot_long，3,647 token | 67.75 | 99.26 | 16.20 | 14.88 | 1,089.30 | 1,036.83 |

这不是沿用旧版本、旧 MTP 深度或旧 token budget 的对照。当前 long 的差距约 2.48 倍；
short 首 token 接近，单请求 decode 相差约 5.8%。上述样本量适合定位差距，不足以签发
稳定提升、置信区间或任务质量结论。

## 长输入：最大的差距来自递推执行形式

native 的两个 prefill graph 分别记录 14,001/14,003 个 kernel。每层 GDN 按捕获宽度
逐行记录 `delta_entry`：48 层 × 256 行 × 两次图 = 24,576 次，其中 511 个真实 token
对应 24,528 次，另 48 次为填充。它仍每个 token 读写状态，graph replay 没有消除图内的
串行依赖。来源是 [逐行 dispatch](../../crates/backend/cuda/src/resident/batch.rs) 和
[F32 recurrent kernel](../../crates/backend/cuda/src/resident/recurrent.rs)。

vLLM 两个 GPU prefill scope 都记录到每层四个 FlashInfer CP 阶段，数量为
48 层 × 四阶段 × 两个 scope = 384。该数来自实际 CUDA trace，不根据文件名推测算法。

| GDN 阶段 | native kernel 次数 / GPU ms | vLLM kernel 次数 / GPU ms |
| --- | ---: | ---: |
| 标量 delta 递推 | 24,576 / 110.17 | — |
| CP TPrecompute | — | 96 / 0.45 |
| CP MNPrecompute | — | 96 / 2.93 |
| CP FixupHmma | — | 96 / 0.87 |
| CP Prefill | — | 96 / 3.68 |
| GDN 合计 | 24,576 / 110.17 | 384 / 7.93 |

这表明值得研究的是块内预计算、并行读出和少读写状态。kernel 数减少 64 倍不等于
数学乘加减少 64 倍；CP 有额外预计算，优势是改变执行依赖和数据复用。此处 GPU 时间
来自不同 profiler，约 13.9 倍的阶段差异是定位线索，不能作为可直接移植的同精度收益。

native 的 long 图内 kernel 时间合计 217.56 ms，其中 GDN 110.17 ms，约 50.6%。
vLLM 激活仍为 BF16；本仓过去的 WY/CP/读出候选已有质量反例，不能直接换成它的算术。
下一候选须保留 F32 state、原 F64 对照/阈值、原 256 题门槛及完整服务输出检查。

用时间模型估算，把 110 ms 的段降到 8 ms，会把约 240 ms 的 TTFT 降到约 138 ms，
说明该段有足够大的研究价值；138 ms 是定位用的外推，不是已实现的结果。

## 其他差距与优先级

**Prefill attention 是第二个明确的 GPU 差距。** 两端 long 都有 32 次 attention
kernel，native `decode_entry` 合计 15.05 ms，vLLM FlashInfer 对应段约 0.77 ms。
native [attention_prefill](../../crates/backend/cuda/src/resident/attention_prefill.rs)
按 query/head 对 KV block 逐块遍历；下一轮需要检查读取复用和实际 query tile。
相同 kernel 数并不意味着相同工作组织。已有 attention 候选未过原质量门槛，本轮不启用。

**并发 prompt 的分组还有差距。** native batch4 记录四次私有 prompt 图；第二组任务
包含一次 decode 和三次 prefill，三个 prefill 仍顺序执行。vLLM 的 GPU annotation 为
`execute_context_1(52)_generation_0(0)` 和
`execute_context_3(156)_generation_1(1)`，把后面三个 prompt 的投影并入一次执行。
这支持研究跨请求的投影复用与 mixed batching；batch4 的 TTFT 包含排队/资源准备和
组内等待，不能只用矩阵 MAC 或中位数判断。

**单请求 decode 的大矩阵不是当前最大的差距。** native short 的 63 次稳态 decode 中，
FP8 `matmul_entry` 合计 445.39 ms、NVFP4 `packed_entry` 342.21 ms，约 12.50 ms/token。
vLLM 对应窗口约 12.62 ms/token，时间边界有少量首尾 kernel 裁剪，因而只作近似比较。
两端应继续检查量化、norm/pointwise 融合和 pooled 小投影，而非立即扩大大 GEMM 参数扫描。
native pooled BF16 小投影是 `[1,2,1]` 的两 CTA grid，63 轮约 105.18 ms；它已有原质量
失败的候选，不能以局部速度替代服务验收。

**先前的末层裁剪应降优先级。** 对该 511-token workload，只有第一块可跳过最后状态
之后的读出，末块必须输出。逐节点 profile 中候选的 12 个节点（1142–1153）合计约
0.60 ms，约占当前 TTFT 的 0.25%。这只是观测段完全消失的乐观量级，候选实际收益还
没有测；它解释了为什么“每行省 2.99 亿 MAC”不足以决定下一步。

硬件性能计数器受 `RmProfilingAdminOnly=1` 限制，本轮没有实测 DRAM 字节或带宽。
权重大小除以 kernel 时间只能得到流量模型，不能据此宣称 GEMM 已达到显存带宽极限。

## Profiler 干扰、输出检查和采集边界

native 依次运行未插桩 A、图边界 event、逐节点 event、Nsight、未插桩 B。Nsight 只
抓每个 cold case 的第一个测量 repeat；vLLM 使用内置 Torch CPU/CUDA profiler，并
以单独未插桩进程为参考。注入 Nsight 的 vLLM 在启动前 `anon_pipe_read` 卡住，故障和
只停止本任务 unit/session 的记录保存在 `vllm-nsys-stall.json`。

- native 图边界计时 TPOT 干扰约 0–1.1%；逐节点 event 对 short/long TTFT 的干扰
  约 +28%/+9%，对 batch4 TPOT 约 +58%、makespan 约 +53%。逐节点时间不当作服务基线。
- native Nsight TPOT 干扰约 +1.9–2.8%。两次未插桩运行与 Nsight 的全部 21 个测量请求、
  每个完整 64-token 流一致。图边界和逐节点 profile 的 batch4 有逐 token 差异，均保留。
- vLLM Torch 的被抓取 long repeat TTFT 为未插桩中位数的约 1.41 倍，short 为约 1.87 倍。
  不能拿 profiler 内的 TTFT 比引擎。vLLM long 被抓的 repeat 0 输出流一致，repeat 1
  及部分 batch4 请求在两进程间不一致；单独记录，没有删掉失败样本或宣称质量通过。
- native Nsight 在每个 trace 都警告可能丢失 CUDA/OSRT events。63 次 decode 的主干
  kernel 数、两次 prefill 的 GDN 数与结构相符，但记录活跃区间仍只是下界。
  `cuStreamSynchronize` 时间大量与 GPU 活跃重叠，不能把它整个当作 CPU 开销；空白区间
  也不能直接叫 GPU idle。两端 kernel 名称按实际调用、shape 和 scope 归类。

## 证据与复算

原始目录是 `/home/r/benchmarks/inference.rs/profile-before-optimization-2026-10-13/`：
`plan.json`、冻结二进制/输入/调参表、七个有效 run 的完整 serving JSON/telemetry/
process snapshots、native `.nsys-rep/.sqlite`、vLLM 四个 `.pt.trace.json.gz`、所有失败记录。
`evidence-manifest.json` 保存 105 个证据文件的大小与 SHA256；每次 invocation 的原始 runner
版本已按其 SHA256 恢复并保存到 `runner-snapshots/`，没有把后来修改的 runner 当作早期版本。

`analyze.py` 按 trial 字节偏移复算 host/event 数据和完整 token 流；`analyze-traces.py`
只读 Nsight SQLite，区间先合并再求活跃时间；`analyze-vllm.py` 用 trace 的
`baseTimeNanoseconds` 对齐服务窗口，并额外读取 GPU prefill annotation，排除首个 decode
对 prefill 分段的污染。重新运行分析不启动模型。

本轮结论是下一步的工作顺序：GDN 分块依赖/状态流量 → prefill attention → 并发投影复用
与辅助 kernel。通过原质量和端到端门槛的优化直接成为默认，实验调度留在测试与诊断路径。
