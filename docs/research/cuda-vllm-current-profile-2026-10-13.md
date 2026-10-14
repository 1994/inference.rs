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

## 后续：GDN 合并 kernel 的数值障碍已定位（2026-10-14）

沿上述主线检查实际 SASS，原标量与寄存器分块路径的 F32 归约树不同。
状态投影和输出读出还各自使用不同的加法树；仅修复投影可以使状态逐位一致，
但输出仍有 4,600 / 6,144 个末位差异。用原始 F32 中间值按指令顺序复算，
两端各 6,144 个投影值及原标量的 6,144 个读出值均逐位吻合。

新 `delta_grouped` 显式保留两处归约顺序，跨 token 保留状态。
8 组原始间隙用例的完整状态和输出均逐位一致；新增 64/256 行捕获图的空块、
单 token、尾块、负 offset 与弱衰减/相关 K，以及较小 head 数，共 11 组也全部逐位一致。
F64 最大绝对误差为 `1.155110407e-6`，保留原 `2e-5` 门槛。

同一完整 27B checkpoint、同一 511-token 提示，在 64/256 两种捕获宽度下，
所有返回的 hidden、非空 logits 和每次 39,223,296 个状态值均逐位一致。
比较覆盖每块 prefill、三步验证、提交一 token 后的回滚，以及随后 16 步 decode。
测试额外检查返回数量、状态 ID/张量数量、长度和有限值；空 logits 单列计数，
不把它们当成已验证的词表输出。

[冻结数值汇总](artifacts/cuda-gdn-ordered-reduction-2026-10-14.json)记录二进制与实际
生产 kernel/dispatch 源码哈希、原始日志哈希和 manifest。原始目录为
`/home/r/benchmarks/inference.rs/gdn-scalar-chunk-lowering-2026-10-13/`；
该目录沿用创建日期，后续结果没有覆盖旧试验。一次运行因新出现的桌面 RustDesk
进程被预检查拒绝，未启动 GPU 测试；确认它的身份后只在功能验证中保留该后台进程，
没有把该次重试用于计时。

本阶段未做配对性能或原 256 题任务质量验收，不能据此宣称端到端加速。
现有捕获调度在已验证的全列 128 维、64/256 行条件下使用新 kernel，
量化默认切换仍需后续服务与质量证据。没有新增用户算法开关。


## 10-14：独立 value 列分块与剩余热点

保持 GDN 原标量的 F32 归一化、prediction 和 readout 求和树，把互不依赖的
128 个状态列分成 4 个 32 列 block，每个 block 跨 token 保留状态。原 128 列
分支保留已验证的编译器归约；分列分支显式固定求和树，避免状态末位变化。

生产内核和分块选择都在 cfg(test) 外。cfg(test) 挂载比较/基准模块，并提供
只用于对照的旧 Delta 捕获钩子。符合现有准入条件的非量化模型默认使用 32 列，
不需要用户选择算法。量化模型及 MoE 的生产准入不变；27B NVFP4 的自动量化
候选在冻结源码归档中单独测量，不能把它的结果说成生产默认收益。

本次 CUDA release 通过严格 Clippy（全部 target）、格式/布局/政策检查。
45 组 fixture 覆盖 32/64/128 列、64/256 宽度、零/部分/满块、负 offset、
相关输入及弱衰减，state/output 全部逐位一致，原 F64 误差门槛 2e-5 不变。
新增默认 F32 dense 捕获检查不设置 loader 实验选项，比较完整 hidden、非空
logits 和每次 786,432 个 state 元素。27B 的诊断准入检查覆盖 511-token
prefill、verify 3/commit 1 回滚和 16 步继续 decode，每次比较 39,223,296 个
F32 recurrent/conv state 元素，均逐位一致；不声称比较了 FP8 KV buffer。
独立 CUDA event 对标量的速度比分别约 3.16（64 宽度）和 3.32（256 宽度），
这是 kernel 回归门槛验证，不是服务速度证明。

重新执行完整 256 题，标量与自动量化候选都为 240/256，输出 token、文本和
finish reason 全部相同。相对原生标量的单侧 95% 退步率上界 1.1634%，通过
原 2% 门槛；相对固定 vLLM 243/256 仍有 5 项退步、2 项收益，上界 4.0626%，
未通过。干净标量对该参考也有完全相同的失败项；分块没有新增这些质量差异，
也没有修改门槛来宣称完成 vLLM 非劣验证。

五组独立进程配对中，511-token TTFT 的进程中位数汇总为标量 259.63 ms、
候选 167.54 ms；配对速度比中位数 1.5167，bootstrap 95% 区间
1.4454–1.8888。所有样本保留。batch4 有 8/60 对请求流不同，标量自身重复
也存在变化，多个时间指标的进程间漂移超过原 10% 门槛。因此服务验收未通过，
这些仍是诊断记录；TPOT 没有稳定的大幅收益。

候选的新 long Nsight 图内记录为 3,620 个 kernel、144.88 ms，其中 GDN
96 个、37.08 ms，占 25.59%；transpose 为 0.62 ms。旧标量 profile 为
28,004 个、217.56 ms，GDN 为 24,576 个、110.17 ms。这些诊断指向继续优化
GDN，但不是无插桩的同条件性能证明。新 trace 提示 CUDA/OS runtime 事件
可能未完整收集，且未采集 OS 调度；GDN 数量与 48 层 × 两次图一致，仍不能
据此断言 API 完整、CPU 因果关系或实测 HBM 带宽/occupancy。

下一步减少重复计算：27B 的 value/key 头数比为 3，结合 4 个列块，Q/K
归一化重复 12 次，每个 value head 的 beta/decay 重复 4 次。原型按
token/head 预计算 F32 normalized Q/K 和门控，再由独立状态列读取。
256 宽度额外 Q/K 存储 4 MiB、门控 96 KiB；计算次数减少不是速度证明，
新增启动、读写与完整状态验证都要计入测量。新 profile 中 attention 仍约
15.34 ms，作为第二优先级。

命令、SHA、结果、失败记录见
[分列实现与候选诊断证据](artifacts/cuda-gdn-value-column-split-2026-10-14.json)。


## 10-14：沿 GDN 减少重复计算，集成共享输入与 16 列块

在上面的 32 列实现之上，按 token/head 预计算 F32 normalized Q/K、beta 和
decay，独立状态列读取同一结果。与原 32 列的逐块归一化相比，27B 的 Q/K
归一化从 12 次变为一次，门控从 4 次变为一次。状态精度、求和树与原有准入
保持不变；现有允许分块的默认路径自动生效，没有新增用户算法开关。

先对原型再 profile：256 宽度的归一化/门控单 kernel 中位数之和约 13.22 us，
递推约 339.64 us，后者占阶段中位数之和约 94.5%。这是包含预热和计时执行的
插桩阶段诊断，不能把阶段中位数之和当成配对图中位数。于是下一步测试更小的
状态列块，而没有继续优化已经很小的预计算阶段。8 列在 64 宽度上回退约
40%，十次调用全部被原 5% 门槛拒绝，未继续测其 256 宽度；失败记录全保留。
16 列更有收益，经过固定探索后，再用五组新进程独立确认。

独立确认仍按进程内 A/B 或 B/A 交替 CUDA event 配对，50 ms 预热、100 ms
计时预算、10–100 次采样；原始样本全保留。下表单位为 us，速度比由五组配对
进程中位数求中位数，95% 区间由全部 5^5 bootstrap 组合得到。

| 缓存 / 宽度 | 原 32 列 | 共享输入 + 16 列 | 配对速度比 / 95% 区间 |
| --- | ---: | ---: | --- |
| cold / 64 | 106.27 | 97.76 | 1.0872 / 1.0856–1.0877 |
| cold / 256 | 395.44 | 345.44 | 1.1448 / 1.1445–1.1449 |
| warm / 64 | 103.50 | 84.94 | 1.2190 / 1.2175–1.2203 |
| warm / 256 | 393.15 | 309.79 | 1.2695 / 1.2678–1.2703 |

这是 isolated kernel 收益，不是现有 27B NVFP4 默认服务的 TTFT 提升。
各组进程中位数漂移均低于 0.3%。运行时采用 16 列并共享输入；在正式集成的
CUDA release 上重新验证 75 组 fixture，150 次 state/output 比较全部逐位
一致，最大 F64 绝对误差 1.1552e-6，原 2e-5 门槛不变。默认 dense 捕获的
两种宽度均逐位一致。27B 的诊断加载对照仍覆盖 511-token prefill、verify
3/commit 1 回滚及 16 步继续 decode；26/20 次 state checkpoint，每次
39,223,296 个 F32 recurrent/conv 元素均逐位一致，hidden/logits 也一致。
四项预算测试验证跨层共享、不同 key 几何、全部保留宽度和分配前 overflow。

缓冲区按 (key heads, value heads, dimension) 跨层共享，准入与分配使用同一
几何集合。256 宽度的新增 Q/K 4 MiB、门控 96 KiB；加原输出后总计
10,584,064 bytes，64 宽度为 2,646,016 bytes，两种保留宽度共
13,230,080 bytes，不乘以 48 层。

此前的完整 256 题与五组服务结果属于旧 32 列自动量化归档，并没有在新 16 列
共享实现上重新执行，不能沿用为新版本的服务/质量验收。量化及 MoE 的正式默认
准入仍不变；不宣称显式比较了 FP8 KV buffer。下一轮需要先验证量化候选的
完整服务行为，处理已发现的 batch4 路由/流一致性差异，再决定准入。
实现、源码与二进制 SHA、原始样本和全部失败索引见
[共享输入实现与独立确认](artifacts/cuda-gdn-shared-inputs-2026-10-14.json)。

## 10-14：集成共享输入后的实际模型 profile

CUDA 代码提交 ac2a7fdc804804d71fb2d6811059982dd9351810 的完整 CI 已 10/10
成功。随后从该提交创建隔离归档，只移植原量化诊断的按节点准入；计算内核、
16 列分块和共享输入与生产提交相同。该归档用于 profile，正式量化默认准入
仍未扩大，也没有新增用户算法开关。固定模型、511-token long 输入、46 项
tuning cache、上下文/并发/输出长度与上一份 32 列诊断一致。

长输入的两个 256 宽度 prefill 图合计 3,812 个 kernel、136.895069 ms GPU
kernel 时间。其中状态递推 27.930901 ms，Q/K 归一化 0.446588 ms，门控
0.625690 ms；完整 GDN 阶段共 29.003179 ms，占图 GPU 时间的 21.1864%。
transpose 0.614394 ms，attention 15.027716 ms。

历史 scalar profile 的 GDN 为 110.167342 ms、占 50.6%；32 列候选为
37.075934 ms、占 25.59%。三份 trace 来自独立运行且有插桩，不能用这些
差值登记正式 TTFT/TPOT 或置信加速比。各 trace 的 CUDA/OS runtime 不完整
及缺少 OS scheduling 警告全部保留；不推断 HBM 带宽、occupancy 或 CPU 原因。

下一步的 GDN 主线应针对状态递推，它占当前 GDN 时间约 96.3%，继续优化
约 1.07 ms 的共享准备阶段已难以贡献大收益。projection 各大族合计占据较多
时间；已核实 dense projection 的 N tile 是 64，不能假定 beta/alpha 融合
会使 padded MMA 计算量减半。attention 仍有明显差距，checkpoint 的 32 个
BF16 KV 缩放值在 CPU 枚举所有 254 个有限 FP8 code 后，第三个 BF16 残差
分量均为零；这只是后续删去恒零乘法的线索，尚未完成 GPU/F64/全模型验证，
没有修改其生产计算。最初 BF16 字节解码错误的记录已明确作废并保留。

此轮诊断四种 workload 均完成，外来 GPU 进程 guard 无违规，冻结 tuning
cache 和二进制 hash 未变。此前旧候选的完整质量/服务结果不作为新候选验收。
原始目录为 /home/r/benchmarks/inference.rs/gdn-scalar-chunk-lowering-2026-10-13/shared-inputs-current-profile-v1；
[实际模型 profile 汇总](artifacts/cuda-gdn-shared-inputs-current-profile-2026-10-14.json)
记录准入 diff、编译身份、trace、警告及 CI 证据 hash。

## 2026-10-14：消除递推循环中的行边界检查

共享输入后，长 prompt 的诊断 trace 中，状态递推仍占 GDN 的 96.3%
（27.930901 / 29.003179 ms）。因此这一轮继续减少递推自身的重复工作。

cuTile JIT 日志显示，以常量 LANES 为循环上界时，多项加载的行检查留在
条件分支内部。生产代码改为从 normalized partition 的行维度取得循环上界，
并在循环前建立 partition。prepared buffer 仍有 LANES 个 padding 行，
原有 count 和 offset 判断、F32 运算树、状态布局及 workspace 大小保持一致。
行维度证明由编译器处理，跨输入的约束在 launch 校验，不再每个 token 检查。

两个实际宽度的 SASS 中，整个 kernel 的静态指令由 992 条降至 904 条；
主循环区间由 560 条降至 457 条。寄存器由 127 变为 128，STACK 由 160
降至 112 bytes；共享内存仍为 30,988 bytes，循环区间仍有 7 条同步指令。
这些是静态区间，包含不同控制流路径，不能当作动态执行次数或硬件计数器。

候选先经过固定五组独立进程的发现测量，再以反向 AB/BA 顺序进行五组新进程
确认。每组都测冷、热 L2，并保留完整原始向量；下表只列独立确认结果。

| L2 | 宽度 | 原共享输入实现 ms | 本次实现 ms | 配对 speedup | 95% 区间 |
| --- | ---: | ---: | ---: | ---: | --- |
| 冷 | 64 | 0.097312 | 0.088016 | 1.1055 | 1.1034–1.1066 |
| 冷 | 256 | 0.345216 | 0.307488 | 1.1227 | 1.1213–1.1229 |
| 热 | 64 | 0.084576 | 0.075424 | 1.1221 | 1.1204–1.1230 |
| 热 | 256 | 0.308736 | 0.275328 | 1.1212 | 1.1202–1.1216 |

四组原、候选进程中位数的范围漂移均小于 0.3%，置信区间下界均超过原有
1.02 useful-gain 门槛。该数字包括相同的共享准备、递推和输出转置，是隔离
CUDA graph 的收益，不能推导成同幅度的端到端 TTFT 收益。

归档原型通过 75 组数值 fixture；实际修改的 15 组 16 列 fixture 中，state
和 output 的 30 次逐位比较均为零差异。独立 F64 reference 的最大绝对误差
为 1.1551104e-6，保持原 2e-5 门槛。默认捕获路径的 64/256 宽度亦逐位一致；
完整模型分别通过 26/20 次 checkpoint，每次比较 39,223,296 个状态元素，
hidden、logits、verify 3/commit 1 回滚和 16 步继续 decode 均逐位一致。
生产构建的独立验证结果见下方证据文件。

本轮也保留两个未合入的结果：8 worker warps 在宽度 64 就触发原 5% 回退
门槛；内部列布局虽然冷 L2 快 5.7%–7%，热 L2 只有约 1%，未满足原收益门槛。
它们的源码、失败及原始测量均保留，未降低门槛。

这一优化直接作用于已准入的默认有序路径，不新增 CLI 或环境变量开关。
量化模型的生产准入仍受此前 serving 和相对 vLLM 的质量门槛约束。
后续重点仍是循环中的加载、布局转换和同步；当前 trace 与 SASS 尚不能证明
哪种硬件资源决定这些开销，需要继续用可分离的实验验证。

[行边界检查优化证据](artifacts/cuda-gdn-axis-bounds-2026-10-14.json)。

## 2026-10-14：行检查优化的完整模型 profile

CUDA 代码提交 0e7a969e9a05b193d1a86faee458021f57cebaa2 的
[完整 CI](https://github.com/1994/inference.rs/actions/runs/38473384061) 已 10/10 成功。
同一生产 kernel 在量化诊断归档中，采用与上一轮相同的准入补丁、四个固定
场景及 46 项冻结 tuning 表，重新采集了完整模型 trace。

511 token 的两次 256 token 图仍执行 3,812 个 kernel。递推从上一轮
27.930901 ms 变为 24.683036 ms；共享 normalization 为 0.445523 ms，
gates 为 0.623499 ms，GDN 总和为 25.752058 ms，较上一轮 29.003179 ms
少 3.251121 ms。attention 为 15.027409 ms，输出 transpose 为 0.615148 ms，
与上一轮基本一致；整体图 kernel duration 总和为 133.509172 ms。
GDN 在该总和中的占比从 21.2% 变为 19.3%，减少的时间集中于目标递推阶段。

四个场景均完成，GPU guard 无干扰记录；protected profile 峰值 5.6 GiB，
自身 swap 为零。Nsight 仍报告未采集 OS scheduling、部分 CUDA events 和部分
OS runtime events，全部保留。这里是一次带 profiler 的诊断，和独立隔离复测
相互印证，不能作为配对 serving 收益或量化模型默认准入的结论。

[完整模型 trace 与 CI 证据](artifacts/cuda-gdn-axis-bounds-current-profile-2026-10-14.json)
记录生产 kernel hash、诊断二进制及准入身份、原始 trace/hash 和全部警告。

## 2026-10-14：输入打包的失败结果

以已提交的行检查优化版本为 control，分别测试 Q/K 与 beta/decay 全部配对，
以及仅配对 beta/decay。两者均通过原 75 组数值 fixture，state/output 逐位一致，
F64 最大绝对误差保持 1.1551104e-6。

每个候选按固定五组进程分别测冷、热 L2，全部原始向量保留。整包版本的
64/256 宽度配对 speedup 冷 L2 为 1.0090/1.0145，热 L2 为 0.9669/0.9620；
gate 单包版本冷 L2 为 1.0159/1.0216，热 L2 为 0.9690/0.9686。
都未满足原有全部场景的 1.02 lower95 useful-gain 门槛，因此未独立确认、
未做完整模型推广，也未改生产代码。

整包版本循环区间静态指令从 457 降至 426，却从 7 条同步变为 8 条，
shared 从 30,988 变为 32,012 bytes。gate 单包保持 7 条同步，循环区间
为 460 条，热 L2 同样退化。因此不能把回退仅归因于额外同步，亦不能仅凭
源码 load 数或静态指令减少判定加速；缺少硬件计数器时不推断具体资源原因。

下一项假设是把 padding 输出初始化与有效 token 循环分离，使 JIT 有机会把
剩余 head 检查放入循环 preheader。该假设尚未实现或测量，需特别防止短请求、
全 padding 和部分计数路径回退。

[失败实验与原始样本索引](artifacts/cuda-gdn-input-packing-rejected-2026-10-14.json)。

## 2026-10-14：有效 token 循环与 head 检查前移

以 c89fa3a 中已提交的行检查优化为 control，本轮先初始化 padding 输出，
再循环处理 [max(0, -base-offset), min(count, normalized_rows))。
所有 partition 和 head 下标在循环前计算，F32 更新与两种 reduction 的源码
表达式逐 token 保持一致。checked load/store 仍由 cuTile 验证，未绕过检查。

第一版数值通过后，JIT 仍指出两个 head 加法在循环体中；移至循环前后，
三组实际几何的编译日志均不再报告循环内 bounds check。75 个 fixture 的
150 次 state/output 比较全部逐位一致，独立 F64 最大绝对误差保持
1.1551104e-6，小于原 2e-5 门槛。默认捕获及完整模型 64/256 宽度亦逐位一致，
分别比较 26/20 次完整 state checkpoint，每次 39,223,296 个元素；hidden、
logits、verify 3/commit 1 回滚及 16 步继续 decode 都通过。生产版本还重新
验证全部 8/16/32/64/128 列几何，结果见证据文件。

主循环区间静态指令由 457 降到 324–327，同步由 7 条降到 4 条；
shared 仍为 30,988 bytes。但整个 kernel 静态指令变为 1,848/1,888，
REG 从 128 增为 168，STACK 从 112 增为 176 bytes。这些静态数据不能
解释为实际动态指令数、占用率或 spill 测量，最终采用与否由运行证据决定。

固定五组发现进程后，再以反向 AB/BA 顺序运行五组新进程确认；每组均测
冷、热 L2。范围覆盖 64/256 满块，1 token、51/64、63/64、255/256，
零 token，以及 offset=-3/-5。每轮保留 110 组图时间向量、220 次 state/output
逐位检查；未删样本。下表为独立确认的满块结果。

| L2 | 宽度 | 原实现 ms | 本次实现 ms | 配对 speedup | 95% 区间 |
| --- | ---: | ---: | ---: | ---: | --- |
| 冷 | 64 | 0.087840 | 0.074432 | 1.1793 | 1.1784–1.1806 |
| 冷 | 256 | 0.307952 | 0.255584 | 1.2050 | 1.2044–1.2059 |
| 热 | 64 | 0.075520 | 0.066912 | 1.1285 | 1.1277–1.1301 |
| 热 | 256 | 0.276192 | 0.245504 | 1.1251 | 1.1235–1.1269 |

满块四组 lower95 均超过原 1.02 useful-gain 门槛。全部部分输入的配对
speedup 中位数至少 1.1249，未触发原 5% 回退门槛；所有进程中位数漂移
均小于原 10% 上限。因此 bulk zero 的额外写入在本轮覆盖范围内没有回退。
这些是隔离 CUDA graph 的数据，端到端 serving 仍需相应复测。

这一修改替换已准入的默认 kernel，不增加分配、预算或用户算法开关；
量化模型的默认准入仍受已有质量和 serving 门槛约束。
[有效循环优化证据](artifacts/cuda-gdn-active-range-2026-10-14.json)。

## 2026-10-14：有效循环优化的完整模型 profile

CUDA 代码 036f56ecbeebcc42bb70605fd4bb1a13c9715bab 的
[完整 CI](https://github.com/1994/inference.rs/actions/runs/38480229924) 已 10/10 成功。
同一 kernel 在量化诊断归档中，采用和上一轮完全相同的准入补丁、四个固定
场景与 46 项冻结 tuning 表，再次采集整模 trace。

511 token 的两次 256 token 图仍有 3,812 个 kernel。GDN 递推由上一轮
24.683036 ms 降至 21.710563 ms；共享 normalization 为 0.444349 ms，
gates 为 0.625337 ms。GDN 总和从 25.752058 ms 降至 22.780249 ms，
减少 2.971809 ms，时长减少约 11.54%，占图 kernel duration 总和的 17.4%。
该总和为 130.841668 ms，输出 transpose 为 0.624758 ms。

attention 从上一轮 15.027409 ms 变为 15.526432 ms，约上升 3.3%；
本轮未改 attention。由于这是单次带 profiler 的诊断，该变化和请求延迟
均保留，不能据此做配对 serving 判定。隔离图的独立复测支持目标递推收益，
端到端验收仍按原质量、漂移、回退与请求完整性门槛进行。

四场景均完成，guard 无 GPU 干扰，protected profile 峰值 5.6 GiB、自身
swap 为零。仍保留未采集 OS scheduling、部分 CUDA events 和部分 OS runtime
events 三类警告。量化模型的默认准入仍未改动。
[trace、编译身份与 CI 证据](artifacts/cuda-gdn-active-range-current-profile-2026-10-14.json)。

后续量化 serving 的一项证据缺口已有源码定位：ExecutionTask 含 request 和
state 两个 ID，现有 CUDA host profiler 只输出 state。记录已有 request ID
即可建立关联；此项尚未实施或验证，不能把 private/pooled 路由当作 token 差异
的已确认原因。

## 2026-10-14：补齐请求与 state 的路由关联

CUDA host profiler 现将已有 ExecutionTask.request 与 state 一起输出；
serve-compare 客户端现记录实际提交的 payload.id。两处只是现有记录的字段，
不改调度、数值或算法选择，不记录 token 内容到 host profile。

在当前标量默认路径和最新有序 GDN 诊断路径之间，按固定三组独立进程配对、
交替启动顺序，使用相同四场景、token 输入、64 输出上限、MTP0、固定 tuning，
完成六个进程，逐条关联 126 条测量 stream。63 条配对 stream 中有 7 条
token 不同，全部位于 batch4；两边在位置 52 是否经过一次 private decode
均不同。首个差异出现在生成 token index 2、12、14、20 或 22，当时双方已
在 pooled 路径，不能把当前 pooled replay 当作已确认原因。

相同 case/repeat/slot/输入 token hash 的跨进程比较中，标量路径有 4 条、
有序路径有 7 条差异。这与“只有新 GDN 错了”的判断不一致，但仍只是关联。
不同 repeat 的冻结输入本来不同，不能作 self-repeat；一版分析曾这样比较，
已明确作废其 self-repeat 字段，保留原文件和更正说明，配对字段不受影响。

host profiler 会扰动调度，因此这些请求耗时不作 serving 性能验收。全部
stream、request/state/slot/position 与原始 host 日志/hash 保留；默认量化准入
和原质量/性能门槛未变。

数值定位需按实际 52 token 场景，从相同 state/FP8 KV 和输入 token 比较
直接 pooled 与 private 一步后迁入 pooled。已有小投影算术实验不得因路由关联
重新启用：高精度 private MMA 和 pooled F32 实验均曾未通过原完整质量门槛。
[请求与路由证据](artifacts/cuda-gdn-request-state-route-2026-10-14.json)。

## 2026-10-14：实际容量下复现 private/pooled 数值分歧

请求关联代码 aa83fea8d480f65009cd05b093c21e442d26a8da 的
[完整 CI](https://github.com/1994/inference.rs/actions/runs/38484453543) 已整体成功，
10 项 job 全部成功。下面的数值实验只在该提交的独立归档中增加诊断测试，
没有改变生产算术、调度、默认准入或用户开关。

选取原六进程记录中三个实际 52 token 输入，从相同前缀的完整 F32 state 和
所有有效 FP8 KV 位置出发。服务 private 请求容量为 116，实际分配 128；
四槽 pooled 容量为 8192。一个槽在位置 52 直接 pooled，另一个槽先执行
private，再将其状态逐位一致地迁入 pooled。之后两个槽都采用默认 pooled
图，逐步使用同一输入 token，覆盖生成 token index 2 至 24。

| 原服务案例 | 首个不同 token index | private 一步后迁入 / 直接 pooled |
| --- | ---: | --- |
| batch4-repeat1-slot0 | 14 | 594 / 321 |
| batch4-repeat1-slot1 | 20 | 19797 / 4851 |
| batch4-repeat0-slot2 | 2 | 220 / 19 |

三个案例的两条受控轨迹，直到并包括首个分歧 token，都与相应原服务 stream
逐个 token 完全一致。初始状态和迁移复制均逐位一致；此前 decode 路径的差异
可以在后续同一 pooled 图中复现这些服务差异。这是上述三个案例的直接数值
证据，不能自动推广为全部七条差异或完整 serving 验收。

受控干预进一步区分小投影算术与容量。在相同 128 容量下，只将 pooled 的
96 个小 BF16 投影替换为 private 的 F32 GEMV，首步 hidden、全 vocabulary
logits、全部 F32 state 与所有有效 FP8 KV 均逐位一致。保持该干预而采用实际
8192 pooled 容量，三个案例的 state 差异分别仍有 830,802、8,145,780、
29,751,503 处；首步全 vocabulary logits 在前两个案例中仍逐位一致，
第三个案例的全部 248320 项已不同。
源码的 split-KV 规划由容量决定，128 与 8192 选择不同的 attention 组织；
结合上述控制，这说明统一小投影算术并不足以统一两条实际路径。

未经干预的默认 pooled 则在第一层 LinearAttention state 已出现差异，
三个案例的 state/有效 KV 差异分别有 38,127,153、38,068,079、38,120,767 处。
其首步 hidden 全 5120 项及 logits 全 248320 项均不同，尽管该步 argmax 相同。
因此后续应同时验证小投影与 attention 的路径一致性，并按原完整质量和配对
性能门槛验收；此前质量失败的 pooled F32 单项方案仍不准入。

准备过程中曾错误设置 verification width 为 1，被加载校验拒绝，修正为 MTP0
对应的 0；同容量 128 的早期实验得到 index 11、15、无分歧，没有复现服务尺寸。
首次实际容量实验把“F32 小投影应消除所有差异”作为断言，因 831,825 项差异
失败。这些源码、构建、失败与原始记录均保留；最终实验增加同容量控制，
单独测量容量干预，没有将该失败重写为通过。最终三案例共 75 条数值事件，
GPU guard 无干扰，protected 单元峰值 4.7 GiB、自身 swap 为零。
[构建、逐位 census、逐 token 服务复现与失败记录](artifacts/cuda-gdn-route-arithmetic-census-2026-10-14.json)。

## 2026-10-14：统一小投影与实际 KV 长度的数值原型

先在独立归档中强制 attention 单分片，并同时选用 private 算术的小投影：
三个实际 52 token 输入，每个连续 24 步，首步和后续共 72 次完整比较的
hidden、全 vocabulary logits、全部 F32 state 与所有有效 FP8 KV 全部逐位一致。
这只是容量差异的受控干预；全局关闭 split-KV 不能作为长 KV 的生产方案。

随后保留原有最大捕获 grid、scratch 和 merge，根据实际已用 KV 长度选择
有效分片。短于现有 1024 token 阈值时使用一个有效分片；达到阈值时按现有
每分片至少 128 token 的规则，从捕获的分片上限按二次幂下调。该上限仍受
硬件并行度与容量约束。因此容量提供安全上限，已用长度决定实际归约组织。
这没有新增用户开关、分配或额外捕获图，也没有全局取消长 KV 并行。

先验证短 KV 的动态单分片，再验证上述完整长度规则；两版在相同三个完整
模型输入的 72 次比较中均逐位一致。完整规则另用四种 head/dimension 几何、
11 个因果位置/窗口场景、1/2/8/16 个捕获分片和 F32/FP8 KV，进行 352 组
独立 F64 对照，全部满足原 2e-5 绝对误差门槛。新增位置覆盖阈值前后一个
token 及 2048 长度，未改变误差门槛。

减少计算量的小投影方向也有明确形状依据：当前四槽输入走 16 行 tile，
F32 激活拆为两个 BF16 分量后分别执行 MMA；四行 F32 核保留 private 的
累加树，减少补齐行和两次点积的工作。旧的小投影隔离计时曾显示热缓存
约 1.81 倍速度，但旧 F32 单项方案完整质量失败，该旧数据不是新联合方案
的性能或质量验收。

本轮原型仍只在测试归档中启用 F32 小投影；生产默认和量化 GDN 准入未改动。
上述完整模型只覆盖三个短输入；独立 F64 边界对照补充 attention 覆盖范围。
仍需验证其他 pool 宽度、inactive/rebinding、长上下文、冻结 256 题质量，
以及无 profiler 的独立配对 serving 和持续混合负载。动态分片仍保留最大
grid 和 merge，功能通过本身不证明速度改善。全部版本、构建失败、原始
事件、冻结输入/cache 与二进制身份保留。
[实际长度原型的数值证据](artifacts/cuda-gdn-covered-kv-prototype-2026-10-14.json)。

## 2026-10-14：联合原型完整质量失败，转向保留原算术的小投影布局

联合方案已做成独立 release CLI，符合条件的 2/3/4 行小投影直接采用 F32，
没有实验环境变量。严格 release Clippy、测试和 CLI 构建通过。完整模型检查
扩展至 2/3/4 槽，每种三个实际输入、24 步，共 216 次 hidden/logits/F32 state/
有效 FP8 KV 比较逐位一致；inactive 与 release/rebind 的八轮检查也逐位一致，
覆盖 5,575,680 项输出及完整 state/FP8 KV。这里的 private 控制运行在候选的
attention policy 下；旧测试的 private_arithmetic_unchanged 标签不能证明容量
1024 的原 private attention 算术没有变化。

该 release 候选随后完成冻结 256 题单请求质量集，保留 512 token 上限、MTP0、
32768 context、原 scorer/冻结表，全部请求完成，测得 prefix reuse 为零。
五个达到长度上限的样本保留，没有删去。候选为 240/256，固定 vLLM 为
243/256；相对 vLLM 有六项退步、三项收益，95% 退步率上界 4.5733%，
超过原 2% 门槛，质量失败。因此未做默认切换或性能验收。

相对旧 clean49bd native 记录，两边总分虽同为 240/256，仍有三项退步、
三项收益，109 条完整 token stream 不同；其退步率上界 3.0008%。
这是历史 native 的诊断对照，尚未运行新的当前 native 质量进程，不能称为
fresh paired 验收。由于固定 vLLM 的硬门槛已经失败，没有继续消耗并发质量
和正式性能验收。所有错误项、输出上限样本、输入、二进制与原始记录保留。

重新查询原始 Nsight sqlite，小投影的布局差距有直接时间证据：short 的
private decode 共 6048 次 dense_entry、grid[48,1,1]，总计 12.619115 ms，
平均 2.0865 us；batch4 的 pooled 共 6048 次、grid[1,2,1]，总计
105.176557 ms，平均 17.3903 us。这是原始插桩 trace 的 kernel 均值，
不是新方案的配对延迟验收，也没有据此声称测得 occupancy。

下一独立原型保留 pooled 的 BF16 high/low 分量、逐 K 块 low 后 high 的 MMA
与原 KV/private/GDN 路径，仅将 output tile 从 32 缩至 8。48 输出列的
grid 从两个 block 变为六个，补齐输出列从 64 变为 48。先验证 96 个真实
checkpoint 投影逐位一致，再做冷/热缓存配对计时。实际 8 列版本在第一个
投影的 192/192 项上不逐位一致，计时前即停止；进一步测试的 16 列版本
也在同一关失败。反汇编显示两者均没有 HMMA，分别出现普通乘加展开与
176/1736 bytes 的 STACK；原 32 列版本有 32 条静态 HMMA.16816.F32.BF16，
STACK 为零。这是静态生成代码，不是动态指令或 occupancy 测量。两种布局
均不准入，也没有继续采集性能。下一步保留原 16×32 MMA tile，探索复用同一
输入的 BF16 high/low 分解，消除重复转换而保持原 MMA 累加顺序。
[完整拒绝记录与原始 profile 查询证据](artifacts/cuda-gdn-covered-kv-quality-rejection-2026-10-14.json)。


## 2026-10-14：复用 BF16 分解保持逐位一致，但 serving 尚未准入

保留原 16×32×128 tile、两个 BF16 输入分量和每个 K 块先 low 后 high 的 MMA，
同一归一化输入的 beta/alpha 共用一次分解。独立 release 归档默认执行此路径，
没有用户算法开关；main 尚未采用。持有的 high/low 缓冲区按实际 arena 预算
检查，捕获图先于缓冲区释放，两种 slot 捕获重置输入复用记录。

96 个真实 checkpoint 小投影的 192 项输出逐个逐位一致，四行普通输入的
40,960 个 high/low 分量也与独立 CPU BF16 转换逐位一致。完整模型另从相同
private 前缀复制 state/KV，覆盖三个实际输入、2/3/4 槽、八步固定输入轨迹、
inactive 和 release/rebind：204 次完整 hidden/logits 输出、216 次完整 F32 state/
FP8 KV 检查均无差异。这没有覆盖全部任务质量或任意服务路由。

先做五个 discovery 进程，再做五个新进程交替先后顺序的确认。每次候选 replay
实际包含 48 次分解与 96 次 MMA，全部原始计时保留。确认的冷缓存中位数
1.676384→1.212096 ms，配对速度 1.38299 倍，95% 区间 1.38264–1.38411；
热缓存 1.677376→1.199328 ms，1.39834 倍，区间 1.39750–1.39876。
这是小投影隔离成本，不是服务收益。

候选 release Nsight trace 的 batch4 出现 6048 次 dense_prepared 和 3024 次
prepare_components，合计 74.987835 ms，确认生产形态二进制实际执行新路径。
该插桩计时没有配对，不能与旧 trace 105.176557 ms 作为正式延迟验收。
静态 SASS 保留原 32 条 HMMA，MMA 内 FADD 从 128 条变为零；shared memory
从 52,340 增至 68,756 bytes，REG 仍为 255。没有动态硬件计数器，不声称测得
occupancy 或实际执行指令量下降。

无 host/GPU profiler 的五组独立进程服务配对全部完成，两边各 105 条测量
stream，64 token 上限、四个冻结场景、三次测量重复、固定输入/tuning、MTP0、
65536 context、prefix cache 与原来相同，启动顺序交替。
batch4 TPOT 进程中位数 18.93510→18.42998 ms；配对速度中位数 1.02804，
95% 区间 1.01746–1.03691，低于原有下界至少 1.02 的有用收益门槛。
整体完成时间 1375.76025→1341.15097 ms，配对区间下界也仅为 1.00648。

105 条配对 stream 中有九条 token 不同，全部位于 batch4，严格输出门槛失败。
相同冻结 case/repeat/slot 在 unit0 与后四个进程之间，旧路径有五条、候选有
八条差异；每边为 84 条同输入对照。这提示既有路由波动，但不能把九条差异
自动认定为无害，也不作接受新优化的豁免。short TTFT 两边进程漂移分别
16.58%、13.98%，hot_long 候选漂移 13.12%，均超过原 10% 门槛。
没有删样本或改变门槛，没有宣称无 profiler 服务验收通过。完整 256 题质量
和持续混合负载尚未为此候选执行，不能以之前其他方案的记录替代。

原 plan 沿用了旧模板的 units=3、ordered 顺序名和 route-profile 描述；
实际 run.py、十个 invocation 与 progress 均记录五组 scalar/candidate，并确认
没有启用 profiler。原 plan 保留，分析与本证据显式注明这些继承字段。
当前 native 控制二进制的构建补丁与 aa83 提交逐字节一致，aa83 至本实验
54aafe7 的 crates/tools/bench 差异为空，main runtime 控制有效。

下一步仍沿减少重复工作的主线，把相邻共用输入的 beta/alpha 投影联合为
96 列。原来两个 48 列输出各补齐到 64 列；保持 32 列 MMA tile 后，联合输出
只需三个 block，减少一个 block 的补齐工作。必须同时计入输出拆分成本，
先验证逐位一致，再验证完整模型和原服务/质量门槛。
[逐位验证、微基准、dispatch 与完整服务失败记录](artifacts/cuda-gdn-bf16-prepared-2026-10-14.json)。

## 2026-10-14：联合 beta/alpha 投影消除重复转换与补齐计算

本轮沿 GDN 小投影减少重复工作，保留原精度和累加树。两个投影共用同一输入，
利用 concat(X·W_betaᵀ, X·W_alphaᵀ) = X·concat_rows(W_beta, W_alpha)ᵀ，
在建图时按原 BF16 bits 合并权重。原来两个 48 列输出分别补齐至 64 列，
联合 96 列仅需三个 32 列 block，名义补齐列工作减少 25%。输入 high/low
分解也共享；原 16×32 MMA、K128、每块 low 后 high 的顺序保留。

独立 release 归档中，2/3/4 槽的相邻、共用输入、48 输出 Dense BF16 投影
默认联合，不包含激活量化或常量输入，没有用户算法开关。private GEMV、
GDN recurrence、KV split 和 verification 路径保持原实现；预算不足时保留
旧 prepared 路径。新增联合权重与全部 high/low、联合输出缓冲区计入原 arena/
prepared 工作区预算。只配对相邻、不重叠的节点，并检查两个输出 slot 与输入
不别名；图的释放早于其持有的权重和缓冲区。该形状限定仍是实验范围，main
runtime 未合入。

每个 replay 包含 48 个 prepare、48 个联合 MMA、48 个 split，总计 144 次，
与上一 prepared 方案相同，不能声称总 kernel 启动减半。昂贵 MMA 从 96 次
减至 48 次；原始未准备路径只有 96 次投影启动。计时包含全部输出拆分成本，
不包含只在建图时执行的权重合并，也不包含模型其他工作。

先做一轮 discovery，再以五个新独立进程交替先后顺序确认。每个进程都在
全部 48 对真实同层权重上验证 zero/ordinary/sparse/wide_exponent 四类输入，
192 次、73,728 项投影输出逐位一致。确认的冷缓存中位数
1.677824→0.664880 ms，配对速度 2.52374 倍，95% 区间 2.52053–2.52645；
热缓存 1.668864→0.654736 ms，2.54969 倍，区间 2.54823–2.55332。
全部原始向量保留。这些只是完整小投影 replay 的隔离结果。

实际建图后的完整模型覆盖三个 52 token 输入、511 token 输入和完整
3647 token 输入，每个采用 2/3/4 槽，连续八步并覆盖 inactive 与 release/rebind。
从相同 private prefix 的完整状态复制后，340 次完整 hidden/logits 输出比较、
360 次完整 F32 state/FP8 KV 比较全部逐位一致。长输入按正常 64 token 分块
预填充，仅最后块读 logits；每个候选 slot 图断言捕获 48 个 prepare、48 个
联合 MMA，避免 fallback 造成伪通过。完整模型测试与 release CLI 均严格
Clippy/构建成功，GPU guard 无干扰，数值测试 protected 单元峰值 4.7 GiB，
自身 swap 为零。这不能替代全部质量集或任意服务路由的验证。

首次集成的权重拷贝 kernel 被 host launch 上限拒绝：96 行目标 grid 的条件
被用于只有 48 行的源。修正为有界 pid.0 % N 源行索引，保留编译器检查。
下一轮全部短输入比较通过后，新增长输入被 width64 的直接 prefill 接口拒绝；
整轮仍保留为失败，改用正常分块重跑。两次准备性 Clippy 修复及一次类型
qualification 修复也保留原始源码和构建记录，未将失败改写为成功。

反汇编确认联合 MMA 仍有 32 条静态 HMMA.16816.F32.BF16、REG255、
STACK0、SHARED68756；split 为 REG18、STACK0、SHARED0。
这是静态代码证据，不是 occupancy 或动态硬件计数测量。

随后完成无 host/GPU profiler 的五组、十个独立服务进程，交替启动顺序，
每个进程运行原四个冻结场景、三个测量 repeat，64 输出上限、65536 context、
MTP0、固定46条 tuning与原 token 输入不变。每边105条 stream 全部完成。
batch4 TPOT 的进程中位数 18.905391→17.927086 ms，约下降5.17%；
配对速度中位数1.05457，95%区间1.05437–1.06331，原下界1.02门槛通过。
整体完成时间1385.720685→1313.646923 ms；其配对速度中位数1.04838，
区间1.03710–1.05826。完整每进程、每场景、原始stream与telemetry均保留。
其余场景没有超过5%的配对中位数回退，但short TTFT置信区间不能排除5%
回退，不能因此宣称全部场景均完成无回退认证。

整体准入仍失败：12/105条配对token stream不同，全部位于batch4。
short 与 hot_long 的基线 TTFT 进程漂移分别15.22%和12.74%，超过10%；
候选相应漂移6.22%和8.71%。相同输入在unit0与后四个进程之间，每边84项
比较，旧路径有19条、候选有11条输出差异。这些波动保留，没有删样本。

九条候选差异的完整64-token序列能在本轮其他基线进程找到。另对照上轮五个
相同基线二进制、输入、tuning、65536 context的进程，仍只有九条匹配：
unit0的batch4-repeat1-slot1、unit2的同输入和repeat2-slot1三条尚未匹配。
这是额外的观察性census，不是扩充或替换本轮五组统计样本，也不是路由的
因果证明。原始文件输入hash2d88...与harness规范化输入hash855e...区分记录；
扩展查询曾把二者直接比较而拒绝，纠正hash类型后才做逐序列观察。原有12条
失败仍保留，不能用受控逐位一致或基线自身波动为其豁免。

因此没有切换main默认，没有以小算子的2.5倍或batch4的约5%代替整体通过，
也没有用其他候选的256题结果替代本候选单请求/并发质量及持续混合负载。
下一步应对上述实际请求关联state/slot/初始private与pooled历史，从相同输入
和实际状态路径重放，区分布局数值、迁移和调度，避免未经定位继续更改算术。
[完整数值、性能、失败与逐序列census](artifacts/cuda-gdn-bf16-joint-2026-10-14.json)；
[可审阅实验补丁，未合入runtime](artifacts/cuda-gdn-bf16-joint-prototype-2026-10-14.patch)。


## 2026-10-14：完整重放定位首次路由，兼容活动请求原型消除冻结场景差异

上轮三条未匹配的输出集中于 batch4-repeat1-slot1 的两次进程和
repeat2-slot1 的一次进程。新两组、四进程 host 诊断将实际 request/state/slot
关联：原 scalar 的 repeat2-slot1 在位置52与三个prefill任务同组，执行private；
位置53起pooled，其完整64-token序列与此前候选的一条差异序列一致。
源码run_grouped只合并连续eligible decode段，未绑定的普通decode段不足两条
会退出slot分配并执行private。已有pool在load时捕获，故这里是准入条件的
路由选择，不是新联合投影算术本身的数值证明。

随后用原自动prefill配置、私有请求116/实际容量128、pooled容量8192，
对上述两个实际52-token输入做完整64-token贪心重放。两种历史分别为直接
pooled，以及先private一步再逐位复制迁入pooled；每种在两个不同物理slot
重复。对相同初始状态和相同历史，原96MMA与48联合MMA的500次完整
hidden/logits、504次完整F32state/FP8KV比较全部逐位一致；16次迁移检查
的完整F32与有效FP8也无差异，物理slot位置不改变该历史的64-token序列。
两种历史首个token差异分别在index20和12。全部三条此前未匹配候选的
完整64-token输出均匹配private一步历史，配对原native均匹配直接pooled。
这是受控干预和完整序列复现；原无profiler记录没有初始路由日志，不能
倒推为已记录了其实际历史，也没有豁免原12/105输出门槛失败。

隔离release原型使一个ready decode在存在其他兼容活动Logits状态时也可bind
到已捕获pool，包含其余请求还在prefill的情况。poisoned和非Logits状态不作为兼容peer，未绑定的超容量状态也排除；
已持有slot的状态视为兼容。只有一个兼容活动状态仍用原private路径，已有slot lease
在peer结束后继续pooled。两个控制源都用同一路由规则，一个保留原96MMA，
一个执行48联合MMA；没有用户算法开关，原精度、KVsplit和GDN未变。
两边严格release Clippy/CLI构建成功。两组host诊断42条配对输出全部相同；
每边unit0与unit1的21条同输入输出也全部相同。全部48条batch4诊断stream
从首次decode就pooled；36条单请求stream没有pooled decode。该诊断不作计时验收，
late arrival、异构长度/不兼容peer与持续混合负载仍待验证。

再完成无profiler的五组、十个独立进程，原四场景、输入/tuning、三次测量重复、
64输出、MTP0、65536context、prefix cache与原门槛不变。每边105条stream
全部完成，配对输出差异为零，每边unit0与后四进程的84项同输入比较也全相同。
batch4 TPOT进程中位数18.875866→17.911129ms，约下降5.11%；配对速度
1.05606，95%区间1.05379–1.05716，原有下界1.02门槛通过。
整体完成时间1377.713982→1317.443578ms；配对速度1.04766，
区间1.04395–1.05323。这是相同实验路由下联合投影的隔离服务对照；
当前main原始router的12条失败及质量参考仍保留，不以新控制替换。

整体仍未准入：short TTFT控制/候选进程中位数44.376119/47.287863ms，
配对速度中位数0.93843，即候选配对中位数约慢6.56%，超过原5%门槛。
其控制进程漂移17.05%，hot_long候选漂移12.37%，超过10%。其余TPOT和
整体时间没有>5%的配对中位数回退；短请求总完成时间稳定不能代替TTFT。
没有删样本、改变门槛或据private代码未改就把回退认定为无害噪声。
本原型的完整256题单请求/并发质量、主分支native和固定vLLM对照尚未完成。

为定位TTFT，在独立构建复用已有INFER_CUDA_EXECUTION_PROFILE，增加state
reserve、prefix reuse生命周期，以及执行和客户端起止UTC。两个版本Clippy/
release构建通过，按相反启动顺序完成两组诊断。生命周期span不记录成功，
不能把span结束单独当成功证据。单请求36个样本的时间分解闭合误差最大约
3.06us；batch4共享组时间不能独占归因到某请求，未进入单请求分解。

short的prefill执行中位数控制36.241、候选36.405ms，接近；客户端请求到reserve
开始为3.352/4.152ms，reserve本身2.333/2.509ms。hot_long reserve为2.190/3.112ms，
prefix reuse为1.145/1.569ms。各阶段中位数不可直接相加当总TTFT，原逐请求
记录与闭合残差均保留。日志会改变调度，这些结果只支持下一步细分准入/准备
过程，不能替代未插桩的5%回退和10%漂移门槛。main默认仍未切换。
[受控重放、原型路由、完整服务失败和TTFT边界证据](artifacts/cuda-gdn-cohort-route-replay-2026-10-14.json)；
[可审阅路由原型补丁，未合入runtime](artifacts/cuda-gdn-cohort-router-prototype-2026-10-14.patch)。


## 2026-10-14：预算公式消除反复图规划，短请求 TTFT 降低但原路由门槛仍失败

TTFT 边界诊断后继续 profile before optimization：用当前 main 75dd 的独立
release 归档，实际 Qwen3.8-27B-NVFP4，自动 prefill 256/64、MTP0、
1,155 个 node、2,134 个 tensor，先测未修改的 sequence_budget。四种实际
容量 × Logits/Full × private/shared 共16组，每组64次，CPU中位数
796.085–805.050us。服务准入与 reserve 反复询价；原函数每次重新遍历图、
规划 activation 生命周期并汇总投影/递推空间。该计时只量 CPU 定价，
不能单独解释客户端 TTFT 的全部准备耗时。

按原公式，只有 KV 和 Full-history CPU 读回随容量线性增长，其他几何在
LoadedModel 加载后固定。隔离原型将程序存储价格写为 fixed + per_token ×
rounded_capacity，最多复用四组 private/shared × Full/other 成功价格。
容量仍先检查 1..=32768，再按原 next_power_of_two；所有组合与乘加均 checked。
失败价格不缓存；并发首次询价可重复计算相同不可变价格，没有全局锁或用户
算法开关。StatePool 的 retained.max(fresh)、实时 physical_available、active
budget、prefix/state eviction 及 reclaim barrier 完全保持原实现。

MTP0 与 MTP3 分别将所有16个二进制容量档位及相邻输入、None/Logits/Full、
private/shared 与 verbatim 旧公式逐项对照，每边258项有效报价全部相同；
0、32769、usize::MAX 的错误文本也相同。独立 CPU 基线的16项字节价格全部
相同。10项预算/递推存储测试通过，包含乘积溢出和总和溢出。首次四组报价仍
约0.8ms；复用报价的逐次样本约0.010–0.011us，接近此微基准计时开销，
不据此宣传几万倍服务加速。严格 release Clippy、测试及 CLI 构建成功，
测试/模型单元峰值4.8GiB、swap0、GPU guard无干扰。

随后五组十个新独立进程按固定交替顺序，保留原四场景、输入/tuning、
三次测量重复、64输出、MTP0、65536context、prefix cache，无host/GPU
profiler。控制仍为原main runtime ca5；aa83..75dd 的 crates/toolsbench
差异为空，候选仅改预算公式，没有 cohort router 或联合投影。
short TTFT进程中位数45.715304→39.852980ms，约下降12.82%；配对速度
1.15159，95%区间1.13917–1.17228，有用收益下界1.02门槛通过。
long TTFT239.343486→233.410671ms；配对速度1.03028，
区间1.02458–1.03506。单请求short/long/hot_long的45项完整配对stream
均一致；所有场景均没有>5%配对中位数回退。

整体仍未准入：每边105条stream全部完成，但18条batch4配对输出不同。
unit0与后四进程的84项同输入对照，原路径10条、候选9条差异。
候选batch4 TTFT进程漂移10.989%，超过原10%门槛；其他case/metric未超。
GPU算术未变支持继续调查原路由对CPU时序的敏感性，不能把18条差异
自动豁免或断言每条已获得因果证明。原profile与完整受控重放的private/pooled
历史证据仍保留。下一步在同一cohort路由下组合联合投影与预算公式，检查
已测得的batch4 TPOT收益和短请求TTFT是否同时成立。完整256题单请求/
并发native及固定vLLM质量、持续混合负载仍待验证。main默认runtime未切换。
[CPU原始样本、精确对照与五组完整服务记录](artifacts/cuda-admission-affine-2026-10-14.json)；
[可审阅预算公式原型补丁](artifacts/cuda-admission-affine-prototype-2026-10-14.patch)。
