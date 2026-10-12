# ReplaySSM：F32 GDN 缓存与推测解码探索

本记录回应 [Dao AI Lab ReplaySSM](https://dao-lab.ai/blog/2026/replayssm/) 方向，
研究基点为 inference.rs `46d0144`。第一轮已经完成隔离 GPU 数值/成本实验；
没有把原型接入 native serving，当前没有新的完整模型性能或质量资格。

## 源码基点与算法选择

45 上的官方 vLLM main 为 `cc9b0f097d49abfa6fc94d1d773dca1cc2b9619c`。
该版本的 ReplaySSM 注册/测试目前覆盖 Mamba2，不能把存在 `--use-replayssm` 解释为已支持
本项目的 GDN 模型。作者 [参考仓库](https://github.com/Johnny-Liou/ReplaySSM) 已在 45 拉取，
冻结 commit `5d3ea597d479ef0c8e9e2eb2845a2adfdb9a8b44`；README 明确区分已进入 upstream 的 Mamba2 standard 与其他路径。
[GDN standard](https://github.com/Johnny-Liou/ReplaySSM/blob/5d3ea597d479ef0c8e9e2eb2845a2adfdb9a8b44/vllm/third_party/flash_linear_attention/ops/fused_recurrent_replayssm.py)
及 [GDN speculative](https://github.com/Johnny-Liou/ReplaySSM/blob/5d3ea597d479ef0c8e9e2eb2845a2adfdb9a8b44/vllm/third_party/flash_linear_attention/ops/gdn_replayssm_spec_decode.py)
仅作参考，不修改官方 checkout，也不把作者 fork 当作现有官方 vLLM 基准。

对于本项目的 Delta recurrence，缓存每步校正向量 u、归一化 K 与 gate，而不是只保存原始 V。
u 依赖该步之前的状态；漏掉这一依赖会改变 GDN 算法。令 checkpoint 为 S0，近期有效项为
`(u_j,k_j,g_j)`，状态可表示为 decayed S0 与加权 rank-one 项之和。
普通 decode 只需要该状态对当前 q/k 的作用；本次 F32 readout 原型把它展开为两个
checkpoint matrix-vector readout 加上历史 key 的内积与 u 的加权和，非 flush 步不写完整状态。
到边界才物化完整 F32 状态，随后清空历史窗口。该展开在实数代数上相同，浮点求值顺序仍需独立验收。

这与 [主机前缀缓存](cuda-memory-for-latency-2026-10-12.md)作用不同：这里的工作窗口常驻 GPU，
不能逐 token 从 RAM 恢复。RAM tier 可保存较冷的完整状态边界，两个方向可以组合。

作者 `low_precision/gdn/fp32.py` 虽以 fp32 命名，仍使用 FP16 ring、BF16 输入/输出与架构相关
TCGEN05 路径，不能直接符合本项目的 F32 activation/ring 契约。本轮不使用它，不引入 TF32 或
低精度 checkpoint。作者 standard Triton 的两处 F32 dot 在独立 adapter 中明确指定 IEEE；
另一个独立 F32 readout 原型全部使用 F32 reductions/rank-one 运算。

## 独立数值与完整周期成本

几何固定为 `KH=16,VH=48,K=V=128`，window=4/8/16/32；state、activation、u/K/g 缓冲、
输出均为 F32。独立 CPU F64 参考直接从原始 F32 输入构造，保持既有 `max_abs<2e-5`。
普通输入与弱衰减/相关 K（a_log=−10、beta input=8）均覆盖多个 flush、尾部与中途强制 flush。
baseline、作者 IEEE adapter、F32 readout 共 24 组检查全部通过。最大 output/state 误差为
`5.7369923e-9 / 6.7220303e-8`；每个非 flush 步还检查 checkpoint 逐位不变。
这不包含完整模型质量、变长 batch、slot 迁移和 prefix 边界验收；推测路径的后续独立检查见下文。

成本实验使用 CUDA Graph，测量完整 window 周期，包含 flush；每项五轮 AB/BA 顺序，
每次 12 次 graph replay。baseline 为参考仓库 packed GDN scalar kernel，
不是 inference.rs kernel。三种 arm 使用相同 F32 输入，编译/分配在计时外。
baseline 使用默认 BV=32、warps=1、stages=3；作者 IEEE adapter 为 BV=32、warps=1、stages=3、NK=2，
F32 readout 为 BV=32、warps=4、stages=1。除历史窗口外尚未穷尽 launch 配置，收益包括计算方式与布局差异，
不能全部归因于 state 写回；后续需对优化过的 native baseline 直接配对。
一层 footprint 容易受 L2 复用影响，因此另测 48 层独立状态链，避免把单层 warm kernel
结果直接当作模型；该链未包含投影权重、attention、norm、卷积或 sampler。

| 工作量 | 历史窗口 | baseline µs/层/token | F32 readout µs/层/token | baseline/readout |
|---|---:|---:|---:|---:|
| 48 层独立状态链，batch 1 | 4 | 6.164 | 5.062 | 1.218 |
| 48 层独立状态链，batch 1 | 16 | 6.114 | 5.723 | 1.068 |
| 48 层独立状态链，batch 1 | 32 | 6.149 | 6.668 | 0.922 |
| 48 层独立状态链，batch 4 | 8 | 15.886 | 11.988 | 1.325 |
| 48 层独立状态链，batch 4 | 16 | 15.847 | 12.525 | 1.265 |
| 48 层独立状态链，batch 4 | 32 | 15.899 | 14.774 | 1.076 |
| 1 层独立状态链，batch 16 | 4 | 24.191 | 16.134 | 1.499 |
| 1 层独立状态链，batch 64 | 16 | 271.500 | 147.111 | 1.846 |

作者 IEEE adapter 的 48 层 batch=1 各 window 均慢于 baseline（speedup 0.857/0.882/0.890/0.615）；
不能在本项目直接照搬其计算布局。F32 readout 在同一工作量最高 1.218 倍，batch=4 最高
1.325 倍；随着 window 增大，flush 与历史计算成本抵消省下的写回。
batch=64 的一层测试最高约 1.85 倍，但不代表验证机能同时容纳 64 个完整 27B 请求。
五轮都来自同一进程，不能当作五个独立服务进程的统计认证。

window=4/8/16/32 的 F32 ring 在 48 层每序列分别约 6.035/12.070/24.141/48.281 MiB。
这是少量显存换较少状态写流量，不是整份 state 消失：checkpoint 仍需 144 MiB。
模型中每多一份完整 recurrent checkpoint 又增加约 144 MiB/sequence；
推测解码减少这类快照有较大的容量价值，但尚未测得 native 并发/吞吐收益。

受保护运行约 23.6 秒，cgroup memory peak 约 1.3 GiB、swap=0；其他 GPU 进程保留。
冻结 runtime 使用官方 Torch 2.13 CUDA 环境。试验只检查外部 compute，不使用服务 benchmark
的外部图形显存严格一致性资格。首个 attempt 因相对包导入失败，在数值/计时前停止；
修正模块加载上下文后完成，原源码、失败日志与新 manifest 均保留。

## 推测路径的拒绝／提交与跨界数值检查

作者冻结 GDN speculative wrapper 保持 F32 state/activation/ring，并显式设置 verify/flush 的
dot precision 为 IEEE；没有沿用作者测试的 `5e-3` 容差。独立 F64 参考从原始 F32 字节构造，
在被接受的流上保持 F64 state，避免将 GPU rounding 回灌给参考。

真实 `KH=16,VH=48,D=128`，验证窗口 T=3（对应两个 speculative token 的验证宽度），
历史容量 8/16，普通与弱衰减/相关 K 共四组；每组 40 次验证，接受计数依次重复
`1,3,2,0,3,1,2,3`。四组均检查 120 个位置的输出，包含每组五次全拒绝、多次提前 flush、
ring wrap 与最后一段未物化历史；device commit 的 cursor/base/下轮 flush 标志逐步核对。
checkpoint 只在 flush 改变，slot 0 保持不变。

四组全部通过原 `max_abs<2e-5`；最大 output 误差 `1.2250533e-8`，
flush 后 checkpoint 误差 `1.2787037e-7`。这证明隔离 GDN 的候选/接受计数状态表达可行，
不代表 native MTP 整体回滚合格：卷积/KV/hidden/draft、变长并发、取消与迁移仍需自己的测试。
GPU cost 尚以独立 prototype 筛选，不替代完整模型质量。

原始 `spec-numerics-results.json` SHA256 为 `1945eefdb415c9a78c5ba7ba877cbfd75c616492b86b6f3a3a8de435e475076e`，冻结脚本与接受 pattern 见
`spec-preparation.json`；原始成功日志 `spec-run.log`，运行约 7.9 秒、memory peak 807.7 MiB、swap=0。

## 推测验证的完整周期成本筛选

增加独立 scalar checkpoint twin：保持参考 packed Delta 的算术，同时写 live state 和
每个验证位置的完整 snapshot，模拟本项目 [checkpoint twin](../../crates/backend/cuda/src/resident/recurrent_checkpoint.rs)
的两次写回。新增 writer 先通过原 F64 `2e-5` 输出/state 检查。它是 Triton 参考实现，
不是 Rust backend 的实际测量，不能把算子时延直接当成 native 绝对成本。

验证窗口 T=3，all-accept 的三项均提交；Replay 测量包含两类 graph kernel 的 runtime routing、
完整周期中的 flush，以及一次 device commit。每份图包含 24 次验证，五轮交替顺序，
每项五次 replay，按每层/验证窗口报告。历史容量 8/16，物理 pow2 ring 16/32。
未包含 draft model、sampler、卷积/KV/hidden、主机调度和 prefix，未搜索全部 launch 配置。

| 独立状态链／batch | history／物理 ring | scalar+checkpoint µs/层/窗口 | F32 IEEE replay µs/层/窗口 | speedup |
|---|---|---:|---:|---:|
| 48 层／1 | 8／16 | 16.403 | 20.951 | 0.783 |
| 48 层／1 | 16／32 | 16.422 | 21.449 | 0.766 |
| 48 层／4 | 8／16 | 42.392 | 30.235 | 1.402 |
| 48 层／4 | 16／32 | 41.270 | 30.291 | 1.362 |
| 1 层／64 | 8／16 | 1234.565 | 375.795 | 3.285 |
| 1 层／64 | 16／32 | 1237.698 | 338.362 | 3.658 |

单请求的作者 IEEE 推测布局在 48 层链上约慢 28–31%，不能直接替换全部 decode。
四并发则约加速 1.36–1.40 倍；一层 batch=64 的 3.29–3.66 倍不外推为完整 27B 服务收益。
这是明确的 workload 取舍：普通 decode、小 batch 的直接 F32 readout 与批量 verify 需独立调优。
GDN-only 层的 T=3 三份完整 snapshot 约 432 MiB/sequence；F32 ring 在本次物理容量
16/32 时为约 24.141/48.281 MiB，checkpoint、conv/KV 等仍需保留。
实际并发能否增加要核对 resident budget、graph/workspace 和 admission 后再测，当前没有 pool slot 收益结论。

原始 `spec-cost-results.json` SHA256 为 `13f37af99c6432b8aee18ab7222b97590a8547b59a55e3fe5867b3753069e2ee`，
writer/launch/runner 身份见 `spec-cost-preparation.json`，每次样本与原始日志 `spec-cost-run.log`
完整保留。受保护运行约 12.6 秒，memory peak 676.7 MiB、swap=0。

## 接入本项目需要的状态契约

[单步 Delta](../../crates/backend/cuda/src/resident/recurrent.rs)每 token 写回完整 state；
[pooled checkpoint twin](../../crates/backend/cuda/src/resident/recurrent_checkpoint.rs)还将每个
验证位置写入完整 snapshot。Replay 适合首先替换这部分数据表达，复用已有稳定三角 WY
处理窗口内校正向量；不能用 Mamba 的 V/K 缓存公式直接替换 GDN。

1. resident 资源保存 checkpoint、有效校正窗口与 device cursor。logical position 覆盖
   已接受 token，checkpoint position 可以落后，二者不可再混用。
2. 验证使用此前已提交历史与当前 draft，commit 按真实接受数保留窗口，rollback 丢弃未接受项。
   flush 只能归纳此前已提交历史，不能覆盖仍可能回滚的 draft；跨边界要提前留够下轮验证空间。
3. convolution、KV、hidden 与 draft 状态仍有各自提交/回滚契约。仅正确恢复 GDN 不代表整个模型
   回滚正确。算法策略应由 workloads 控制，CUDA 管理 buffers/graph/设备提交。
4. [prefix snapshot](../../crates/backend/cuda/src/executor/prefix_state.rs)当前按完整 state 捕获。
   RAM/GPU prefix、外部 checkpoint 和迁移边界需先 materialize，或采用显式携带窗口的新格式。
   不能只复制滞后的 checkpoint 后设置最新 position；图必须在对应资源之前销毁。
5. 首先直接对比 Rust baseline、replay standard、replay verify，分别测非 flush/flush/完整周期，
   MTP 接受率与四并发准入；通过原独立数值、边界状态测试和固定完整模型质量协议后，再验收服务性能。

window、value 分区、重建/readout 方式、batch/验证宽度均进入按硬件与工作量的 tuning：
缓存键包含设备/驱动/编译身份、kernel fingerprint、shape、F32 contract 和 schema，
冷/暖配对后确认，失败回退原实现。普通 decode 的最佳 window 不可直接用于 MTP；
spec 的逻辑历史容量与物理 pow2 ring 容量也须分别计价，不能硬编码论文的 16 或验证机的 4/8。

普通单请求的整体提升取决于真实 GDN 成本占比，算子 1.22 倍不能外推为模型 1.22 倍。
当前 native 的长提示 TTFT 差距仍是 prefill 问题，需继续独立优化；Replay 优先解决 decode 和
推测状态写回/快照。隔离拒绝/提交与 flush 跨界的首轮数值检查已通过；下一项验收重点是 native 的整体状态契约、实际成本和完整模型质量。

## 可复算证据

远端目录 `replayssm-gdn-exploration-v1` 保留作者冻结源码、source identity、原型与 adapter、
准备 manifest、失败/成功日志、每次样本与完整 F64 结果。`results.json` SHA256：
`dfa1a5c85260457a707033028042b82df29b9439de13f10cfc109ed9f1d2ff85`。当前全部数据是可行性筛选；没有修改 inference.rs 的默认执行策略。
