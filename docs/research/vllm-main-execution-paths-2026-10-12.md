# vLLM main 执行路径与当前性能差距（2026-10-12）

当前差距涉及 GDN 的计算方式、推测解码的数据流和资源准入；单独改 GEMM tile 无法覆盖这些成本。扩大 prefill 图宽度已降低长提示首 token 延迟，但最近一次有效反向配对没有复现 batch4 收益，热前缀也退化，当前候选不适合整体合并。

本文是源码调查与诊断实验，正式对照和验收沿用[性能基线方案](../plans/performance/baseline.md)。下文区分最新 main 的源码、已安装发行版的实测和 native 候选，避免把源码更新当成性能更新。

## 版本与实测边界

| 对象 | 身份与范围 |
|---|---|
| native main | `b41ddbbcbb6d832fb2cf983c21870226f8ae62c2` |
| native 候选 | 源码树 `43f86a125b095a40e5a5c144750c436db79f6bb0`；包含 compact head、图宽度和 prefix/resident 资源修复，未合并 |
| 旧逐节点 profile | `0dd5899` 源码身份，诊断计时；不替代当前候选的服务计时 |
| 原 vLLM clone | `5550ff9c25b8d6d2ba35572883602423f74fb9bf`，2025-10-16；过旧，不能代表当前模型实现 |
| 首次更新的 vLLM main | `c059bee812842fb76c2b045e240c02f37c0d498d`，2026-10-12 00:57:07 +08:00；已在 45 上通过 `git pull --ff-only upstream main` 更新，工作区干净 |
| 本轮再次更新的 vLLM main | `f26d9f432ced7b1087d02d29bffed2249df1a18e`，2026-10-12 01:55:00 +08:00；再次从官方 upstream 快进拉取，main 已跟踪 upstream/main，工作区干净 |
| 实测 vLLM | 0.31.0；FlashInfer 0.7.0.post1、Torch 2.13.0、Triton 3.7.1、Python 3.12.11 |
| 验证硬件 | 45 上 RTX 5090，driver 615.78.08；实现与参数选择应依赖设备能力、shape、精度和资源预算 |

本轮新增提交只修改 GLM KDA recurrent-state dtype 的三个文件，未修改下文引用的 Qwen/GDN/MTP 路径。源码链接保留首次调查的 c059 提交作为可复核基点；已安装和实测的 vLLM 仍为 0.31.0。另见[GPU draft 采样六组配对](cuda-device-greedy-mtp-2026-10-12.md)：单请求收益约 1.6%–2.0%，未达到默认启用条件。

模型目录名含 Qwen3.8，配置实际是 `Qwen3_5ForConditionalGeneration`：64 层，48 层 GDN，16 层 full attention，hidden 5120、vocab 248320、GDN K/V head dim 128，K heads 16、V heads 48。checkpoint 混用 NVFP4、FP8、BF16，不能把整个计算路径概括为一种精度。native 当前 activation 为 F32，vLLM 为 BF16，两边 recurrent state 为 F32；这是计算契约差异。

已安装版三个关键源码文件与官方 `v0.31.0` 完整字节一致：GPU model runner、autoregressive speculator、Qwen GDN。最新 main 已将 autoregressive 改名为 `target_dependent_ar`，并增加动态草稿长度及其 graph 专门化。NVIDIA GDN 的 FlashInfer 包装函数在两个版本中的 AST 完全一致；新增加的 AITER FlyDSL 路径属于 ROCm。实际 benchmark 仍对应 0.31.0，尚未构建或测量最新 main。

最近通过背景检查的 native p1 配对诊断如下。MTP 深度 2、greedy、输出 64 token；vLLM 服务参考为另一组固定版本诊断运行，不能视为同步配对基线。长提示 511 token，热前缀每请求实际复用 3584 token。

| 指标 | native main（p1） | 候选（p1） | 候选相对 main | vLLM 0.31.0 参考 |
|---|---:|---:|---:|---:|
| short TTFT | 52.83 ms | 52.20 ms | -1.19% | 33.21 ms |
| short TPOT | 11.208 ms | 11.204 ms | -0.03% | 7.976 ms |
| long TTFT | 365.57 ms | 293.24 ms | -19.79% | 85.69 ms |
| long TPOT | 10.904 ms | 10.894 ms | -0.09% | 7.991 ms |
| batch4 完成时间 | 1964.15 ms | 1965.75 ms | +0.08% | 645.40 ms |
| batch4 TPOT | 14.869 ms | 15.967 ms | +7.38% | 8.559 ms |
| hot_long TTFT | 80.68 ms | 88.71 ms | +9.96% | 31.59 ms |
| hot_long TPOT | 10.314 ms | 10.300 ms | -0.13% | 7.709 ms |

p1 的 28 个输出与 main 完全相同，实际热复用也相同。p0 曾测得 batch4 完成时间 -27.91%，但背景资格检查未通过；p1 反向配对未复现该收益。目前只有一个满足背景条件的新进程配对，未满足预先固定的五对要求。长 TTFT 的方向有两轮支持，decode 收益基本为零。鉴于 batch/hot 已出现退化，不继续扩样来追逐这个整体候选的有利结果。

## 1. 长 prefill 的 GDN 需要减少状态工作量

native 的[workspace 选择](../../crates/backend/cuda/src/resident/recurrent_prefill/workspace.rs)在量化权重且未 opt-in 时保持逐 token 的 Delta 实现。每层状态有 `48 × 128 × 128` 个 F32，即 3,145,728 字节；511 token、48 层、每 token 一次完整状态读写的逻辑工作量估算为 154,316,832,768 字节。该数值不等于实测 DRAM 流量，未扣缓存，也未计 QKV、临时结果和其他算子。

单纯把多 token 放进一个 kernel、在寄存器内串行推进 state，仍保留逐 token 递推的主要算术依赖。旧逐节点 profile 的 target Delta 合计约 115 ms；target prefill 从 64 扩到 256 后，Linear 160→84 ms，Delta 115→115 ms。这个不同日期的诊断不是当前服务延迟的精确分解，但解释了扩大 GEMM 图宽度为何无法覆盖剩余差距。

按 native 的 K×V 状态布局，递推可写成：

`H_t = d_t (I - beta_t k_t k_t^T) H_(t-1) + beta_t k_t v_t^T`

分块算法利用块内 QK/KKT、三角求解和矩阵乘法重用状态，块边界再传播状态。这才是需要研究的算法变化。[最新 main 的 GDN 包装](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L234)调用 FlashInfer 分块实现；CUDA backend 选择同时检查能力、head dim 与运行时版本，并保留 Triton fallback。

新补充的独立 GPU 诊断使用实际模型的 heads/dim，单序列、全零初始 F32 state，冷 L2、GPU event 计时：

| token 数 | native 逐 token F32 | native chunk 内串行 F32 | vLLM/FlashInfer BF16 包装 |
|---|---:|---:|---:|
| 32 | 0.171952 ms | 0.072256 ms | 0.049152 ms |
| 64 | 0.335168 ms | 0.131536 ms | 0.049152 ms |
| 128 | 0.661984 ms | 0.252576 ms | 0.065536 ms |
| 511 | 未测 | 未测 | 0.155648 ms |

FlashInfer 一侧 10 次预热、100 个计时样本，用 CUDA graph replay。Torch profiler 确认单序列调用 CP 的 T/MN precompute、fixup 和 prefill kernels；四序列几何则调用 `FullyFusedDeltaRuleSm120`。这些轨迹只证明微基准几何的实际 dispatch，不能推断整个 HTTP batch 必然使用同一路径。

这不是相同算术的正式 speedup：native 为 F32 QKV/output 且包括 gating；参考为 BF16 QKV/output、预先准备 g/beta，并包括 QK normalization 与 g.exp。native callback 在计时内部同步，参考在 end event 之后同步。不同精度、gating 范围和同步方式必须保留。

已有 F32 chunk 与逐 token 实现末位不同，量化模型会放大为 greedy 分歧。workspace 注释将差异归因为编译器 contraction，但当前调查尚未证明这个根因。继续搜索编译选项不能代替分块算法与数值契约验证。BF16/algebraic GDN 应作为独立数值 profile，对照 checkpoint 的 BF16 路径验证 logits、state、长上下文和任务质量；不能拿少量输出相同冒充质量通过。

## 2. 推测解码应让候选、采样和 hidden 留在设备上

native 的[readback](../../crates/backend/cuda/src/device/readback.rs)在 graph 后同步复制到 pinned host，再构造 CPU Vec；[候选与接受](../../crates/backend/cuda/src/executor/execution.rs)、[批量验证](../../crates/backend/cuda/src/resident/slot_verify.rs)读取完整 logits 与 hidden，CPU sampler 选择 token，后续 draft 再上传 hidden。

MTP2 的一轮若读取 target 三行和 draft 两行，每行 `(248320 + 5120) × 4` 字节，仅 logits/hidden D2H 就是 5,068,800 字节/请求，尚未计 catch-up 和其他传输。差距不只来自这些字节，还涉及每个依赖点的 stream 同步、CPU 全词表扫描、分配与 hidden H2D。现在没有双端完整 decode profile，不能给这些因素分摊百分比。

最新 main 的对应机制：

- [TargetDependentAR proposal](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py#L240)保留 target padded query 几何，由 GPU 的 accepted/rejected 数量准备 draft 输入，避免为了知道接受长度先同步 CPU。
- [_prefill / _generate_draft](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py#L468)在 GPU 取所需 hidden 行、采样并复用 hidden。
- [RejectionSampler](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py#L244)在设备上处理 sampling 参数和接受判断。
- [输出与下一轮 proposal](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/worker/gpu/model_runner.py#L2176)先启动小结果的异步 D2H，再提交下一轮 draft。

vLLM 同样需要用 target-conditioned hidden 追赶 draft 状态，不能据此删除 native 的 catch-up/rollback。实际 FlashInfer attention 条件下，多步 draft fusion 也有 fallback，不能概括成全链条无 CPU 或所有 draft 已完全融合。

可保持当前 F32 数学语义的实现方向是 GPU greedy argmax、接受行选择和 hidden 复用。应由 SPI 表达设备驻留结果及验证/提交/回滚契约，让 engine/workloads 保持算法策略的共同所有权。先验证 tie、NaN、penalty 与请求 sampler 的行为，再用小 token/count readback 代替完整 logits；随机采样和需完整 logits 的接口保留已有路径。

## 3. 相同模型的 projection 已被 vLLM 合并

[Qwen GDN](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)使用 q/k/v/z 合并 projection、b/a 合并 projection；[full attention](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/model_executor/models/qwen3_next.py)使用 QKVParallelLinear；[dense MLP](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/model_executor/models/qwen2_moe.py#L76)使用 merged gate/up projection。

按 target IR 的逻辑 Linear 计数，native 是 `48 × 8 + 16 × 7 + head = 497`；vLLM 是 `48 × 5 + 16 × 4 + head = 305`，少 192 个 projection。这是结构计数，不能当成实际 kernel launch 数或 38.6% 的模型性能收益。量化、epilogue 和其他 fusion 会继续改变实际执行。

合并要保留每个 checkpoint projection 的 scale 与 activation 量化规则。特别是 NVFP4/FP8 的全局输入 scale 不能未经证明取最大值后宣称等价。共享输入、拼接权重和切分输出有助于减少重复工作，但应先核对各量化方案的执行契约，再加入 compiler/resident 的可核验设备计划。

## 4. batch4 差距包含资源准备与准入等待

最近通过背景检查的 p1 配对中，候选三个 batch4 trial 的最慢首 token 为 1218、1231、1149 ms；聚合 `resource_preparation` 分别约 1178、1163、1108 ms，`Preparation` 延后计数每轮为 8。vLLM 参考一轮四请求 TTFT 均约 83 ms。

这不能直接解释为 1.2 秒 active CUDA compute，也不能仅凭此判定显存不足。候选的十条 DeviceProgram cold-capture 记录全部位于预热阶段，测量阶段没有这些 cold capture，不能把它们作为测量延后的原因。需要将每次 preparation 的等待、资源预算、slot/graph 复用、创建与回收关联到请求。

候选给约 51/52 token 的短请求也保留 target `[256,64]` 和 draft 128 的几何。扩大激活与图资源可能跨越并发驻留阈值；这是待核验假设。最新 vLLM 的 [graph capture](https://github.com/vllm-project/vllm/blob/c059bee812842fb76c2b045e240c02f37c0d498d/vllm/v1/worker/gpu/cudagraph_utils.py#L439)共享 graph pool 并先 capture 较大 activation 的 PIECEWISE，再利用池内空间 capture FULL；其 packed batch/shared buffers 值得对照。

prefill/verify/draft 宽度、slot 容量、workspace 和 graph 池策略应共同纳入 tuning，候选以完整请求的延迟与并发资源收益评分。构建、quote、cache 命中与分配必须使用相同的实际几何；仅保存算子最快参数会错过服务容量退化。键包括设备能力/driver、模型与 IR、shape、精度、工作区和可用预算，5090 仅承担本轮验证。

## 优化方向的取舍

1. 先使每请求图几何和驻留策略服从实际工作量与资源预算，并补齐 preparation 归因，避免长提示改善拖累短/热/并发请求。
2. 保持 F32 规则，优先减少 greedy/MTP 的同步与完整 logits/hidden 往返；验证设备结果与 rollback 所有权。
3. 对长 prefill 单独验证真正的分块 GDN 和 BF16 数值 profile；算子正确与模型质量过关后，再让 tuning 选择适用 shape/硬件。
4. 核验 projection 合并的量化契约和实际 dispatch，再继续 GEMM tile、prefetch、backend 与 workspace 调优。

按当前证据，技术栈本身不是已证明的性能上限。执行工作量、精度和数据流仍不等价，必须先处理这些差异；尚无证据保证完成其中某项后端到端必然达到 vLLM 的 1.10× 范围。

## 原始证据

45 上调查目录：

`/home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/first-principles/vllm-source-comparison-20261012`

该目录保存 source SHA、官方 tag 的字节核验、完整微基准样本、Torch profiler kernel traces、原始服务报告身份、cold-capture 时间映射，以及最新 main 的 pull 记录和源码 diff。此次只增加最新 main 的独立证据文件，原 `review.json` 和所有性能样本保持不变。

| 清单 | SHA256 |
|---|---|
| `review.json` | `2d85a20031ffdcb964b333f2be87bbb89892e11c84e60eaa8464a9aee58e0e14` |
| `main-source-identities.json` | `1fd4bdc88f9f3625c5f142b480f7f40420e213f43e7b64d4f487af7d43d14971` |
| `main-execution-comparison.json` | `047c8c87f7cc65f2549391146c6471fbc2417b49c63ef3b278c2c17457a2c67d` |
| `vllm-main-pull.json` | `eb83f6a24fd4b3a36eddbe8238f1600f2330da76896b56be265dcd2457ca294c` |

当前微基准没有任务质量验收，服务对照没有达到新基线统计条件，最新 main 也尚未实测。本文的数字用于决定下一次实验的对象，不登记新的性能基线。

## 后续受控证据

上述候选表保留首次调查的身份与数据。当前已合入的 GPU readout、pooled 几何准入修复见对应独立报告。新增 [GDN 数值与 FP8 准备成本](cuda-gdn-numerics-and-fp8-preparation-2026-10-12.md)记录 0f2003f 基点的实验：重复 FP8 准备约 0.28 ms/48 层；修复检查假阳性并保持卷积相同后，Delta 分块仍未过数值门槛。同路径对照为零误差；尚无最新 main 的正式跨引擎实测。

## 再次同步 main

本轮结束前，45 再次 git pull --ff-only upstream main，工作区干净，快进到 3fe917abb13160d60bea5330399f5c09ca282348。相对本报告分析的 f26d9f432ced7b1087d02d29bffed2249df1a18e，5 个文件改动集中在 attention kernel warmup 及其调用/测试，GDN projection、FLA chunk 和推测解码路径未改变。源码参考继续按上述固定提交引用；最新 main 的运行性能对比仍未完成。
