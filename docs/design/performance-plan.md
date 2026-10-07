# 性能路线：从诊断执行器到对照 vLLM

本文给出达到「相同硬件、模型、精度、输入下比 vLLM 快」的具体路径。目标与契约见[技术方案](technical-plan.md)，已验证范围见[验证记录](../validation/index.md)，当前能力见[实现状态](status.md)。本文随实现更新。

> 2026-10-06 验证补充：[sm_120 NVFP4 MMA 探针](../validation/cuda-sm120-mma.md)已通过；较大 tile 的 release cubin 包含原生 `OMMA.SF`，小 tile 数值通过但未生成 MMA。MLP 子图 graph 的三种存储精度小尺寸正确性检查也已通过，整步 target/MTP 设备图也已通过真实模型短验证，ModelScope pilot 的首轮吞吐为 40.23 tok/s，详见[设备图实测](../validation/cuda-resident.md)；同步 BackendProvider、Engine 与 CLI/HTTP 服务已通过真实模型验收，异步执行和服务内 MTP 仍未接通；32-token BF16 MMA prefill 已完成短验证，尚非完整调优。下文阶段速度和框架胜负比例均为假设，不是实测承诺；早期整模型预检曾与另一模型加载重叠，不能作为干净的性能基线。

> 两处实现边界澄清：整步 CUDA Graph 可以先输出 logits，在图外采样，P3 并非 P1 捕获的必要前提；当前 `Mtp::checkpoint` 克隆的是单层 draft attention 的 KV 状态，不是 target 的 48 层 recurrent 状态，151 MB 不能作为当前每轮 checkpoint 拷贝量。设备状态迁移仍须分别验证 target 与 draft 的恢复语义。

## 本轮实现进度（2026-10-06）

| 阶段 | 已落地 | 未覆盖 |
|---|---|---|
| P0/P1 | 激活复用 arena；所有辅助算子与状态驻留 GPU；target/draft 整步图；跳过 lm_head 的 prefill 图；同步 Engine/CLI/HTTP 接线；有界空闲状态/图复用；异步 metadata、pinned 批量回读统一等待 | scheduler 图资源声明与异步执行；异步 H2D、设备采样 |
| P2 | 分块在线 softmax；可选模型标定 FP8 KV；17 种实际投影 dtype/shape 调优，7 种通过独立复测 | BF16 激活、全局融合、硬件 counter 指导的调优 |
| P4 | 节点交错批量 verify、三候选共享权重 GEMV；设备 Conv/GDN 前缀快照；draft KV 逻辑回退 | tensor-core verify、树形 draft；当前 MTP 仍无净加速 |
| P5 | 三 token GEMV / 32-token BF16 MMA prefill 专用图，跳过中间词表投影和回退快照，尾块屏蔽状态更新 | 辅助算子批量化、大块 GEMM 流水、并行 GDN/prefill attention |
| 对照 | 受内存限额保护的 ModelScope pilot、硬件遥测、真实 vLLM MTP0/2 | 并发 SLO goodput、P99 ITL、完整质量评估、H200 |

实测表和精度限制以[CUDA 报告](../validation/cuda-resident.md)为准。下面保留的是路线分析；不应把阶段预计速度当成当前实现性能。

## 0. 判定口径

「比 vLLM 快」必须落到可比的量，否则无法判断：

| 层 | 指标 | 判断 |
|---|---|---|
| B1 算子 | kernel 时间、带宽利用率、tensor core 利用率 | 只在对应形状与精度上成立 |
| B2 步 | decode step 时间、prefill step 时间 | 与权重量成正比，可与 roofline 比 |
| B4 服务 | 同模型/精度/硬件/输入的 TTFT、TPOT、ITL、SLO goodput 的 P50/P95/P99 | 唯一能声明「更快」的口径 |

decode 是**带宽问题**，prefill 是**算力问题**，吞吐是**批处理问题**。三者需要不同投入，任何一个上的领先都不能外推到另外两个。

## 1. 现状解剖

`crates/backend/cuda/examples/model_smoke` 的诊断路径（`framework: "rust-diagnostic"`）在 Qwen3.8-27B-NVFP4 / RTX 5090 上测得：

| 量 | 实测 | 来源 |
|---|---:|---|
| decode 单步 | 16.79 s / 127 步 = **132 ms** | `artifacts/modelscope-rust-mtp0.json` |
| prefill | 61 token / **7.074 s** = 116 ms/token | 同上 |
| 端到端 | **5.41 tok/s**（820 token / 151.6 s） | 同上 |

它不是引擎执行器，因此这些数字是下界而不是对照结果；但它精确定位了瓶颈。

### 1.1 roofline

`model.safetensors` 实际构成（按 safetensors 头部统计）：

| 类别 | 字节 |
|---|---:|
| 全部权重 | 22.568 GB |
| − `embed_tokens` (248320×5120 BF16) | −2.543 GB（decode 只读 1 行） |
| − vision tower | −0.921 GB（文本路径不执行） |
| **每 decode token 实际读取** | **19.10 GB** |
| MTP 层（BF16，未量化） | 0.849 GB / draft step |

RTX 5090 带宽 1792 GB/s ⇒ **10.66 ms/step ≈ 93.8 tok/s 是 batch=1 的物理上限**。实测 132 ms 是它的 **8.1%**。

### 1.2 差距来源

`dataflow::lower` 为 64 层生成 **1155 个节点**（48 层 linear attention × 17 + 16 层 full attention × 21 + Embedding/Norm/lm_head）。其中：

- **497 个 `Linear`** 走 GPU（168 NVFP4 + 233 FP8 + 96 BF16，与投影清单一致）；
- **658 个非 `Linear` 节点全部落到 CPU**：`forward` 的 `match` 只对 `Linear` 走设备，其余进入 `infer_backend_host::reference_operation`（`crates/backend/cuda/examples/model_smoke/mod.rs:165-196`）。attention、Conv、Delta（gated delta rule）、GatedNorm、Rope、Norm、Silu 都在主机上，KV 也是主机侧 `PagedRows`。

叠加三个逐算子开销：

1. **每个投影 3 次全流同步**：`Projection::apply` 依次调用 `upload`（H2D+sync）、`matvec_tiled`（`api::zeros` 分配 + kernel + sync）、`read`（D2H+sync）——`crates/backend/cuda/examples/model_smoke/weights.rs:87-100`，同步点见 `device.rs:37/74/84`。每步约 1491 次同步。
2. **每个节点输出做全量 `is_finite` 扫描**（`mod.rs:205-210`），按 64 层中间张量累计每步约 9M 个 f32（另加 248320 个 logits）。
3. **每步重建 `BTreeMap<TensorId, Vec<f32>>` 并克隆整个节点表**（`mod.rs:141-142`），约千次堆分配。

算力不是瓶颈：除投影外的所有算子，在约 200 上下文长度下合计约 0.3 GFLOP/token；而投影本身是 54 GFLOP/token，却处在带宽受限状态——132 ms 折算约 0.4 TFLOPS，是 5090 tensor core 峰值的千分之一量级。**132 ms 里几乎没有时间花在数学上。**

### 1.3 已具备但未使用的基础设施

这项工作比看上去便宜，因为多数契约已经存在：

| 已存在 | 位置 | 作用 |
|---|---|---|
| `plan_lifetimes()` / `scratch_elements` | `crates/foundation/ir/src/dataflow.rs:185-235` | 激活 arena 尺寸与复用槽位已算好 |
| `GraphVariant{batch_capacity,max_tokens,workspace_bytes}` | `crates/foundation/ir/src/execution.rs:67-74` | 图变体 IR 已定义 |
| `StateRecipe` + `PageTable`/`DevicePrivate` | `crates/foundation/ir/src/state_recipe/` | 设备页表已可表达，Metal 已物化 |
| `BackendProvider`（ticket/submit/poll） | `crates/foundation/spi/src/backend.rs:11` | CUDA 只需实现 |
| `KvCacheManager`（COW/prefix/refcount） | `crates/engine/state/src/kv.rs` | 后端无关，可直接复用 |
| cuTile `GraphNode` / `CudaGraph` / `tune::Autotuner` | 依赖 `cutile 0.4.0` | 捕获、回放、自动调优 |
| Metal executor 状态机 | `crates/backend/metal/src/executor/` | submit/completion/prefix/checkpoint 的参考实现 |
| `MAX_SUBMISSION_BATCH = 64` | `crates/backend/api/src/lib.rs:6` | 批处理上限已定 |

`crates/backend/cuda/src/mlp/`（未提交）已经按这个模式实现了一个 MLP block：7 个设备常驻 buffer、`CudaGraph::scope` 捕获 6 个 kernel、`graph.update` + `graph.launch` 回放、融合 norm/silu_mul/residual。**下面 P0–P2 就是把 `MlpGraph` 从「一个 MLP」推广到「一整步」。**

## 2. 优化路线

每一阶段独立可验收；预估按 batch=1、MTP 关闭、同一 61-token 输入推算，不是测量值。

### P0 激活常驻设备，消除 CPU 回退

- **做什么**
  1. 按 `plan_lifetimes()` 的槽位与 `scratch_elements` 预分配一块激活 arena，`BTreeMap<TensorId, Vec<f32>>` 换成 `TensorId → device offset`。
  2. 为 12 个非 `Linear` 的 `TensorOp` 写 cuTile kernel：`Norm`、`Rope`、`Split`、`Attention`、`Conv`、`Delta`、`GatedNorm`、`Silu`、`Sigmoid`、`Multiply`、`Add`，以及 `Embedding` 的设备 gather。
  3. 去掉逐算子 `sync_on`：整步只在末尾同步一次。
  4. `is_finite` 扫描移到显式诊断模式（`--validate`），热路径不做。
  5. 去掉每步 `nodes.clone()` 与 arena 重新分配。
- **证据**：`mod.rs:165-196`、`weights.rs:87-100`、`mod.rs:141-142`、`mod.rs:205-210`。
- **预估**：132 ms → 25–35 ms（29–40 tok/s）。剩余成本是 1155 次 kernel 提交，此时 CPU 提交时间超过 GPU 执行时间。
- **验收**：golden 数值 parity 不变；`profile` 显示每步同步次数 ≤ 2，主机↔设备传输次数与批大小同阶。

### P1 整步 CUDA Graph 回放

- **做什么**：把 `MlpGraph` 推广为整模型的 `DecodeGraph`；位置、page table、采样参数进设备常驻 metadata buffer，全部 tensor 指针在捕获后保持不变；按 batch 1/2/4/8 建立变体。捕获一次，之后每步一次 `cuGraphLaunch`。
- **阻塞项**（必须一起改）
  - runtime 从不宣告图：`crates/engine/runtime/src/pipeline/scheduling/resources.rs:29` 为 `graphs: vec![]`；
  - 校验直接拒绝任何带图的步：`crates/engine/scheduler/src/validation.rs:172-174`；
  - `StepPlan.program` 必须等于引擎唯一 program（`validation.rs:171`），per-variant program 无法派发。
- **预估**：→ 13–16 ms（63–77 tok/s）。
- **验收**：graph 与 direct launch 输出逐 token 一致；`device_depth` 与 launch 次数进观测。

### P2 带宽利用率与读取字节

- **做什么**
  1. GEMV 带宽 76% → 90%+：`api::zeros` 输出改为预分配、去掉清零、128-bit 向量化 load、消除 `A/B` 之间多余的中间物化。
  2. **激活从 F32 改 BF16/FP16**：当前 `TensorSpec` 一律 F32（`crates/model/recipes/src/decoder.rs:18`），激活流量与中间 buffer 直接减半。
  3. **KV cache 用 FP8**：模型已随包提供 `k_scale`/`v_scale`，vLLM 也走 FP8 KV；当前 `PagedRows` 是 f32，attention 读带宽是 FP8 的 4 倍。
  4. **融合 lm_head + 采样**：`lm_head` 单项 1.27 GB（占 19.10 GB 的 6.7%），把 GEMM 与 top-k/top-p/min-p/温度/惩罚、argmax 合成一个 kernel，避免把 248320 个 f32 写回 HBM。
- **预估**：→ 11–12 ms（83–91 tok/s），即贴近 batch=1 上限。
- **验收**：单算子带宽利用率与 roofline 百分比同时进基线文件。

### P3 采样与 logits 留在设备

- **现状**：Metal 每次完成都把 logits 与 hidden 读回主机（`completion.rs:63,73`），采样在 CPU（`crates/engine/workloads/src/sampling.rs:31`，由 `stages/output.rs:37-51` 调用）。CUDA 诊断路径同样把 248320 个 logits 变成 `Vec<f32>`。
- **做什么**：设备侧采样 kernel，只回传 token id（+ 可选 top-k）。这同时也是 P1 的前提——图内不能有主机分支。
- **收益**：每步省去 1 MB D2H 与全量 CPU 归约；对两个后端都成立。

### P4 MTP 高效化

- **现状**（`crates/backend/cuda/examples/model_smoke/decode.rs`）：每个被接受的 token 单独跑一次 `model.step`，逐 token 串行验证；`checkpoint()`/`restore()` 深拷贝整个状态映射（`crates/backend/cuda/examples/model_smoke/mtp.rs:116-122`），其中仅 recurrent 状态就有 48 层 × 48×128×128 f32 ≈ 151 MB。
- **做什么**
  1. **批量 verify**：对已接受 span 一次前向，而不是每 token 一次。
  2. checkpoint/restore 改增量或 COW，去掉整状态深拷贝。
  3. MTP 层权重目前是 BF16 849 MB 且未量化，考虑量化。
  4. 树形/多候选 draft，配合 `StepPlan` 已表达的 token frontier。
- **预估**：在 P2 基础上有效吞吐 ×1.7–2.0 → 140–180 tok/s。**这是单流能真正超过 vLLM 的主要来源。**

### P5 Prefill 的批量 GEMM 与 Tensor Core

- **现状**：prefill 是 token-by-token 循环（`suite.rs:65-72`）；CUDA 侧只有 GEMV（`kernels.rs` 的 `dense`/`fp8`/`nvfp4` 输入都是 1-D `Tensor<f32,{[K]}>`）。**没有 M>1 的 GEMM**，NVFP4 还是解包成 F32 做 FMA（`quantized.rs` 文档明说未使用 block-scaled Tensor Core）。
- **做什么**：tiled GEMM（M=chunk，tensor core MMA、Blackwell block-scaled FP4/FP8）+ chunked prefill 与 decode 交错，保护 TPOT。
- **预估**：7.074 s → 20–40 ms（两个数量级）。
- **风险**：这是最容易输给 vLLM 的一项（FlashAttention + CUTLASS）。**应当购买而不是自研**：`KernelProvider` 已允许 vendor 路径（见[技术方案](technical-plan.md)的 Kernel provider 段），优先 cuBLASLt / CUTLASS / FlashInfer 的 AOT cubin。

### P6 连续批处理与设备页表

- **已具备**：`StepPlan.work: Vec<PlannedWork>` 可表达多请求；`ExecutionRole::Mixed` 已实现并有测试（`crates/engine/scheduler/src/policy/packing.rs:549`）；`MAX_SUBMISSION_BATCH = 64`。
- **缺什么**：CUDA 无 `BackendProvider`（`crates/service/cli/src/backend/cuda.rs:5-8` 直接返回 unsupported）；设备侧分页 KV 与 u32 页表（Metal 用 `uint row=page_table[...]`，CUDA 侧没有）。
- **收益**：权重读取被 batch 摊薄——batch 32 时 10.66 ms 读一次权重换 32 个 token，理论 ~3000 tok/s。这是吞吐数字的来源，也是与 vLLM 正面竞争的地方。

### P7 架构性限制

- **单 flight**：Metal 与线程化后端都只允许一个在途批次（`crates/engine/runtime/src/runner/worker.rs:27,310`，`control.rs:199-201` 拒绝第二次提交），`BatchArena` 有 2 个 slot 却用不上。需要双缓冲 scratch 才能做到技术方案里的 N+1 规划。
- **无编译期融合**：`compile` 强制一 op 一节点（`crates/model/compiler/src/compilation.rs:63-82`），`lower()` 与 `compile()` 之间没有任何 pass。没有 torch.compile，norm+quant、rope+KV append、SwiGLU、residual+norm 这几组必须手写融合——数量有限（约 8 组），但需要先加 pass 基础设施。
- **`KernelRegistration` 只有元数据**（`crates/foundation/spi/src/kernel.rs:9-26`），没有 launcher 链接信息，AOT vendor kernel 无法真正接入。
- **张量并行/多 GPU**：当前完全没有；vLLM 在 27B 单卡上不需要，但在更大模型上是硬门槛。

## 3. 与 vLLM 的胜负面

| 场景 | 判断 | 依据 |
|---|---|---|
| batch=1 decode | **最多赢 5–15%** | 双方都带宽受限，差距只来自提交开销与 MTP |
| MTP 推测解码 | **可赢** | vLLM 的 MTP 走通用 speculative 路径；批量 verify + 树形 draft 有余量 |
| 48/64 层 linear attention | **有窗口** | GDN 状态只有 3 MB/层且不随上下文增长；vLLM 的 `linear_attn`/`gdn_attn` backend 同样是新的 |
| Prefill / TTFT | **易输** | FlashAttention + CUTLASS 质量差距；必须借力 |
| 高并发吞吐 | **靠调度赢 10–30%** | 权重同样被摊薄；差异来自 prefix cache、chunked prefill、SLO goodput，而现有调度器（slack/WFQ/aging/成本探测）比 vLLM 的 FCFS+priority 更细 |
| Embed / Rerank / Decision | **可赢** | vLLM 不做这些 workload，同引擎原生支持是纯增量 |
| 观测/回放/确定性 | **可赢** | typed 证据、checkpoint/replay 是 vLLM 没有的能力 |

**最重要的战略判断：「不用 Python 运行时」不等于「所有 kernel 自己写」。** 若把 no-torch 执行成自研全部算子，prefill 与 attention 必然落后一代；正确形态是把 cuBLASLt/CUTLASS/FlashInfer 的 cubin AOT 打进二进制，只自研有差异化的部分（linear attention、融合算子、调度）。

## 4. 测量协议与当前缺口

- 对照必须走 B4：相同硬件/模型/精度/输入，closed + open-loop Poisson，报告 offered/accepted/completed/rejected、TTFT/TPOT/ITL/E2E 的 P50/P95/P99、吞吐、CPU/GPU 利用率、idle gap、VRAM。
- 现有 `tools/bench/vllm-compare.py` 已经固定了 `max_num_seqs=1`、`enable_prefix_caching=False` 的对照配置，但**vLLM 侧尚无结果**：`artifacts/modelscope-vllm-mtp0.json` 不存在，只有 `modelscope-vllm-mtp0-hardware.jsonl`。在补齐之前没有任何真实的 head-to-head 数字。
- 算子基线（`benchmarks/baselines/`）不能替代模型吞吐；debug/release、冷/热 L2 不能混合比较（[CUDA 性能基线与调优](../guides/cuda-performance.md)）。
- 调优结果必须另做独立 A/B 复测后才能设为默认，并记录 GPU/驱动/Toolkit/编译模式/模型指纹。

## 5. 优先级建议

按「收益 ÷ 确定性 ÷ 工作量」排序：

1. **P0**（激活常驻 + 12 个辅助算子 + 单次同步）——工作量最大但收益最确定，10× 量级。
2. **P3**（设备侧采样）——P1 的前置条件，且本身就有收益。
3. **P1**（整步 graph replay）——`mlp` 模块已证明可行，需要先解除 runtime/validation 的图阻塞。
4. **P2**（带宽）——在 P1 之后才看得到真实效果。
5. **P4**（MTP）——单流超越 vLLM 的主要手段。
6. **P6**（连续批处理 + 设备页表）——吞吐数字的来源。
7. **P5**（prefill tensor core）——优先买 vendor kernel，只在被挡住时才自研。
8. **P7**——架构性欠账，按需偿还。

## 6. 风险

- 自研 prefill GEMM 与 attention 会持续落后于 CUTLASS/FlashAttention 的迭代速度。
- `unsafe_code = "deny"`（workspace lints）意味着任何低层 CUDA 都必须经过 `cuda-core`/`cutile` 的安全封装或受审计的 `#[expect(unsafe_code)]`；若 cuTile 缺少某个能力（如 tcgen05 特定指令、vendor library 绑定），会直接变成路线阻塞。
- graph 回放要求指针稳定，与「状态可变长、页表动态增长」天然冲突，需要在设计上把变长状态全部放进设备常驻 metadata，而不是靠 kernel 参数传递。
- 在补齐 vLLM 对照数据之前，任何「更快」的声明都不成立。
