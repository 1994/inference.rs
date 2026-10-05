# 推理引擎技术方案

本文定义目标和模块契约；[实现状态](status.md)记录完成范围，[验证记录](../validation/index.md)记录证据。

## 目标与平台

建设 Rust 原生推理引擎，单节点性能对照 vLLM、SGLang、TensorRT-LLM 和专项 Embedding/Rerank 引擎，同时提供可验证的正确性、请求进展、状态恢复、观测与扩展能力。

Linux 是首要生产平台，NVIDIA CUDA 是首要 backend；优先 NVIDIA 40 系及以上，首要验收组合为 Qwen3.8-27B / RTX 5090。Metal 用于本地 GPU 开发和回归；CPU 执行器只用于显式测试构建。分布式 routing、discovery 和 orchestration 位于引擎之外，通过协议与状态传输接口集成。

## 架构约束

1. Core 提供身份、所有权、生命周期和进展机制；模型、workload、调度策略、设备与协议通过 SPI 扩展。
2. 加载期完成模型解析、能力匹配、kernel 选择和执行图编译。热路径消费静态 `ExecutionProgram` 与数值化 `StepPlan`，不搜索 registry 或解释 JSON。
3. 设备能力按 backend 分组；选择依赖 dtype、layout、memory、transfer、graph 和 kernel 要求，不依赖产品名称分支。
4. Workload 与 modality 正交；模型拆为 backbone、state 和 head，模型名称不进入设备运行时。
5. Scheduler 只规划与提交状态转移，I/O、tokenization、media 准备、GPU 编码、文本交付和查询各有 owner 与预算。
6. KV、recurrent、conv、speculation 与 media 状态统一描述生命周期，但保留各自物理布局和安全条件。
7. 每个等待状态有原因、唤醒条件和超时策略；取消不越过 GPU fence 或 CPU reader 生命周期。
8. 优化必须保留独立 reference；性能结论必须附输入、环境、计时范围、正确性和 SLO。

```text
Model × Workload × Modality × Feature × Precision + DeviceCapabilities
    → 编译与规划 → ExecutionProgram + StateRecipe → StepPlan → Backend
```

目录与依赖方向见[代码布局](../architecture/layout.md)。

## IR 与编译

| IR | 必须表达的内容 |
|---|---|
| Request | typed ID、model、input、workload、QoS/SLO、sampling、输出约束和扩展 |
| Model | encoder/decoder/hybrid/recurrent backbone、attention/linear/SSM mixer、dense/MoE FFN、position、state 与 heads |
| Precision | storage/compute/accumulator dtype、scale 粒度、量化参数和 tensor layout |
| ExecutionProgram | tensor、op、weight/state binding、依赖、lifetime、scratch、kernel 与源码身份 |
| StateRecipe | KV block、私有状态、tokens/page table、host mirror/readback/lease 的 checked sizing 与布局 |
| StepPlan | 阶段、token span/frontier、资源预留、成本、recipe 和决策证据 |

模型包首先导入 HF config、索引、Safetensors 和 tokenizer/processor，再降低为统一 IR。权重指纹覆盖实际 payload；加载失败释放已创建资源。加载暂存、权重驻留、私有状态、共享 KV、scratch 与观测内存分别预算。

产物面向二进制与 kernel bundle 部署；生产运行不依赖 Python、Torch 或现场编译器。分发格式包含 manifest、weights、IR/precision、校验与 kernel hints，签名验证属于正式制品契约。

## Workload 与输入准备

| Workload | 专项规划与输出 |
|---|---|
| Generate | continuous batching、chunked prefill、增量 decode、paged/prefix KV、stream、结构化输出、tool、speculation |
| Embed | 长度分桶、forward、pooling、MRL、normalize、融合 head 与最小物化 |
| Rerank | query/document pair packing、template 复用、长度预算、scoring head 与 top-k |
| Decision | Binary/Categorical/Ordinal/Continuous、选项 scoring、分布/期望值、校准与 abstain/escalate |
| Classify / Reward / LateInteraction | 独立 head、packing、成本、任务指标和 benchmark profile |

Decision 支持 hidden tap、共享上下文下的多问题/选项规划和 affinity；成本包含 context、media、问题数、选项数与 head。质量指标包括 Brier、ECE、NLL、top-k、ordinal MAE 和决策翻转率。

多模态输入经 text/image/video/audio encoders、fusion 后进入 backbone。fetch、decode、preprocess、encode、cache 与 fusion 是独立的有界阶段；scheduler 只消费 ready work。缓存区分原始输入、解码数据、tensor 与 encoder embedding，允许跨请求 encoder batching。

## Backend、GPU 与 kernel

公共 backend 合约覆盖 capability query、compile、state allocation、submit 和 completion；专属能力、driver 与 kernels 放在各 backend group。NVIDIA 优先；Metal 使用独立的 UMA 和 storage 契约；后续 backend 不要求改写 scheduler。

NVIDIA runtime 管理 context、stream、event、module、graph、device/pinned buffers 与 memory pools。加载期预建 recipe、events 和 graph variants；batch 更新持久 metadata，连接 transfer/compute 依赖，选择 graph 或 direct launch。CUDA Graph bucket 可按 batch 1/2/4/8 等扩展，兼容性须包含 shape/layout/state。

Kernel provider 支持原生 Rust GPU 路径、CUDA/PTX/cubin、vendor BLAS/NCCL 与 Triton/CuTe AOT bundle。descriptor 包含 op、architecture、precision、shape、workspace、能力与性能提示；启动期选择/调优并缓存结果。

精度优先 BF16、FP8、NVFP4、MXFP4；每种精度分别验收数值、状态布局、kernel 和性能。legacy GPTQ/AWQ/bitsandbytes/INT8 不属于首要实现范围。

## CPU 流水线与内存

每 GPU shard 采用 SchedulerOwner 和 DeviceSubmitOwner 单写可变状态，外围为有界 preparation、output、delivery 与 collector。网络使用 async；CPU 密集任务使用固定 worker，不能占用网络或调度 owner。

热表、token storage、planning scratch、batch/flight arenas、metadata/output slots 和确认池启动预留。队列传带 owner/generation 的 handle；输入和输出共享只读 payload，decode 提交新增 token 与页表增量。准入限制 slots、tokens、bytes 和 terminal credits，释放遵循 fence/reader。

CPU 可在 GPU N 执行期间准备独立请求 N+1；compute depth 由 backend scratch/output/state hazard 决定。同请求下一 token 依赖前一步结果。Linux 提供 cpuset 内物理核选择与 GPU NUMA 放置，macOS 提供 QoS/affinity hint。

完整线程、队列、zero-copy、fence 与性能预算见 [CPU Runtime 设计](cpu-runtime.md)。

## 调度与进展

调度分为 admission、workload packing 和 device micro-scheduling：

- Admission 校验输入就绪、tenant 配额、最坏 token/state 预算、CPU/device 容量及可选 SLO 风险。
- Packing 按 workload、阶段、program/layout 兼容性组合任务；prefill、decode 和 forward 有独立入口。
- Micro-scheduling 综合 TTFT/TPOT slack、aging、tenant WFQ，以及 batch/token/GPU time/workspace/page/transfer/encoder 预算，优化满足 SLO 的有效吞吐。

ReadyDelta 更新常驻索引；候选窗口、成本探测和维护工作受预算限制。资源阻塞请求进入独立 waiter，由资源 epoch/credit 变化唤醒，不能挡住其他可行请求。成本反馈校准须有冷启动 fallback 和 epoch；自定义非线性模型不假定可加或单调。

Progress epoch 在 token、prefill、media、state transfer 和生命周期推进时增长。runnable 且资源允许但重复空计划应生成 invariant 诊断；等待、超时、抢占与恢复保留结构化原因。可见请求终止不等于已完成设备资源回收。

运行时持续检查三类不变量：可行 runnable 最终被调度；allocated state 按唯一物理资源核对 owned/shared/reserved 账本；submitted step 最终 completed/failed，accepted request 始终具有明确执行、等待或终止状态。违反时产生结构化诊断。

## 状态、缓存与 PD

State SPI 提供 reserve、allocate、reference/share、evict、transfer 和 release。目标是 scheduler 持有统一分配决策 ledger，device 持有执行镜像；批次预留、KV/recurrent/conv 与 prefix 恢复必须原子一致。

Paged KV 支持增量增长、refcount、generation、tail COW、prefix LRU、pin 与延迟回收。容量按独占物理块计数，区分 active、cache-only、pinned 和 reserved；压力先回收 cache-only，再按策略抢占可重算请求。

HBM 是首层；Host DRAM、NVMe、remote 是扩展 storage tier。PD 分离通过 typed state transfer recipe、身份/版本、传输完成确认和接收方预算衔接；IPC/NVLink/NIXL/RDMA 属传输实现，不进入调度策略。

Checkpoint 仅在 quiescent 边界保存请求、调度、状态、资源、配置、seed 和必要历史；restore 校验身份与物理数据。诊断 snapshot 可捕获在途情况，不能直接作为恢复 checkpoint。Replay 记录控制动作、成本输入与策略身份，后续支持 failure minimization。

## Feature 与扩展

Speculation 使用统一 proposal、hidden taps、candidate、verification、acceptance 与 state 契约；MTP/DFlash2 优先，EAGLE 后续。grammar、tool、LoRA、prefix 和自定义计划走 FeatureProvider，必须声明资源与状态影响。

15 类 SPI：ModelProvider、WorkloadProvider、ModalityProvider、FusionProvider、FeatureProvider、PrecisionProvider、SpeculationProvider、SchedulingPolicy、SequenceStateProvider、StateStorageProvider、StateTransferProvider、BackendProvider、KernelProvider、ProtocolAdapter、Observer。

Native Rust 用于热路径；动态扩展使用稳定 C ABI，避免 Rust ABI 假设；WASM 用于策略/控制。Python sidecar 可用于早期扩展和 reference，生产优先 load-to-IR 或 AOT kernels；共享设备数据须使用显式 IPC/lease 协议。

## 服务、观测与 Agent

协议适配器把 Native、OpenAI/Responses、gRPC 和 Agent/OpenEngine 输入转为 CanonicalRequest。Gateway 负责 parsing、tokenization、stream/backpressure、auth、rate limit、cancel 和 health；分布式路由通过外部 RPC/capability/events 对接。

观测分 L0 metrics、L1 timeline、L2 GPU counters、L3 kernel profile。热事件为固定 POD，格式化与导出在 collector；详细 trace 可丢失并计数，完成和资源确认必须可靠。因果关系为 Request → Decision → Step/State → Program/Op → Graph/Kernel → Source，证据包含 Action、Reason、Evidence、Dependency。

NVIDIA profile 对接 CUPTI activity/external correlation、graph metadata、PM/PC sampling。通过 macro/build metadata 把 OpId、KernelId、PolicyId 映射到 crate/module/file/function。任何计时须区分 CPU encoding、GPU command、kernel 与网络等待；histogram 和 sampling 开销有 A/B 验证。

Agent 提供 inspect/query/explain、trace、scheduler/state、snapshot/replay、profile、accuracy 和 benchmark。实验隔离 baseline/candidate，验证实际输出和资源，再依据 SLO goodput/P99 接受或拒绝；配置实验与代码制品实验分开。CLI、JSON-RPC、gRPC/MCP 是外部适配，不进入推理核心。

## 验收契约

一项能力只有同时具备 execution、correctness、benchmark、profiling、observability 和 Agent 可见证据，才记为生产 Supported。

一期要求 Ada/Blackwell 能力路径与 Generate/Decision/Embed/Rerank 原生执行成立，BF16 reference、graph/batch/prefix/decode invariance 通过；高并发 media 准备不阻塞 scheduler；调度/运行时故障能由 snapshot 做确定性控制回放。模型吞吐须在相同条件下对照主流引擎，不能以语言或局部小测替代。

| 正确性层 | 检查 |
|---|---|
| C0 / C1 | tokenizer/template/processor parity；独立 kernel reference |
| C2 / C3 | op/block/layer 首处分歧；logits max error、RMSE、cosine、KL、top-k |
| C4 / C5 | decode 轨迹；batch/graph/prefix/chunk invariance |
| C6 / C7 | 任务评测；Decision 校准与 abstain 行为 |

| 性能层 | 测量 |
|---|---|
| B0 | 同 CUDA 调用的 Rust/C++ submission，对照开销目标 ≤ +2% |
| B1 | kernel 时间、register/shared memory、occupancy、bandwidth、SASS |
| B2 | prefill/decode/embed/decision/rerank model step |
| B3 | 无网络 engine 的 scheduler/state/CPU/GPU pipeline |
| B4 | 相同硬件、模型、精度、输入和 SLO 的 serving 对照 |

负载覆盖 closed、open-loop Poisson、trace replay；报告 offered/accepted/completed/rejected、TTFT/TPOT/ITL/E2E P50/P95/P99、throughput、CPU/GPU utilization、idle gap 与 VRAM。SLO goodput 为满足 P99 条件的有效吞吐；最大容量搜索不能用单档测量替代。

矩阵覆盖长度/并发、长短混合、media、问题/选项、query/document pairs、prefix、speculation acceptance/verification cost。故障场景覆盖 cancel、disconnect、timeout、media failure、state pressure、prefix eviction、invalid schema、speculation failure 和 OOM；要求无请求/状态泄漏、deadlock 或 livelock。

## 设计项索引

| ID | 对应契约 |
|---|---|
| ARCH-001 / CORE-001 / SPI-001 | 分层、核心机制与扩展边界 |
| IR-001 / MODEL-001 / MM-001 | 统一 IR、模型描述、多模态图 |
| EXEC-001 / HW-001 / CUDA-001 / KERNEL-001 | 编译执行、设备能力、NVIDIA runtime、kernel provider |
| STATE-001 / SCHED-001 / COST-001 / PROGRESS-001 | 状态 ledger、调度、成本与进展 |
| GEN-001 / DECISION-001 / EMB-001 / RERANK-001 / MMRT-001 | workload 与输入流水线 |
| PREC-001 / SPEC-001 / EXT-001 / PYEXT-001 | 精度、推测解码、native/AOT/sidecar 扩展 |
| VERIFY-001 / PROF-001 / OBS-001 | 分层正确性、profile 与因果观测 |
| SNAP-001 / AGENT-001 / BENCH-001 | 恢复回放、诊断实验与性能验收 |
| RPC-001 / DYNAMO-001 / PACKAGE-001 / SECURITY-001 | 服务集成、分布式边界、制品与安全 |

依赖顺序：IR/SPI → 模型与执行 → 状态/调度/CPU pipeline → workload/服务/观测 → NVIDIA kernels/精度 → media/speculation/分布式扩展。每阶段按上述验收契约更新[实现状态](status.md)。
