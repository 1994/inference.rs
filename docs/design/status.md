# 实现状态

Linux / NVIDIA CUDA 是首要生产目标：cuTile Rust 投影算子已在 RTX 5090 上验证并有 A/B 性能基线，完整执行器仍待接入；Metal 已有本机 GPU 执行。目标见[技术方案](technical-plan.md)，测量范围见[验证记录](../validation/index.md)。本文随代码更新。

## 已实现

| 模块 | 能力 |
|---|---|
| Foundation / SPI | typed owner/generation、arena/credit、有界事件；请求/模型/设备/执行 IR 与 15 类扩展接口 |
| Package / Compiler | HF config/index、Safetensors payload/binding、分块 WeightLoadPlan、tokenizer/template；decoder/hybrid 编译与 shape/lifetime/scratch/kernel 选择 |
| Backend | CUDA/Metal 分组能力；Metal F32/BF16/F16 权重、F32 计算与状态、分块 prefill、增量 decode、真实 completion 与 profile；CPU 对照仅在显式测试 feature 下可用 |
| State | 逻辑页事务、物理 StateRecipe、KvCacheManager、固定 BlockPool、prefix LRU/refcount/generation、tail COW/pin/rollback、checkpoint |
| Scheduler | tenant/lifecycle 多队列、ReadyDelta/K 窗口、常驻 workspace、slack/WFQ/aging、混合 packing、多维预算、bounded cost probe/EWMA、独立校验与证据 |
| Runtime | scheduler/device owner、共享增量输入、BatchArena/SPSC、两阶段确认、N+1 planning、异步 resource/output、取消/超时/隔离/排空、重算抢占、quiescent checkpoint/replay |
| CPU / Linux | frozen host/token/history budget、采样与确认/回收缓冲复用；动态 affinity mask、cpuset/SMT/PCI NUMA 发现、Bind/Prefer、scope 恢复与启动握手 |
| Workload / Service | Generate/Embed/Rerank/Decision 与 projection head；固定 preparation/delivery pool、原生 token/text HTTP/SSE、OpenAI 格式 models 与非流式 chat/completions、共享请求 ID、背压与断连取消 |
| Observe / Agent | POD/L0、独立 collector/query、共享事件页与 reader credit、Prometheus/W3C/OTLP JSON；typed command、源码关联图、经实际输出验证的隔离配置实验 |
| Quality | 独立微型模型 golden、状态/并发/故障回归、CPU 分配门禁、Metal CLI/Agent/HTTP 与固定到达负载 |

## 剩余缺口

| 范围 | 缺口 |
|---|---|
| NVIDIA / 目标模型 | CUDA driver 与 kernel、pinned transfer、stream/graph、量化，以及完整 Qwen3.8-27B / 5090 的正确性、profiling 与 SLO goodput |
| CPU 性能 | 全线程与 native driver 分配 profile；生产 P99 与吞吐达标；物理 growth snapshot 的全状态刷新成本、非线性成本下的大 batch 尾延迟 |
| Linux 放置 | Linux 原生与 NUMA 硬件运行证据；真实引擎 pool 驻留、迁核与性能影响（Mac 交叉编译仅证明可编译） |
| 状态所有权 | 物理分配/淘汰决策统一到 scheduler ledger；异构状态组、分层 offload、远程 transfer |
| Pipeline / PD | 多 compute flight 的 hazard/fence 验收、多 GPU、远程 prefill/decode 分离与传输协议 |
| 模型与 feature | MoE、多模态/encoder、任务专项 head 与质量、其他精度、speculation、grammar/tool/LoRA |
| 扩展与制品 | 完整 provider 组合、动态 C ABI/WASM loader、Python/AOT 集成、正式签名 bundle |
| 服务集成 | 完整 OpenAI 协议（SSE、stop、tools、批量输入等）/Responses/gRPC、身份认证与生产配额、分布式 routing 与 state transfer adapter |
| 观测与 Agent | CUPTI/counter、网络 exporter、自动根因与失败最小化、代码制品实验、重复测量与置信区间、远程 adapter 与持久实验仓库 |

局部零分配与 tiny Metal 数值证据只适用于[对应测量范围](../validation/cpu.md)。能力达到 execution、correctness、benchmark、profiling、observability 与 Agent 六项验收后，才记为生产 Supported。
