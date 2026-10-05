# CPU Runtime 设计

本文规定 GPU 引擎的 CPU 控制、准备、提交和交付契约。实现范围见[状态表](status.md)，分配和延迟证据见[CPU 验证](../validation/cpu.md)。

## Owner 与线程预算

```mermaid
flowchart LR
    NET[Frontdoor / Tokio] --> PREP[有界 preparation pool]
    PREP --> SCH[SchedulerOwner]
    NET -. cancel / shutdown .-> SCH
    SCH -->|BatchHandle| DEV[DeviceSubmitOwner]
    DEV --> GPU[Backend / GPU]
    GPU --> DEV
    DEV -->|fenced output| CPU[有界 output pool]
    DEV -->|launch / resource ack| SCH
    CPU -->|逐请求确认| SCH
    SCH --> OUT[Delivery bridge]
    OUT --> NET
    SCH --> OBS[Collector / exporter]
    DEV --> OBS
```

| 角色 | 独占内容与工作 |
|---|---|
| Frontdoor | 连接、外部 ID、IngressSlot、协议与网络 await |
| Preparation | worker-local tokenizer/template/scratch、冷 workload 规划 |
| SchedulerOwner | request/tenant/ready 表、host token、逻辑状态、flight、调度与完成提交 |
| DeviceSubmitOwner | context/queue、buffers、metadata、ticket/fence、GPU 编码与完成检测 |
| Output pool | sampling、projection 与共享结果 reader |
| Delivery | token 游标、detokenization、SSE/JSON、断连与消费确认 |
| Collector | 有界事件/trace、只读 snapshot、诊断格式化与导出 |

每 GPU shard 一个 scheduler 和 device owner；固定 worker 在启动创建。CPU 重任务不得进入 Tokio 网络任务或 scheduler。准备、输出、交付、观测与库内部线程池统一核算可用核预算，避免嵌套并行。多 GPU 按 device 分片，跨 shard 状态转移先排空在途引用。

目标物理 lease 决策 ledger 归 scheduler，设备只维护执行镜像；当前逻辑与物理 owner 的实际边界见[KV Manager](../architecture/kv-manager.md)。

## 队列与背压

多生产者入口使用有界 MPSC，确实只有单生产者/消费者的交接使用 SPSC。batch payload 保存在 arena，ring 只传 owner/generation handle；不得把 cloned producer 当作 SPSC。

| 边界 | 预算与满队列动作 |
|---|---|
| 网络 → preparation | pending bytes、jobs、queue age；限时等待或容量拒绝 |
| prepared → scheduler | staging/token credits；未接入对象留在受限 worker，不继续取新任务 |
| scheduler → device | batch/metadata/output/ack slots；停止发布，继续完成和 control |
| device → scheduler/output | 发布前预留可靠 ack/completion 与 reader credits |
| scheduler → delivery | token/output bytes 与 terminal credit；仅阻塞或取消对应请求 |
| owners → collector | 有界 POD events；允许丢 trace 并计数，不丢完成和资源确认 |

准入先取得准备额度，接入前取得 request、token/state 与 terminal credits；失败按阶段归还。bulk 与 cancel/shutdown 分 lane，取消使用带 generation 的持久 mailbox，shutdown 使用 sticky 状态。generation 比较与写入须原子，防止 slot 重用造成 ABA。

每轮 completion/control、prepared ingress、maintenance 和 planning 均有有限份额；不能 drain-until-empty。按条数、持有字节和最老等待时间限制队列。terminal 额度在 accepted 时保留，慢客户端不能无限保留输出或阻塞整个 shard。

## 增量调度

常驻 `RequestHotSoA`、tenant/lifecycle 队列、ready/deadline/aging 索引和 `PlanningScratch`。冷 prompt、JSON、traceparent 和错误文本与热字段分离；tenant/program 字符串在冷路径转换为整数身份。

ReadyDelta 只更新变化项；按兼容域、tenant WFQ、urgent/aging 和阶段选择候选。候选窗口 K、batch B、成本探测与维护预算约束单次工作。窗口外与资源不足分别解释，轮转和 aging 防止反复只访问相同 K 项。

批次限制 tokens、GPU time、workspace、logical/device pages、private state、transfer 与 encoder。资源阻塞进入 waiter，由 epoch/credit 变化唤醒。单调/可加成本声明才允许摘要或二分；其他 provider 使用有界完整预测。校准在 owner 边界提交，不在一次 decision 内改变排序键。

工作量取决于变动 H、候选 K、batch B 和增页 ΔP；索引更新约 `O(H log R)`，packing 可包含 `K × B` 检查。完整诊断/快照仍可能遍历全部状态，不能据候选窗口宣称整个引擎 O(1)。

决策记录 ready/resource/cost epoch、窗口、选中与检查过的候选、阻塞原因；冷查询在有界基线与 deltas 上展开，缺少历史报告 gap。热验证检查身份、frontier、重复 state、容量与转移；全表 lease 平衡、heap 与冷热一致性由独立 verifier 检查。

## 内存与 zero-copy

启动冻结容量，checked sizing 后预留并触碰 request/batch/flight arenas、token storage、索引、页/lease 表、scratch、metadata/output 和确认池。更改 pool、compute depth 或 placement 必须排空重建；动态策略变更带 config epoch。

```text
host_bytes = request/tenant/index/token storage + preparation/staging
           + batch/flight/metadata/output + delivery retention
           + observation/snapshots + queue/ack storage
```

managed budget 包含固定存储与请求峰值，history 独立限额；它不是 RSS 或 driver allocator 上限。默认 mimalloc；热路径计数同时检查 alloc/realloc/dealloc。共享权重/program，避免逐 token Arc/Mutex 和冷对象最后一次 drop 进入 owner。

prompt 与结果共享只读 payload，生成 token 追加到预留存储。prefill 提交 token range，decode 只提交新 token/frontier/page delta，CPU 拷贝量不随累计上下文增长。rerank 当前连续 query/document 拼接在 preparation 中完成并计预算，不能计作零拷贝。

普通 Generate 只保留所需 hidden/logits；Full readout、probe、profile 和 prefix snapshot 显式报价。CPU sampling 使用常驻 scratch 与有界输出池。SoA、cache-line 分离、批量处理和向量化依据实际 profile；手写 SIMD 必须检测 ISA，并保持 tie/NaN/采样语义。

## 发布、完成与回收

`Published → LaunchAck → DeviceComplete → CpuConsumed → Committed` 分别记录传输、实际设备执行、CPU reader 与状态提交。发布时只形成 provisional/reserved frontier，失败原子回滚；只有匹配身份和 fence 才推进 committed frontier。

| 资源 | 复用条件 |
|---|---|
| ring cell | consumer 已取 handle，仅释放 ring 空间 |
| host input/metadata | DMA 完成；共享/映射读取则等最后 GPU reader |
| device metadata/input/output | 最后 device reader/writer 与 D2H 完成 |
| KV/recurrent/workspace | device fence、pin 与 active/cache refcount 均满足 |
| host output | device 写入完成且所有 CPU reader 已消费 |
| flight/request slot | completion commit、reader/回调退出、credits 和资源确认收敛 |

CPU Release/Acquire 不替代 DMA 或设备同步。cancel、timeout、ticket drop 和客户端终止均不能提前释放。generation 耗尽退休 slot；旧/外池 handle 和重复确认必须拒绝。

GPU N 执行时可准备独立请求 N+1；provisional plan 在 ready/resource/cost/config epoch 变化后重新校验。同请求下一 decode token 等上一步结果。metadata 数、published depth 与 compute depth 分别配置，提升并行度必须证明 scratch/output/state hazard 隔离。

Quiesce 停止准入和 dispatch，排空 flight、resource command 与 CPU reader 后捕获 checkpoint；不保存 native ticket。恢复校验 schema、program/weights/provider 身份、游标、成本状态与物理数据。

## 设备传输与 CPU 放置

NVIDIA 使用有总字节限制的持久 pinned host/device pools，按 batch 合并 span 与页表 delta，以 event 连接 H2D、compute、D2H。graph/direct recipe 共享生命周期；离散 GPU 不默认使用 mapped host KV。[CUDA 传输契约](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-and-overlapping-transfers-with-computation)

Metal shared/private storage 遵守 UMA 读写与完成契约；共享地址仍需 fence。[Apple storage modes](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus)

owner 有 progress 时处理有限工作，无工作时根据通知/deadline park。等待前 arm、重读 wake epoch 与 lanes，避免 lost wake。callback 只更新预留的原子状态并唤醒；adaptive poll/短 spin 有预算，其延迟计入 completion detection。

Linux 在继承 cpuset 内选择不同物理核，先设置 NUMA 策略再初始化/first touch，报告实际放置。macOS 只承诺 QoS/affinity hint。部署规则见[Linux 指南](../guides/linux.md)。

## 性能验收目标

条件：R=1024、32 tenants、B=64、K≤256、增量 decode、预热、无 probe/详细 trace；大词表 sampling 单列。

| 阶段 | 初始 P99 CPU service 预算 |
|---|---:|
| control/index/planning/lease | 40µs |
| metadata/token/page delta | 15µs |
| completion/output descriptor | 15µs |
| device apply/driver launch | 15µs |
| host notification/handoff | 15µs |

直接测量 host critical path P99≤100µs 与纯 CPU 完整 cycle≥10k/s；不能相加阶段分位数推算整体 P99。受控 GPU batch 2ms 时 CPU 引起 idle≤5%，另 sweep 0.25/0.5/1/2/5ms。

```text
Tperiod ≥ max(Cowner, Csubmit, transfer/device critical path)
decode_tokens/s ≤ Bdecode / Tperiod
```

该下界只适用于独立工作流水线；同请求 completion→sampling→commit→launch 依赖直接测 TPOT/idle gap。容量规划初始利用率≤70%，以开放到达实验校准；平均 Little 定律不能推导 P99。

验收覆盖 R=1/64/1024/8192、B=1/16/64、context=1K/16K/上限、prefix/COW/页耗尽、集中 deadline、取消风暴和慢 output。并发测试覆盖 lost wake、满队列、ABA、乱序/重复 completion 与安全排空；Metal 验证数值和真实 fence，5090 验证 pinned/graph/overlap。指定硬件运行性能门槛，普通 CI 保持确定性的分配、协议和正确性门槛。

开放到达报告从计划时刻计算延迟，列出拒绝、queue age、资源排空、CPU/RSS、cpuset、线程数、版本与 seed。观测/Agent 查询在冷 snapshot 上执行；报告 trace gap、snapshot age 与观测 A/B 开销。

流水线参考固定版本的 [vLLM EngineCore](https://github.com/vllm-project/vllm/blob/d61081dc3d3f1740a5d8bf82608b62974393c2de/vllm/v1/engine/core.py) 与 [SGLang scheduler](https://github.com/sgl-project/sglang/blob/35f3c96ff4794a4de15daf12caad371084a037ee/python/sglang/srt/managers/scheduler.py)。线程拆分、SPSC 和 Rust 本身不构成性能达标证据。
