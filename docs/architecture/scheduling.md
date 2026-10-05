# 调度、队列与资源预算

`infer-scheduler` 拥有队列、索引、策略、成本和独立决策校验；Runtime 负责把准入、完成、取消和资源确认转成队列变化，并提交设备执行。策略不做 I/O、tokenization 或等待。

## 多队列与唤醒

| 队列/索引 | 管理内容与唤醒 |
|---|---|
| Tenant ready | 按 tenant 与 prefill/decode/forward 分组，ReadyDelta 增量更新 |
| Fair / urgent / aging | virtual finish、slack、最长等待；状态变化与到期维护排序 |
| Lifecycle | runnable、waiting、flight、cancel、terminal 的成员身份 |
| Resource waiters | 有界 FIFO、ticket/epoch；匹配 ack 或 credit 变化重试 |
| Pressure focus | 内存抢占后的优先完成请求，完成后恢复常规公平选择 |

索引容量在启动冻结；取消/转移更新成员，不积累无界 tombstone。每轮 drain、候选窗口和成本探测有上限，资源重候选跳过后继续选择其他请求。窗口外、CPU 探测预算与 GPU/KV 不足有不同延期原因。

## Admission

WorkloadProvider 先给出执行单元和完整状态预留量；ResourceAdmission 校验 tenant 活动数、最坏 tokens/pages、私有字节、资源就绪和最小执行成本，再提交身份与状态。拒绝不消费 accepted ID，失败预留补偿。

deadline 预测完整剩余执行，TTFT 预测首个 prefill，TPOT 预测最大上下文 decode；不可抢占 flight 的剩余时间计入首次等待。默认报告 `slo_at_risk`，`admission.reject_infeasible_slo=true` 才按预测拒绝。SLO 预测不等于延迟保证。

## 批次选择

1. 等待达到 `max_wait_us` 的请求获得 aging 服务机会。
2. 在 `urgent_budget_percent` 的时间份额内按最小 slack 选择紧迫请求。
3. 其余按 tenant weight/virtual finish 选择，每加入一个公平量子更新批内服务量；平局优先 decode，新 tenant 从活动服务下界开始。
4. prefill/forward 按 token quantum 打包，decode 每请求每批最多一个 token。兼容 program/backend 的阶段可合并为 Mixed，完成按各自 frontier 提交。

常驻 Workspace、DecisionStorage 与 K 窗口复用规划存储。`max_planning_probes` 默认 4096，范围 1..=65536；耗尽保留已有批次并给出 PlanningBudget，空批次使用已校验 singleton forecast 保留进展。策略身份为 `slack-wfq-mixed-v3`。

## 资源与进展边界

批次同时约束 batch、tokens、预计 GPU time、workspace、逻辑页、物理块/字节、transfer 与 encoder。workspace 取最大值，其余新增资源按批次计算；已存在块不重复计费，跨页和 tail COW 必须收费。

无普通可行批次时，允许带 `quantum_overrun` 的 singleton，但仍受全部资源与 `max_singleton_gpu_us` 限制；最小工作也超过绝对上限的请求失败，不能永久延期。独立 validator 用完整 queries 核对策略输出，不相信摘要本身。

内存压力先回收 cache-only 块；资源阻塞请求可触发重算抢占，紧迫/aging 请求也可在其他请求仍可运行时获得 focus。只 reset 安全的其他 sequence，保留生成 token/采样位置；focus 持续到目标完成。GPU command 不被抢占，页生命周期见[KV Manager](kv-manager.md)。

graph variant 仅用于符合 program/batch/token/layout/workspace 要求的同构 decode；描述匹配不代表 backend 已实现 graph launch。

## 成本与诊断

CostModelProvider 预测完整候选批次。原生 fallback/EWMA 可声明精确 BatchCostSummary；扩展 provider 默认走完整预测，不假定非线性成本可加或单调。

CalibratedCosts 按阶段、batch、tokens 和 context bucket 保存有界 EWMA：默认 512 profiles、更新权重 25%、预测余量 20%，先精确桶再近邻/同阶段 fallback，冷启动用 `cost_per_token_us`。Metal 输入已完成 GPU command 时间，CPU 对照输入 wall time，CPU encoding 单列。

反馈在 owner 边界提交并写 CostFeedback journal；checkpoint 绑定 provider identity 和校准状态。选择/延期证据包含阶段、slack、等待、公平标签、资源 required/available、成本和依赖。

[配置示例](../../examples/scheduling.json)提供预算与 tenant 配额。请求 QoS 的 tenant/weight 必须匹配配置，TTFT/TPOT 为正且只用于 Generate；deadline 为 caller 单调时钟绝对值。Agent `scheduler.inspect`、`request.trace`、`runtime.explain` 提供证据。
