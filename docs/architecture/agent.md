# Agent 控制面

`infer-agent` 为 Coding Agent 提供引擎诊断、控制与实验；它不调用外部语言模型。依赖方向为 CLI → Agent → Runtime/SPI，核心不反向依赖 Agent。

## 组织与扩展

| 层 | 职责 |
|---|---|
| protocol / service | JSON-RPC envelope、通知、顺序 batch、错误与审计 |
| transport | 有界 NDJSON，独立于业务分发 |
| registry | typed 参数、规范名称/别名、效果与事务注册 |
| context | 控制会话、Engine、审计与实验记录 |
| commands | runtime、diagnostics、state、quality、observability 分域处理器 |
| AgentBackend | backend/kernel catalog、trace/probe 与 timing adapter |
| graph / experiment | 实际源码关联、隔离 baseline/candidate 与验收 |

新增工具调用 `AgentService::register`，新增设备实现 `AgentBackend`。Command/Registry 管理领域行为，Adapter 隔离传输与后端；snapshot 复用 Runtime checkpoint。

## 协议与限额

JSON-RPC 2.0：通知不回复，显式 `id:null` 回复；batch 按输入顺序执行，不提供跨命令事务。解析/请求/方法/参数错误为 -32700/-32600/-32601/-32602，领域错误保留 `error.data.code`。[协议规范](https://www.jsonrpc.org/specification)

单行≤1 MiB、batch≤64、审计≤256 条、实验历史≤16。超大行完整丢弃后继续读取，不执行截断命令；历史公开 dropped。每进程一个本地 stdin 会话和 Engine。

`agent.discover` 返回实际方法、别名、参数与 read_only/control/isolated_execution 效果；效果不是权限机制。typed 参数拒绝未知字段，复合 IR 由自身 validator 检查，当前不自动生成完整 JSON Schema。

## 命令与证据

| 入口 | 内容 |
|---|---|
| agent.discover / agent.inspect | 注册契约与审计 |
| runtime / request / scheduler | inspect/query/explain、submit/tick/cancel、request.trace、scheduler.inspect |
| state / kv / cache / snapshot | 状态统计、ownership、checkpoint、restore/replay |
| runtime.graph / runtime.diagnose | Request→Decision/Step/State→Program/Op→Kernel/Source；生命周期与所有权不变量 |
| gpu.profile / kernel.profile / executor.probes | 实际后端时间、注册来源、执行 op 与有界层探针 |
| accuracy.verify / benchmark.run / benchmark.compare | 数值验证、closed 测量和候选验收 |
| experiment.run / inspect / list | 隔离配置实验及有界结果 |

精确参数和兼容别名以 discovery 为准。关联图来自真实保留记录，公开 trace/journal/event gaps；Source 来自 kernel registration。Metal op elapsed 为 CPU encoding，GPU command 时间单列。

最小发现请求：

```json
{"jsonrpc":"2.0","id":1,"method":"agent.discover","params":{}}
```

## 配置实验

`experiment.run` 接收 baseline/candidate RuntimeConfig、同一 CanonicalRequest 集和约束。两侧共享模型身份与 fork 后的 workload providers，创建独立 Engine/state/cost/request 空间，顺序运行并释放；不改写当前会话。

实际 token、排名和结构严格比较，浮点按 atol/rtol。可提供覆盖全部请求的 reference_outputs；未提供时报告 baseline/candidate parity。该入口不接受调用者声称 `correctness_passed`，而 `benchmark.compare` 的同名字段仍是外部证据。

约束包括成功率、SLO goodput、P99 E2E、最低 goodput 比例与可选 state bytes 上限；要求字节而无测量证据时拒绝。每侧≤30 秒/一百万 ticks、≤256 请求。结果保存 ExperimentId、workload SHA256、配置、测量、正确性与接受/拒绝原因。

代码制品实验、自动性能根因、远程 adapter、持久化与 counters 的剩余范围见[状态表](../design/status.md)。
