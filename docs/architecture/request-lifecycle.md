# 请求生命周期

Frontdoor 管理请求进入 Engine 之前的连接与准备，Engine 管理 accepted 请求的状态与资源。`infer-scheduler` 提供队列与索引，Runtime 提交生命周期变化，backend 与 worker 确认实际执行和消费。

## 阶段与持有资源

| 阶段 | owner / 资源 | 退出条件 |
|---|---|---|
| Preparing | Frontdoor / IngressHandle、job/byte credits | 准备完成、取消或准备失败 |
| Admission | Runtime / workload plan、quote/reserve ticket | 校验与预留提交；拒绝按阶段补偿 |
| Runnable | Scheduler / tenant/phase 队列、逻辑与私有状态 | 被选中、资源阻塞或终止 |
| ResourceWaiting | Runtime / typed ticket、FIFO waiter | 匹配 ack 或资源 epoch 变化 |
| Running | Device owner / batch、reserved frontier、GPU lease | 匹配 fence，不以取消替代 |
| CpuOutputWaiting | Output worker / 结果 reader、sampling/projection scratch | 逐请求消费确认 |
| CancelPending | Runtime / 仍在途的 flight 与 reader | reader / fence 收敛后释放 |
| Terminal / ReleasePending | Runtime / terminal credit、待释放状态 | delivery 消费或断连，并收到 release ack |
| Consumed | ID 退休账本 | slot 可复用，外部 RequestId 不复用 |

被拒绝的请求不消费 accepted ID；同一 Engine 的 accepted `RequestId` 不复用，使用有界已用 ID 窗口与退休下界。内部 handle 同时校验 owner / index / generation，generation 耗尽即退休，旧 completion 不能命中复用的 slot。

## 执行与状态提交

- prefill 使用 token span / range；decode 使用 `{position, token}`；forward 保留 workload 专属阶段与游标。
- prompt 共享，生成尾部预留，Step 不复制完整上下文。
- batch 发布前校验身份、frontier、重复 state、资源与输出额度；逻辑增页作为批量原子事务。
- `Published` / `LaunchAck` 记录实际提交，DeviceComplete 与 CPU reader 确认之后才提交结果。失败时回滚未执行的预留，已执行部分按真实 fence 收敛。
- 普通 Generate 使用最小 hidden / logits retention；Full readout 保留 projection 所需历史并单独报价。Full prefix 可供 compact readout 使用，compact prefix 不能升级为 Full。

## 取消、背压与资源等待

cancel / shutdown 使用独立 control lane 与持久 mailbox，不会被 bulk 队列占满吞掉。runnable 请求取消后移出索引；已发布的 Reserve 即使调用方不再等待也必须补偿；running 或 CPU reader 阶段的取消保留 credits 到实际结束。

资源 waiter 按有界 FIFO / epoch 唤醒，计算与输出分别确认。慢 Reserve、projection 或 delivery 只阻塞相关请求。超时返回结构化原因；非协作任务挂起不能通过强制复用 buffer 解决。

terminal credit 在准入时保留，完成记录保持到消费或断连。共享结果持有 reader lease，最后一个 reader 归还额度；事件丢失不影响可靠确认。资源无法安全收敛时进入[引擎隔离](runtime.md)，而不是伪报 idle。

## Checkpoint 与 replay

Quiesce 暂停准入与 dispatch，排空 GPU flight、resource command 和 CPU reader，确认逻辑 / 物理 frontier 一致后捕获 checkpoint。线程化入口先请求排空，再由 device owner 捕获 typed state；在途诊断 snapshot 不用于 restore。

Restore 校验当前 schema、精确 weights/program/backend、workload/cost/admission provider 身份、配置、成本反馈、请求游标与物理 payload，再创建新的 owner 身份。旧或不匹配的 schema 明确拒绝，失败时保留原状态。具体版本以 [persistence 代码](../../crates/engine/runtime/src/persistence)为准。

Journal 保留 Submit / Tick / Quiesce / Cancel / Drain / CostFeedback 与调用方单调时钟。回放使用记录中的成本输入；历史截断会报告 dropped。异步 GPU 由哪个 tick 观察到可能变化，因此输出语义与控制记录分别核验。

物理页、prefix 与 restore 的安全条件见 [KV Manager](kv-manager.md)，线程与确认协议见 [CPU 设计](cpu-runtime.md)。
