# 请求生命周期

Frontdoor 管理进入 Engine 前的连接与准备；Engine 管理 accepted 请求的状态与资源。`infer-scheduler` 提供多队列/索引，Runtime 提交生命周期变化，backend 和 workers 确认实际执行与消费。

## 阶段与持有资源

| 阶段 | owner / 资源 | 退出条件 |
|---|---|---|
| Preparing | Frontdoor / IngressHandle、job/byte credits | 准备完成、取消或准备失败 |
| Admission | Runtime / workload plan、quote/reserve ticket | 校验与预留提交；拒绝按阶段补偿 |
| Runnable | Scheduler / tenant/phase 队列、逻辑与私有状态 | 被选中、资源阻塞或终止 |
| ResourceWaiting | Runtime / typed ticket、FIFO waiter | 匹配 ack 或资源 epoch 变化 |
| Running | Device owner / batch、reserved frontier、GPU leases | matching fence，不以取消替代 |
| CpuOutputWaiting | Output workers / 结果 reader、sampling/projection scratch | 逐请求消费确认 |
| CancelPending | Runtime / 仍在途的 flight/readers | reader/fence 收敛，再释放 |
| Terminal / ReleasePending | Runtime / terminal credit、待释放状态 | delivery 消费/断连与 release ack |
| Consumed | ID 退休账本 | slot 可复用，外部 RequestId 不复用 |

拒绝请求不消费 accepted ID；同一 Engine 的 accepted RequestId 不复用，使用有界已用 ID 窗口和退休下界。内部 handle 同时检查 owner/index/generation，generation 耗尽退休，旧 completion 不得命中新 slot。

## 执行与状态提交

prefill 使用 token span/range；decode 使用 `{position, token}`；forward 保留 workload 专属阶段和游标。prompt 共享、生成尾部预留，Step 不复制完整上下文。

batch 在发布前完成身份、frontier、重复 state、资源和输出额度校验；逻辑增页为批量原子事务。Published/LaunchAck 记录实际提交，DeviceComplete 与 CPU reader 确认之后才提交结果。失败回滚未执行预留，已执行部分按真实 fence 收敛。

普通 Generate 使用最小 hidden/logits retention；Full readout 保留 projection 所需历史并报价。Full prefix 可供 compact readout 使用，compact prefix 不能直接升级为 Full。

## 取消、背压与资源等待

cancel/shutdown 使用独立 control 与持久 mailbox，不被 bulk 队列占满吞掉。runnable 取消移出索引；已发布 Reserve 即使调用者不再等待也必须补偿；running 或 CPU reader 取消保留 credits 到实际结束。

资源 waiter 按有界 FIFO/epoch 唤醒，计算与输出分别确认。慢 Reserve、projection 或 delivery 仅阻塞相关请求。超时返回结构化原因；不能通过强制重用 buffer 解决非协作任务挂起。

terminal credit 在准入时保留，完成记录保持到消费/断连。共享结果持有 reader lease，最后 reader 归还额度；事件丢失不影响可靠确认。资源无法安全收敛时进入[引擎隔离](runtime.md)，而非伪报 idle。

## Checkpoint 与 replay

Quiesce 暂停准入与 dispatch，排空 GPU flight、resource command 和 CPU readers，确认逻辑/物理 frontier 一致后捕获 checkpoint。线程化入口先请求排空，再由 device owner 捕获 typed state；在途诊断 snapshot 不用于 restore。

Restore 校验当前 schema、精确 weights/program/backend、workload/cost/admission provider 身份、配置、成本反馈、请求游标和物理 payload，再创建新 owner 身份。旧或不匹配 schema 明确拒绝，失败保留原状态。具体版本以[持久化代码](../../crates/engine/runtime/src/persistence)为准。

Journal 保留 Submit/Tick/Quiesce/Cancel/Drain/CostFeedback 与 caller 单调时钟。回放使用记录的成本输入；历史截断报告 dropped。异步 GPU 被哪个 tick 观察到可能变化，输出语义与控制记录分别核验。

物理页、prefix 与 restore 的安全条件见[KV Manager](kv-manager.md)；线程和确认协议见[CPU 设计](../design/cpu-runtime.md)。
