# Runtime 执行与观测

Runtime 协调请求、调度、状态、设备和 workload；通过 SPI 依赖 backend，不导入具体执行器。职责入口见[代码布局](layout.md)。

## 执行流水线

```text
preparation → resource quote/reserve → ready queues → plan/validate
    → batch publish/launch ack → GPU fence → CPU output/逐请求确认
    → completion commit → delivery/consume → release ack
```

SchedulerOwner 单写请求、队列和逻辑状态；DeviceSubmitOwner 单写 device/context、物理 buffers 与 tickets。preparation、output、delivery、collector 和 query 使用有界独立执行单元。Tokio 处理网络等待，长期 owner 使用普通线程；具体协议见[CPU 设计](../design/cpu-runtime.md)。

`Engine::tick_into` 处理完成与 control，再推进资源、规划和提交。GPU 在途期间可构造独立请求的 provisional N+1 plan；提交前检查 ready/resource/cost/config epoch。当前设备 compute depth 为 1，批次内可有多个请求。

资源报价、Reserve/Reset/Prefix/Release 通过 typed command/reply pool 协调；满队列保留意图并重试，确认身份后提交状态。独立 output worker 处理 sampling/projection，每请求确认，慢 reader 不阻塞同批其他完成。

## 故障与排空

请求错误终止对应请求；无法保证 ownership、batch 或 backend 安全的错误使引擎隔离：ready=false、拒绝新请求和停止 dispatch，但继续 poll 实际 flights、处理 readers 和 release。

cancel/deadline/timeout 对客户端可见后仍保留设备与 CPU reader credits。非协作 provider/driver 无法安全强制中止；错误不能伪造资源排空。shutdown 最多等待 5 秒，返回收敛结果；未收敛时保持 owner 与在途资源，不自动恢复 healthy。

请求状态、terminal 消费和 checkpoint 契约见[生命周期](request-lifecycle.md)。

## 观测的数据与线程边界

| 层 | 内容与约束 |
|---|---|
| L0 | event ring 写入前更新固定计数/直方图；backend/kind 等低基数 labels，不使用 request ID |
| Events | 48-byte POD；owner 独立 writer、collector 合并；ring loss 显式计数 |
| History | 有界共享事件页、reader credits 与 publication barrier；查询窗口冻结且报告 gap/dropped |
| Trace metadata | 注册/退休是可靠生命周期，不依赖可丢事件；完成后按保留策略回收 |
| Query/export | 在独立冷 worker 过滤和构建 JSON、Prometheus、Chrome trace、OTLP payload |

快照无空 slot 时保留上一版并报告 age；只读查询不能长时间占用热表。指标 histogram 报告 bucket 上界，详细 timeline/probes 与 GPU profile 有独立预算。

HTTP 接收 W3C version 00 traceparent；非法值忽略。OTLP JSON 表达已保留的 request spans，cursor 按 Engine session 隔离。导出格式不等于已实现网络 exporter/重试。backend timing 区分 CPU encoding、CPU wall 和 GPU command。

接口见[开发指南](../guides/development.md)，源码关联与实验见[Agent](agent.md)，验收和剩余目标见[状态表](../design/status.md)。
