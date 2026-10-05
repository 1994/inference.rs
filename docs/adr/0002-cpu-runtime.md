# ADR-0002：CPU 与 GPU 提交流水线

日期：2026-10-04。状态：采用。设计项：ARCH-001 / SCHED-001 / COST-001 / CUDA-001 / OBS-001 / BENCH-001。

## 决策

1. 每 GPU 一个 SchedulerOwner 与 DeviceSubmitOwner，准备、输出、交付和观测有独立有界 worker；网络使用 Tokio。
2. Scheduler 单写请求、队列、逻辑状态；目标物理决策 ledger 同属 scheduler，设备维护执行镜像。
3. 热表、token/metadata/output、arenas、planning scratch 与确认池常驻；队列传 generational handles，MPSC/SPSC 按实际 producer 数选择。
4. ReadyDelta 和有界窗口驱动规划；decode 提交增量 token/page，普通生成采用最小 readout；CPU sampling 移入 output pool。
5. 发布、launch ack、GPU completion、CPU consume 分别确认；取消/shutdown 独立于 bulk，满队列不吞控制意图。
6. 用可靠通知、有界 drain/adaptive poll 唤醒 owner；Linux 支持 affinity/NUMA，macOS 只提供系统提示。
7. CUDA 使用持久 pinned/device pools 与 event/graph recipe；Metal 使用 shared/private storage。compute depth 由状态与 scratch hazard 决定。

## 理由与取舍

拆分 owner 隔离重工作，单写减少锁与缓存争用；handoff/ack/fence 引入额外协议和服务开销。单 shard 调度服务率有限，需增量规划和准入控制，多队列不能消除同请求 token 依赖。

预分配减少热分配/page fault，但提高常驻内存并产生容量拒绝；staging、delivery、history 和 readers 均纳入额度。Rust allocator 计数不覆盖 native driver，模拟吞吐不代表模型吞吐。

完整契约与预算见[CPU 设计](../design/cpu-runtime.md)，物理 owner 和验收缺口见[状态表](../design/status.md)。
