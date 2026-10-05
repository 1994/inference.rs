# ADR-0002：CPU 与 GPU 提交流水线

- 状态：采用
- 日期：2026-10-04
- 设计项：ARCH-001 / SCHED-001 / COST-001 / CUDA-001 / OBS-001 / BENCH-001

## 背景

GPU 执行期间 CPU 侧仍要准备其他请求、采样输出并维护状态。共享可变状态、无界队列和隐式同步会把网络等待、调度与设备提交耦合在一起，使延迟不可预测，也难以验证资源是否真正回收。

## 决策

1. 每个 GPU 一个 SchedulerOwner 与 DeviceSubmitOwner；准备、输出、交付和观测使用独立有界 worker，网络使用 Tokio。
2. Scheduler 单写请求、队列与逻辑状态；物理决策 ledger 的目标 owner 也是 scheduler，设备只维护执行镜像。
3. 热表、token/metadata/output、arena、planning scratch 与确认池常驻；队列传 generational handle，MPSC/SPSC 按实际生产者数量选择。
4. ReadyDelta 与有界窗口驱动规划；decode 只提交增量 token/page，普通生成使用最小 readout，CPU sampling 移入 output pool。
5. 发布、launch ack、GPU completion、CPU consume 分别确认；取消/shutdown 独立于 bulk 流量，满队列不吞控制意图。
6. owner 通过可靠通知与有界 drain/adaptive poll 唤醒；Linux 支持 affinity/NUMA，macOS 只提供系统提示。
7. CUDA 使用持久 pinned/device pool 与 event/graph recipe，Metal 使用 shared/private storage；compute depth 由状态与 scratch hazard 决定。

## 后果

拆分 owner 隔离了重工作，单写减少了锁与缓存争用；代价是 handoff、ack 与 fence 引入额外协议和调度开销。单 shard 的调度服务率有限，需要增量规划与准入控制；多队列不能消除同一请求内的 token 依赖。

预分配减少热分配与 page fault，但提高常驻内存并可能产生容量拒绝；staging、delivery、history 与 reader 都纳入额度。Rust allocator 计数不覆盖 native driver，模拟得到的吞吐不代表模型吞吐。

完整契约与预算见 [CPU Runtime 设计](../design/cpu-runtime.md)，物理 owner 与验收缺口见[实现状态](../design/status.md)。
