# CPU 分配与性能验证

运行 `make check-cpu`，原始数据见[allocation.json](../../artifacts/cpu-optimization/allocation.json)，源码/环境范围见[summary.json](../../artifacts/cpu-optimization/summary.json)。本机为 Apple M2 Pro / macOS；CPU 测量不执行真实模型 GPU kernels。

## 分配计数范围

| 范围 | 输入与测量 | alloc / realloc / dealloc |
|---|---|---|
| 协议 cycle | R=1/64/1024/8192、B=1/16/64、K=min(R,256)、context=1K/16K；24 组，每组预热 128、测 1,000 次 | 均为 0 |
| KV/资源 primitives | 10,000 次固定 map/arena/credit/ResourcePool、logical growth/reset、physical append/fork/COW/pin/release | 均为 0 |
| 实际 Engine tick | R/B=1/1、16/16、64/64；每组 10,000 ticks，计入首次 1024-token prefill；随机采样、反馈/history、completion 与 live cancellation | 各阶段均为 0 |

协议 cycle 使用实际 queue、ready projection、policy、独立 validation、CpuStepPool、BatchArena 与 SPSC，走 seal→apply→launch ack→synthetic fence→commit/requeue。primitives 不包含 prefix 发布和设备计算。

[Engine harness](../../tools/bench/cpu/src/engine/mod.rs)调用生产 `Engine::tick_into`，backend 为持久合约替身。计数涵盖主线程资源确认、调度、逻辑状态、提交合约、采样、反馈、回收、完成与取消；要求有限 tick 排空和 invariants 成立。

计数**不覆盖**请求解析/冷提交、线程化 workers、native GPU/driver、Full forward readout、prefix 发布、HTTP/JSON/遥测导出。它证明上述路径，不能声明全进程零分配。跨线程/fence/reader 安全另由回归及真实 Metal 检查。

## CPU 延迟目标

R=1024、B=64、K=256 的协议报告：

| Context | 全 cycle P99 | cycle/s |
|---|---:|---:|
| 1K | 174.750µs | 约 6,905 |
| 16K | 179.083µs | 约 6,939 |
| 设计目标 | ≤100µs | ≥10,000 |

两项目标尚未达标。该测量为单线程 CPU 协议，不包含跨线程 handoff、模型/driver 或网络；cycle/s 不命名为模型 tokens/s。耗时受机器负载影响，通用 CI 强制分配/正确性而不强制硬件微秒阈值。

## 内存与平台证据

managed host budget 包含常驻热表、token/page descriptors、metadata/feedback/ack、三份 vocabulary sampler scratch、物理 host mirrors 和请求峰值；history 单独限制。预算不等于 RSS/driver 上限。共享 token/result 不复制 payload，rerank 连续输入拼接仍计准备成本。

Metal 原生测试核对 StateRecipe 的私有 GPU bytes、KV block bytes、host regions 和 reset 地址稳定性。Linux OS FFI 与专属测试已 cross compile/lint；本机未执行 Linux affinity/NUMA syscall，真实池驻留与物理 first-touch 必须运行[Linux 硬件门禁](../guides/linux.md)。

全线程/native 分配、物理 ledger、生产 P99、多 compute flight 与远程 PD 的剩余范围见[状态表](../design/status.md)。
