# CPU 分配与性能验证

```sh
make check-cpu
```

该门禁运行隔离的 CPU benchmark（`tools/bench/cpu`），release 构建，原始结果写入 `artifacts/cpu-completion/allocation.json`。本机为 Apple M2 Pro / macOS；CPU 测量不执行真实模型的 GPU kernel。

## 分配计数范围

| 范围 | 输入与测量 | alloc / realloc / dealloc |
|---|---|---|
| 协议 cycle | R=1/64/1024/8192、B=1/16/64、K=min(R,256)、context=1K/16K；24 组，每组预热 128 次、测量 1000 次 | 均为 0 |
| KV / 资源 primitives | 10000 次固定 map/arena/credit/ResourcePool、logical growth/reset、physical append/fork/COW/pin/release | 均为 0 |
| 实际 Engine tick | R/B=1/1、16/16、64/64；每组 10000 ticks，计入首次 1024-token prefill；含随机采样、反馈/history、completion 与 live cancellation | 各阶段均为 0 |

协议 cycle 使用真实 queue、ready projection、policy、独立 validation、CpuStepPool、BatchArena 与 SPSC，走 seal → apply → launch ack → synthetic fence → commit/requeue；primitives 不包含 prefix 发布与设备计算。[Engine harness](../../tools/bench/cpu/src/engine/mod.rs) 调用生产 `Engine::tick_into`，backend 为持久合约替身，要求有限 tick 排空并满足 invariants。

计数**不覆盖**请求解析与冷提交、线程化 worker、native GPU/driver、Full forward readout、prefix 发布、HTTP/JSON/遥测导出。它只证明上述路径，不能声明全进程零分配；跨线程、fence 与 reader 安全由回归测试和真实 Metal 验证覆盖。

## CPU 延迟

R=1024、B=64、K=256 的协议报告：

| Context | 全 cycle P99 | cycle/s |
|---|---:|---:|
| 1K | 174.750µs | 约 6905 |
| 16K | 179.083µs | 约 6939 |
| 设计目标 | ≤ 100µs | ≥ 10000 |

两项目标尚未达标。该测量是单线程 CPU 协议，不包含跨线程 handoff、模型/driver 或网络，cycle/s 不能命名为模型 tokens/s。耗时受机器负载影响，普通 CI 只强制分配与正确性，不强制硬件微秒阈值。

## 内存与平台证据

managed host budget 包含常驻热表、token/page descriptor、metadata/feedback/ack、三份 vocabulary sampler scratch、物理 host mirror 与请求峰值；history 单独限制，预算不等于 RSS 或 driver 上限。共享 token/result 不复制 payload；rerank 的连续输入拼接仍计入 preparation 成本。

Metal 原生测试核对 StateRecipe 的私有 GPU bytes、KV block bytes、host region 与 reset 后的地址稳定性。Linux OS FFI 与专属测试已通过交叉编译与 lint，但本机未执行 Linux affinity/NUMA syscall；真实池驻留与物理 first-touch 必须运行 [Linux 硬件门禁](../guides/linux.md)。

全线程与 native 分配、物理 ledger、生产 P99、多 compute flight 与远程 PD 的剩余范围见[实现状态](../design/status.md)。
