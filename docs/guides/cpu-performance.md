# CPU 分配与性能测量

```sh
make bench
```

门禁运行隔离的 release benchmark（`tools/bench/cpu`），结果写入
`artifacts/cpu-completion/allocation.json`。它测量 CPU 控制面，不执行真实模型 GPU kernel。

## 分配门禁

| 范围 | 测量内容 |
|---|---|
| 协议 cycle | R=1/64/1024/8192、B=1/16/64、K=min(R,256)、context=1K/16K；24 组，预热 128 次、测量 1000 次 |
| KV / 资源 primitives | 10000 次 map/arena/credit/ResourcePool、logical growth/reset、physical append/fork/COW/pin/release |
| Engine tick | R/B=1/1、16/16、64/64；每组 10000 ticks，含首次 1024-token prefill、采样、反馈、completion 与取消 |

上述范围要求 alloc / realloc / dealloc 均为零。协议 cycle 使用真实 queue、ready projection、policy、validation、CpuStepPool、BatchArena 与 SPSC，设备完成由 synthetic fence 提供。
[Engine harness](../../tools/bench/cpu/src/engine/mod.rs) 调用生产 `Engine::tick_into`，backend 为持久合约替身，并检查有限 tick 排空与 invariants。

计数不覆盖请求解析、冷提交、线程化 worker、native GPU/driver、Full forward readout、prefix 发布、HTTP/JSON 与遥测导出。局部门禁通过不能描述为全进程零分配。

## 延迟与容量

报告中的 cycle/s 是 CPU 协议吞吐，不能称为模型 tokens/s。普通 CI 强制分配与正确性，不强制依赖硬件的微秒阈值；生产 P99 与 goodput 必须在目标机器上连同模型、driver、线程交接和网络一起测量。

保存机器与工具链、请求数 R、batch B、候选窗口 K、上下文、预热、原始样本、P50/P95/P99 和运行时配置。跨机器、单线程协议与真实服务之间不直接比较延迟。

managed host budget 包含常驻热表、token/page descriptor、metadata/feedback/ack、sampler scratch、物理 host mirror 与请求峰值；history 单独限制。该预算不等于 RSS 或 driver allocator 上限。rerank 的连续输入拼接计入 preparation 成本。

资源协议见 [CPU 架构](../architecture/cpu-runtime.md)；页面驻留、first-touch 与 affinity/NUMA 验收见 [Linux 指南](linux.md)。
