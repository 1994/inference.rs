# 验证记录

证据基线：2026-10-05，macOS 27.0 arm64 / Apple M2 Pro、Rust 1.99.0；最低版本检查为 Rust 1.90。原始报告保留其源码/二进制指纹，不随文档更新重写。

## 自动门禁

[汇总报告](../../artifacts/cpu-optimization/summary.json)记录本机分项结果：

| 门禁 | 结果与日志 |
|---|---|
| Rust / CPU | 通过；[rust-gate.log](../../artifacts/cpu-optimization/rust-gate.log)，包含 policy、fmt、严格 Clippy、测试、Rustdoc、release 与分配门禁 |
| 工具 | 通过；[tools-gate.log](../../artifacts/cpu-optimization/tools-gate.log) |
| 依赖/凭证 | 通过；[security-gate.log](../../artifacts/cpu-optimization/security-gate.log)；原有 paste 例外未扩大 |
| MSRV | 通过；[msrv-gate.log](../../artifacts/cpu-optimization/msrv-gate.log) |
| Linux 跨编译 | Clippy/MSRV 通过；[linux-cross-gate.log](../../artifacts/cpu-optimization/linux-cross-gate.log)、[linux-msrv-gate.log](../../artifacts/cpu-optimization/linux-msrv-gate.log) |
| Metal 硬件 | 通过；[metal-gate.log](../../artifacts/cpu-optimization/metal-gate.log) |

这些是本机分项证据；Linux 原生/NUMA 硬件与远端 CI 未在该报告中执行。复现命令见[质量门禁](../guides/quality-gates.md)。

## 数值、状态与服务

| 范围 | 已核对内容 |
|---|---|
| 独立模型参考 | 官方 Transformers Qwen3_5ForCausalLM 导出的 [tiny](../../examples/qwen-hybrid-tiny/README.md)、[grouped](../../examples/qwen-hybrid-grouped/README.md)；layer/hidden/logits/cached greedy，atol=2e-6、rtol=2e-5 |
| 加载/forward | 分块 payload/失败清理/指纹、F32/BF16/F16 驻留与 chunk parity、图/shape/binding、batch rollback |
| 状态 | prefix/COW/pin、容量拒绝、重算抢占、取消、quiescent checkpoint、共享页恢复与 StateRecipe 实际 buffer 计量 |
| Workloads/Agent | Generate/Embed/Rerank/Decision、投影 reference、typed discovery、源码图、隔离配置实验与实际输出验证 |
| HTTP/观测 | 8 路并发、5-token SSE、文本 workloads、schema/duplicate、Prometheus/W3C/OTLP、正常关闭 |

[Metal 原始结果](../../artifacts/cpu-optimization/metal-results.json)还记录五块页池压力：发生抢占和 cache eviction，输出保持 parity；最终 active/pinned blocks=0，五块 cache-owned 块可回收。

固定到达短测为 50/200/1000 请求每秒、各 120 次：成功 120/120/86，过载档另有 23 次 HTTP 429 和 11 次客户端容量拒绝。较长的 [9,000 请求报告](../../artifacts/cpu-completion/open-loop-soak.json)属于早先指纹基线；其结果由[对应汇总](../../artifacts/cpu-completion/summary.json)绑定，不混入当前版本性能成绩。

微型模型数值、局部协议和指定负载不推导完整 27B / RTX 5090 容量。CPU 分配范围与未达延迟目标见[CPU 验证](cpu.md)，未实现目标集中在[状态表](../design/status.md)。
