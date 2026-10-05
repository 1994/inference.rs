# 验证记录

本页说明当前已验证的范围与复现方式。原始报告输出到 `artifacts/`，属于本机生成结果，不随仓库发布；结论只在对应的输入、硬件和构建模式下成立。

## 自动门禁

复现命令见[质量门禁](../guides/quality-gates.md)。本机（macOS arm64 / Apple M2 Pro，Rust 1.99.0）的运行情况：

| 门禁 | 覆盖内容 |
|---|---|
| `make check-rust` | 目录与依赖规则、fmt、严格 Clippy、单元/集成测试、Rustdoc、release 构建、CPU 分配门禁与独立 golden |
| `make check-tools` | Ruff 检查与格式化、actionlint |
| `make check-security` | cargo-deny、cargo-audit 与 Gitleaks（含 Git 历史） |
| `make check-msrv` | Rust 1.90 下 workspace（不含 CUDA）与 CPU benchmark |
| `make check-linux` | macOS 上交叉编译并 lint Linux 专属代码；原生放置测试在 Linux CI 执行 |
| `make check-metal` | 真实 Metal 设备上的 CLI、Agent、HTTP/SSE、golden、页压力与固定到达负载 |

Linux 原生/NUMA 硬件门禁与远端 CI 需要对应环境，不在本机结果范围内。

## 数值、状态与服务

| 范围 | 已核对内容 |
|---|---|
| 独立模型参考 | 由官方 Transformers `Qwen3_5ForCausalLM` 导出的 [tiny](../../examples/qwen-hybrid-tiny/README.md) 与 [grouped](../../examples/qwen-hybrid-grouped/README.md) 包；覆盖 layer/hidden/logits 与 cached greedy，atol=2e-6、rtol=2e-5 |
| 加载与 forward | 分块 payload、失败清理、指纹、F32/BF16/F16 驻留与 chunk parity、图与 shape/binding、batch rollback |
| 状态 | prefix/COW/pin、容量拒绝、重算抢占、取消、quiescent checkpoint、共享页恢复与 StateRecipe 的实际 buffer 计量 |
| Workload / Agent | Generate / Embed / Rerank / Decision、投影 reference、typed discovery、源码图、隔离配置实验与实际输出验证 |
| HTTP / 观测 | 8 路并发、5-token SSE、文本 workload、schema/duplicate 错误、Prometheus/W3C/OTLP 与正常关闭 |

Metal 实机还验证了五块页池的压力场景：发生抢占与 cache eviction 后输出保持 parity，最终 active/pinned block 归零，五个 cache-owned 块可回收。固定到达短测覆盖 50/200/1000 请求每秒各 120 次，成功 120/120/86，过载档另有 23 次 HTTP 429 与 11 次客户端容量拒绝。

微型模型的数值、局部协议与指定负载不能推导完整 27B / RTX 5090 的容量。CPU 分配范围与未达标的延迟目标见 [CPU 验证](cpu.md)，未实现能力集中在[实现状态](../design/status.md)。
