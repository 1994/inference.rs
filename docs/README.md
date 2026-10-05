# 文档导航

| 阅读目的 | 文档 |
|---|---|
| 构建、运行和请求接口 | [开发指南](guides/development.md) |
| Linux CPU/NUMA 配置 | [Linux 指南](guides/linux.md) |
| 代码检查与工具要求 | [质量门禁](guides/quality-gates.md) |
| 加载模型与执行 forward | [模型执行](guides/model-execution.md) |
| 目录、职责和依赖 | [代码布局](architecture/layout.md) |
| 线程、交接和观测 | [Runtime](architecture/runtime.md) |
| 队列与批次选择 | [调度](architecture/scheduling.md) |
| 请求状态、取消和 checkpoint | [请求生命周期](architecture/request-lifecycle.md) |
| KV 所有权、页池和 prefix | [KV Manager](architecture/kv-manager.md) |
| 设备能力与扩展 backend | [Backend](architecture/backends.md) |
| Agent 命令与实验 | [Agent](architecture/agent.md) |
| 总体目标与模块契约 | [技术方案](design/technical-plan.md) |
| CPU 性能与内存设计 | [CPU 设计](design/cpu-runtime.md) |
| 已实现与剩余工作 | [实现状态](design/status.md) |
| 检查证据与测量边界 | [验证记录](validation/index.md)、[CPU 验证](validation/cpu.md) |
| 已采用的架构决策 | [ADR](adr/README.md) |

设计描述目标，架构描述代码，指南提供操作，验证记录保存测量范围。运行日志与原始报告保存在 `artifacts/`。
