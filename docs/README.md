# 文档导航

## 指南

| 文档 | 内容 |
|---|---|
| [开发与运行](guides/development.md) | 构建、CLI 命令、HTTP/SSE、checkpoint 与测量 |
| [模型执行](guides/model-execution.md) | 模型包格式、权重加载、prefill / decode |
| [新增模型](guides/adding-a-model.md) | ModelProvider 契约、注册表与接入步骤 |
| [OpenAI 兼容接口](guides/openai-api.md) | `/v1` 路由、参数与错误约定 |
| [Linux 与 CPU 放置](guides/linux.md) | cpuset / NUMA 配置与硬件门禁 |
| [质量门禁](guides/quality-gates.md) | 检查项、工具版本与执行入口 |
| [CUDA 性能基线与调优](guides/cuda-performance.md) | cuTile 基线复现、指标分层与调优规则 |

## 架构

| 文档 | 内容 |
|---|---|
| [代码布局](architecture/layout.md) | 目录职责、依赖方向与推理调用链 |
| [Runtime](architecture/runtime.md) | 执行流水线、故障排空与观测边界 |
| [调度](architecture/scheduling.md) | 队列、准入、批次选择与成本模型 |
| [请求生命周期](architecture/request-lifecycle.md) | 阶段、状态提交、取消与恢复 |
| [KV Manager](architecture/kv-manager.md) | 页所有权、StateRecipe、COW 与 prefix |
| [Backend](architecture/backends.md) | 设备能力分组、选择与新增后端 |
| [Agent](architecture/agent.md) | JSON-RPC 控制面、命令与配置实验 |

## 设计、状态与验证

| 文档 | 内容 |
|---|---|
| [技术方案](design/technical-plan.md) | 总体目标、模块契约与验收标准 |
| [性能路线](design/performance-plan.md) | 现状解剖、分阶段优化与对照 vLLM 的判定 |
| [CPU Runtime 设计](design/cpu-runtime.md) | Owner、队列、内存与性能预算 |
| [实现状态](design/status.md) | 已实现能力与剩余缺口 |
| [NVIDIA 目标](design/nvidia-targets.md) | H200 / RTX 5090 的精度与优化策略 |
| [验证记录](validation/index.md) | 已验证范围与复现方式 |
| [CPU 验证](validation/cpu.md) | 分配计数范围与延迟测量 |

## 架构决策

已采纳的决策记录见 [ADR 索引](adr/README.md)。

> 设计文档描述目标，架构文档描述当前实现，指南提供操作步骤，验证记录说明测量边界；运行日志与原始报告输出到 `artifacts/`，不随仓库发布。
