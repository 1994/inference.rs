# 文档导航

文档说明当前接口、操作方法与能力边界。实现细节以源码为准，命令参数以 `--help` 为准。

## 使用与开发

| 文档 | 内容 |
|---|---|
| [开发与运行](guides/development.md) | 构建、CLI、HTTP/SSE、checkpoint 与测量 |
| [构建与打包](guides/packaging.md) | Zig 平台矩阵、归档、校验与 GitHub CI |
| [模型执行](guides/model-execution.md) | 模型包、权重、prefill/decode |
| [新增模型](guides/adding-a-model.md) | ModelProvider、执行图与注册 |
| [图像模型](guides/vision.md) | 视觉塔、MRoPE、独立参考与支持限制 |
| [OpenAI 兼容接口](guides/openai-api.md) | 路由、参数与错误约定 |
| [Linux 放置](guides/linux.md) | cpuset、NUMA 与硬件验收 |

## 测试与性能

| 文档 | 内容 |
|---|---|
| [质量门禁](guides/quality-gates.md) | 本地/CI 检查、Attention 对照与验收要求 |
| [CPU 性能测量](guides/cpu-performance.md) | 分配计数、延迟与测量范围 |
| [CUDA 性能测量](guides/cuda-performance.md) | 算子/模型/服务基线、能力探针与调优 |
| [性能优化后续计划](guides/performance-roadmap.md) | 当前差距、优化顺序、数值与服务性能验收门禁 |
| [工具导航](../tools/README.md) | 门禁、参考导出与基准入口 |

## 架构

| 文档 | 内容 |
|---|---|
| [代码布局](architecture/layout.md) | 职责、依赖方向、模型与 kernel 边界 |
| [Runtime](architecture/runtime.md) | 执行流水线、故障与观测 |
| [CPU 资源协议](architecture/cpu-runtime.md) | owner、队列、内存、fence 与回收 |
| [调度](architecture/scheduling.md) | 就绪队列、准入与成本模型 |
| [请求生命周期](architecture/request-lifecycle.md) | 状态提交、取消与恢复 |
| [KV Manager](architecture/kv-manager.md) | 页所有权、StateRecipe、COW 与 prefix |
| [Backend](architecture/backends.md) | 设备能力、选择、扩展与支持限制 |
| [Agent](architecture/agent.md) | JSON-RPC 诊断与配置实验 |
| [架构决策](adr/README.md) | 已采纳决策及其理由 |

## 维护约定

- 使用方法写入指南，跨模块契约写入架构；同一事实只在一个主文档维护，其余链接引用。
- 文档跟随代码更新，不保留会话交接、逐轮进度、过时计划、待办清单和重复状态总表；历史通过 Git 查询。
- 原始日志、临时报告和 profiler 输出放入 `artifacts/`；可复用的机器可读基线与数据集定义放入 `benchmarks/`。保留验收门槛与已知限制，不把单机实验写成通用性能承诺。
- 删除或移动文档时修复链接，并运行 `python3 tools/check/layout.py`。
