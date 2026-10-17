# 文档导航

文档说明当前接口、操作方法与能力边界。实现细节以源码为准，命令参数以 `--help` 为准。

## 后续工作

先读 [工程整理与推理优化路线图](plans/README.md)。该页统一维护 P0–P3 优先级、任务编号、启动条件、并行范围与完成标准；按任务进入详细方案，不必逐份阅读历史研究。工程基础、Agent 服务、推测解码与 CUDA 性能方案归档在 `plans/` 的对应分组。

Agent 接入、tool parser、文本流式、max length 和实用配置的缺口与设计见 [Agent 工具调用与服务配置方案](plans/serving/agent-api.md)。该方案按实际使用场景取舍，借鉴 vLLM/SGLang 的机制；当前接口范围仍以使用文档与源码为准。

准备测试或性能对比时，先读 [性能基线与 vLLM 对齐方案](plans/performance/baseline.md)。release、当前配置对齐、矩阵、计时统计和证据规则只在该文维护；模块方案补充自己的接口与数值/资源验收。基线验收完成前，历史矩阵只用于调查线索。

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

## 测试与测量

| 文档 | 内容 |
|---|---|
| [MoE 单卡验证](guides/moe-validation.md) | 5090 缩小模型、独立参考与当前支持范围 |
| [质量门禁](guides/quality-gates.md) | 本地/CI 检查、Attention 对照与验收要求 |
| [CPU 性能测量](guides/cpu-performance.md) | 分配计数、延迟与测量范围 |
| [CUDA 性能测量](guides/cuda-performance.md) | 现有算子/模型/服务工具、能力探针与测量边界 |
| [性能基线与 vLLM 对齐方案](plans/performance/baseline.md) | 待实施的统一测试条件、比较与验收契约 |
| [Serving 稳定性调查](guides/serving-stability.md) | 稳定性机制、接口和生产差距调查 |
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

## 审查与研究

| 文档 | 内容 |
|---|---|
| [近期 CUDA 提交审查](reviews/cuda-commits-2026-10-09.md) | 2026-10-09 基点的源码缺陷、优化判断与验证范围 |
| [vLLM 对比测试审查](reviews/vllm-benchmark-methodology-2026-10-09.md) | 配置、计时、数值与统计缺口；历史差距的可信范围 |
| [Infernix 性能调研](research/infernix-performance.md) | 参考机制、精度变化与收益估算 |
| [CUDA Serving 历史记录](research/cuda-serving-baseline.md) | 既有服务矩阵与瓶颈调查；不作为当前已验收基线 |
| [CUDA 性能实验记录](research/cuda-performance-experiments.md) | 历史 A/B、失败实验与测量条件 |
| [CUDA 长上下文性能结论与任务交接](research/cuda-vllm-performance-handoff-2026-10-17.md) | 已合并 PR #10、五组正式收益、当前 attention 瓶颈、失败方案、质量边界与可拆任务 |
| [vLLM main 执行路径调查](research/vllm-main-execution-paths-2026-10-12.md) | 最新源码与已安装版的边界；GDN、MTP、projection 和资源准入差距 |
| [最新 vLLM main 服务对照](research/vllm-main-serving-comparison-2026-10-12.md) | 官方 3fe917ab 构建与五组诊断、后台资格和热复用差异；未达到性能目标 |
| [当前 GPU 执行路径](research/cuda-current-gpu-paths-2026-10-12.md) | 官方 main 更新、预热后 GDN/MTP 轨迹、小投影逐位反例与采集完整性边界 |
| [GPU draft 采样实验](research/cuda-device-greedy-mtp-2026-10-12.md) | 硬件自适应 reducer、完整输出验证和六组配对；未默认合入的原因 |
| [完整 GPU token readout](research/cuda-full-device-token-readout-2026-10-12.md) | 普通/推测/pooled 输出契约、硬件 tuning、九项 GPU 验证和五组有效服务配对 |
| [Pooled 图几何与准入](research/cuda-pooled-admission-geometry-2026-10-12.md) | capture/workspace/预算一致性、五组批量吞吐收益及 TTFT/TPOT 取舍 |
| [ReplaySSM F32 GDN 探索](research/cuda-replayssm-gdn-2026-10-12.md) | F32 小窗口缓存、完整周期成本、推测/前缀契约；尚未接入 serving |
| [内存换推理效率](research/cuda-memory-for-latency-2026-10-12.md) | 主机前缀缓存、共享 prefill 工作区与传输成本验收 |
| [GEMM 完整图成本与计时纠错](research/cuda-gemm-full-cost-and-fp4-library-2026-10-13.md) | 纯 GEMM 质量、五轮量化复用、真实权重 M256 Lt 完整成本、空图计时纠错 |
| [相邻共享输入投影量化复用](research/cuda-projection-quant-reuse-2026-10-13.md) | 扩展至 GDN qkv/z 与注意力 k/v；原质量评分通过，服务流一致性待完成 |
| [小投影解码路径一致性](research/cuda-small-projection-arithmetic-2026-10-13.md) | 2026-10-13 | 原算术差异定位；补偿和四行 F32 均通过数值/路径检查，但原 256 题质量失败；F32 局部加速仍不提升为默认 |
| [当前默认路径与 vLLM 的性能画像](research/cuda-vllm-current-profile-2026-10-13.md) | 同输入 trace：GDN 110/8 ms、attention 与分组差距；profiler 干扰和完整输出边界 |
| [从数学结构减少推理计算](research/cuda-mathematical-compute-reduction-2026-10-13.md) | 真实投影 MAC/零块审计；终端行裁剪、GDN 读出和 MLP 低秩方案，尚无新增服务收益 |
| [注意力查询共享质量门槛](research/cuda-attention-query-sharing-2026-10-13.md) | 真实 F64 全输出通过、256 题原质量失败；实验开关默认关闭 |
| [并行 WY 完整质量与解码成本](research/cuda-parallel-wy-quality-and-decode-cost-2026-10-13.md) | 两轮性能、256 题失败验收与后续 GEMM 方向 |
| [长提示调参覆盖与独立准备](research/cuda-wide-prefill-tuning-2026-10-13.md) | 宽图缓存补齐、失败候选、CUDA 成本与因果定位 |
| [F32 GDN 与完整模型质量](research/cuda-gdn-f32-quality-contract-2026-10-12.md) | F32 WY、硬件 tuning、完整模型资格与两轮性能差距；默认量化策略仍为 scalar |
| [GDN 数值与 FP8 准备成本](research/cuda-gdn-numerics-and-fp8-preparation-2026-10-12.md) | 修复分块检查假阳性、同路径 sanity control 和重复量化成本；分块仍未验收 |
| [Serial 状态快照融合](research/cuda-serial-checkpoint-fusion-2026-10-12.md) | 复用 pooled writer、逐位回滚验收和独立进程验证成本；适用范围单列 |

2026-10-09 至 10-14 的逐轮调查、实验记录和证据工件已移入[历史归档](archive/README.md)。归档保留源码基点、测量条件与失败结论供复核，不代表当前状态，也不进入日常排期。
## 维护约定

- 已实现的使用方法写入 `guides/`，跨模块契约写入 `architecture/`，已采纳决策写入 `adr/`。
- 待实施工作归入 `plans/`；任务状态、依赖和全局顺序只在路线图维护，详细方案维护本项设计与验收。性能与测试的共同条件由基线方案维护；同一规则由一个主文档维护，其余链接引用。
- 审查与研究分别归入 `reviews/`、`research/`，保留源码基点与实验条件。已完成工作更新当前使用/架构文档并更新路线图，历史实验不再承担当前排期。
- 结论被取代或不再承担当前排期的一次性实验记录移入 `archive/`，归档只改变位置、保留可追溯路径；当前导航只列仍在使用的文档。
- 运行暂存、临时报告和 profiler 输出放入 `artifacts/`；已验收基线的 manifest、输入、原始样本和可复算汇总持久保存，不能只有 `artifacts/` 副本。具体留存与更新按基线方案执行；现有 `benchmarks/` 微基准保持各自测量范围。
- 删除或移动文档时修复链接，并运行 `python3 tools/check/layout.py`。
