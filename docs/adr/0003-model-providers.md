# ADR-0003：模型接入按 provider 注册，而不是代码分支

- 状态：采用
- 日期：2026-10-07

## 背景

`infer_spi::ModelProvider` 早已存在，但只返回 `ModelIr`，无法表达架构差异；实际加载路径把 Qwen 写死：

- `ModelPackage::open`、`QuantizedPackage::open` 直接调用 `QwenProvider.import_manifest`；
- 权重前缀 `["model.language_model.", "model.", ""]`、`lm_head.weight` 特例与 MTP 前缀 `mtp.` 写死在 `bind_weights`/`bind_backbone`/`bind_mtp`；
- `ImportedQwen` 把 provider 私有信息（vision、eos）塞进了通用结构。

结果是「新增一个模型」= 修改 `infer-models` 的核心代码；没有注册表，也没有按 `config.json` 的架构字段分派。参考 vLLM 的 `register_model` + `ModelRegistry`：模型类自己声明支持的架构，注册表按 `config.architectures` 解析，未注册就报错并指出扩展点。

## 决策

- **SPI 契约补全**：`ModelProvider: Send + Sync` 提供配置 `import() -> ImportedModel`、执行图 `graph(model)` / `draft_graph(model)` 与权重命名（`weight_prefixes()`/`anchor_slot()`/`weight_source()`）。`ImportedModel` 是家族的配置声明：`model`（语义 IR）、`mtp_layers`、`requirements`（设备能力需求）、`precision: PrecisionPolicy`（存储/计算策略，按设备能力解析）、`speculation: Option<SpeculationPlan>`（草稿头的前缀、层数与融合槽）、`modalities: Vec<ModalityPlan>`（非文本模态与占位 token）。引擎消费这份声明，不再需要 `if model == ...`。
- **模型执行图归模型所有**：可复用配方位于 `model/recipes`，由 provider 显式调用；compiler 不再构建 decoder，runtime 编译 backend 已绑定的图。没有配方的 provider 返回 unsupported。MTP 的 attention block 选择从 CUDA 加载器移到 Qwen provider。
- **注册表**：`infer_models::ModelRegistry` 提供 `register` / `resolve` / `builtin` / `default_registry`。解析顺序为 `architectures[]` → `model_type` → 嵌套 `text_config`，第一个匹配的已注册 provider 胜出；未知架构返回 `Unsupported` 并提示实现并注册 `ModelProvider`；重复认领同一架构在注册时返回 `InvalidInput`。
- **包层 provider 化**：`ModelPackage`（原 `QwenPackage`）与 `QuantizedPackage` 持有 `Arc<dyn ModelProvider>`。`open` 使用内置注册表，`open_with(registry, ..)` 接受调用方注册了自有 provider 的注册表。权重前缀解析、`lm_head` 特例、MTP 前缀与槽位全部来自声明，核心不再出现模型名。
- **能力与策略分离**：`CudaTarget::supports_compute` 回答"这块卡能算什么"，`PrecisionPolicy` 由模型声明"想怎么存/算"，在加载时解析（如 `[nvfp4, bf16]` 在 Blackwell 取 NVFP4、在 Hopper 取 BF16）。CUDA 侧不再有 `nvfp4_storage` 这类按架构枚举写死的策略分支。
- **注册是显式的**：Rust 没有 import 扫描或 decorator，注册在组合根完成（`register` + `open_with`），内置列表由 `ModelRegistry::builtin` 固定编译进产物。`QwenProvider` 是第一个注册者。

## 后果

- 新增模型 = 实现 `ModelProvider` + 在组合根注册；核心分派、绑定时序与预算校验不改。未知架构的报错直接指明扩展点。一个 omni 家族的模态、精度偏好、草稿头与命名都在同一个 `import` 里声明，见 `registry.rs` 的 `an_omni_family_converges_every_customization_into_one_import`。
- provider 负责自己的配置解析与 HF 张量命名，因此命名差异（前缀、共享 head、MTP 槽位）不再泄漏进通用包层；MTP 前缀与融合投影名来自 `SpeculationPlan`，CUDA 侧不再硬编码 `mtp.fc.weight`。
- `architectures()` 是静态列表，没有通配或正则；一个配置只被一个 provider 认领，重叠必须在注册期解决。
- 模态声明不自动赋予后端执行能力；图像编码与服务调度需分别验收，当前范围见[图像指南](../guides/vision.md)。
- 保留 `pub type QwenPackage = ModelPackage;` 兼容别名供外部调用迁移；仓库内加载路径已统一为 `ModelPackage`。各后端需独立通过设备验收。
- 仍未做：动态加载（C ABI/WASM）与从独立 crate 分发模型；当前扩展点在编译期组合。新增模型的操作步骤见[新增模型指南](../guides/adding-a-model.md)。
