# 新增 backend

状态：Tier-0 接入清单与共享契约；一致性套件当前覆盖 manifest、状态报价/生命周期与 program 身份，
数值、分页与取消场景由后续迭代补齐。设计取舍、盘点数据与分阶段计划见
[Backend 标准化与接入 SOP 研究](../research/backend-standardization-2026-10-25.md)。

本文面向要接入新设备执行器的实现者。目标是让新 backend 只写设备层、kernel 与少量适配，
不复制 CUDA 的常驻图/调优，也不重新实现仓库里已有的装载、状态和测试骨架。

## 1. 先定层级

| 层级 | 首版必须 | 可以显式不做 |
|---|---|---|
| Tier-0 功能后端 | F32 dense 图、目标 dense 算子子集、paged KV、prefix、prefill/decode、checkpoint、CLI 可选 | MoE、融合、MTP、视觉、量化、常驻图、自动调优 |
| Tier-1 性能后端 | Tier-0 全部，加融合、常驻图、调优、量化 KV/权重、BF16/FP16 profile | 仍须通过 Tier-0 一致性套件 |
| Tier-2 模型特性 | 视觉塔、MTP/draft、模型专属融合 | 通过 capability 与命令入口显式报 `Unsupported` |

**参照实现：** Tier-0 看 [Metal](../../crates/backend/metal/src/lib.rs)，Tier-1 看
[CUDA](../../crates/backend/cuda/src/lib.rs)。不要把 CUDA 的 `resident/` 与 `tuning/` 当作接入基线。

开始前先写清楚四件事，它们决定 kernel 与状态布局：

1. 首个目标模型族与算子子集；
2. 设备内存模型（独显 / 统一内存 / 多级显存）与是否需要 pinned/host mirror；
3. 首版是否需要 checkpoint、prefix 与连续批处理；
4. 是否自带 GEMM/attention 库，还是先写朴素 kernel。

## 2. 注册能力与后端判别

新增 variant 后按编译器提示逐处补齐，通常涉及：

- `crates/foundation/ir/src/hardware/`：`BackendKind`、`DeviceBackend`、能力/要求结构；
- `crates/service/cli/src/backend/`：选择、`SelectedBackend` / `SelectedTicket` 转发与 adapter；
- `tools/check/layout.py` 的 `BACKENDS` 集合；
- `README.md`、[Backend 能力与扩展](../architecture/backends.md)、[Backend 分组](../../crates/backend/README.md) 的支持表。

公共结构不添加厂商 `Option` 字段；跨 backend 的能力要求由 `CapabilityRequirements` 表达，
不匹配时 `DeviceCapabilities::require` 必须拒绝，而不是悄悄退回别的 backend。

## 3. 创建 backend crate

按 `crates/backend/<name>` 建 crate，feature gate 设备依赖，参考
[Metal manifest](../../crates/backend/metal/Cargo.toml) 与
[CUDA manifest](../../crates/backend/cuda/Cargo.toml)。生产依赖只允许 foundation、model、
engine 契约与 kernel-api；不要依赖 CLI、agent 或 frontdoor。

```toml
[dependencies]
infer-core.workspace = true
infer-ir.workspace = true
infer-spi.workspace = true
infer-models.workspace = true
infer-compiler.workspace = true
infer-model-recipes.workspace = true
infer-state.workspace = true

[target.'cfg(target_os = "…")'.dependencies]
# 设备 FFI / 运行时
```

`lib.rs` 只做模块声明与导出，不写实现；设备 buffer、ticket 与 unsafe FFI 留在本 crate。

## 4. Kernel 清单

实现 `KernelProvider`，为每个可执行 `Operation` 声明一条注册。当前 `KernelRegistration`
仍以 `max_shape_elements` 为主，S1 的声明式适用域（dtype、head_dim、mask、KV 布局）尚未落地；
在此之前遵守两条规则：

1. 只注册真正能执行的算子与精度，宁可在加载期报 `Unsupported`，不要注册后靠内部 fallback；
2. 每条注册的 `backend`、`requirements` 与 `source` 必须真实，`source` 指向实际实现。

最小形态（`my_backend_kind` / `my_kernel_id` / `source_location` 由本 backend 提供）：

```rust
impl KernelProvider for MyKernels {
    fn kernels(&self) -> Vec<KernelRegistration> {
        [Operation::TokenEmbedding, Operation::MatMul, Operation::RmsNorm, Operation::Attention]
            .into_iter()
            .enumerate()
            .map(|(index, operation)| KernelRegistration {
                backend: my_backend_kind(),
                id: my_kernel_id(index), // 本 backend 内的稳定命名空间
                operation,
                precision: PrecisionPlan::f32(),
                requirements: CapabilityRequirements::default(),
                max_shape_elements: usize::MAX, // 换成真实上界
                workspace_bytes: 0,
                priority: 0,
                estimated_ns: 1,
                source: source_location(operation),
            })
            .collect()
    }
}
```

`KernelRegistry` 目前的选择等价于“注册表成员检查”；新增 backend 不要依赖 `priority` /
`estimated_ns` 做选型，同 op 多实现应在 backend 内按几何选择，等 S1 落地后再上报适用域。

## 5. 权重与状态：用共享契约

### 5.1 权重

用 `infer_models::{ModelPackage, load_weights, WeightTarget}` 装载，backend 只实现
`allocate` 与 `upload`。参考 [Metal loading](../../crates/backend/metal/src/executor/loading.rs)。
不要自写权重命名、分片、精度转换与字节预算；量化/专家/视觉作为 Tier-1 扩展。

### 5.2 状态

用 [`StateRecipe`](../../crates/foundation/ir/src/state_recipe/mod.rs)：

```rust
let recipe = StateRecipe::compile(&model, &graph, block_size, element_bytes, probes, lease_bytes)?;
let layout = recipe.layout(capacity, OutputReadout::Full)?;
recipe.visit(capacity, OutputReadout::Full, |region| {
    // 按 region.memory（DevicePrivate / HostMirror / KvBlock）分配或映射
    Ok(())
})?;
```

报价、页数和 region 偏移由共享代码给出；backend 只按 region class 分配/释放。
不要手写 `kv_block_bytes * pages + hidden * capacity` 一类公式——那是报价漂移的来源。
KV 页与 prefix 默认用 [`KvCacheManager`](../../crates/engine/state/src/kv.rs)；若必须用快照语义，
在文档和一致性测试里明确标注，不要静默改变复用行为。

## 6. 实现 `BackendProvider`

`BackendProvider` 有 47 个方法，只有 6 个没有默认实现：
`identity`、`capabilities`、`validate_program`、`execution_graph`、`submit`、`poll`。
默认实现会让 prefix、容量报价、checkpoint 静默退化，因此 Tier-0 还要显式实现：

- `state_recipe` / `state_reservation_bytes_for` / `reserve_state_for` / `release_state`；
- `reusable_prefix` / `reuse_prefix`（或明确返回 0 并在支持矩阵里写明）；
- `free_state_bytes` / `state_capacity_bytes`（若参与准入）；
- `capture_execution_state` / `restore_execution_state` / `validate_state_ownership`（或显式 `Unsupported`）。

`validate_program` 必须拒绝：别的 backend、别的 model、设备不支持的 precision、未知 kernel id、
与已加载图不一致的 dataflow。`submit` 出错时不得留下设备工作；一旦提交，`poll` 必须返回 ticket
对应的完成或错误。

## 7. 图执行

首版按 `DataflowGraph` 顺序解释节点即可，参考
[Metal forward](../../crates/backend/metal/src/executor/forward.rs)。需要覆盖目标模型用到的
`TensorOp`；每个算子先用
[reference-ops](../../crates/testing/cpu/reference-ops/src/kernels.rs) 做独立数值对照，再接图级夹具。

注意：

- `Split` 的 widths/heads、`Attention` 的 GQA 分组与 KV append；
- prefill 分块与 decode 单行的 readout 行、logits 行；
- scratch 复用按 `lifetimes`/`scratch_elements`，不要按节点数线性放大；
- 不支持的 dtype/shape 在 encode 前拒绝，不要发出越界 launch。

## 8. 接一致性套件

新增 crate 已提供共享检查：`crates/testing/cpu/backend-conformance`。在 backend 的
`Cargo.toml` 增加（同时把 `infer-backend-conformance` 加进根 `workspace.dependencies`）：

```toml
[dev-dependencies]
infer-backend-conformance.workspace = true
```

在设备门禁测试里组合三项检查：

```rust
use infer_backend_conformance::{backend_contract, kernel_manifest, program_contract};

let mut report = kernel_manifest(&MyKernels, &caps);
report.extend(backend_contract(&mut backend, capacity));
report.extend(program_contract(&backend, &model, &program, &caps));
eprintln!("{}", report.summary());
report.require()?;
```

当前覆盖：

| 检查 | 捕获的问题 |
|---|---|
| `kernel_manifest` | 注册了别的 backend、能力无法选择的 kernel、重复 id、未注册算子仍可选 |
| `backend_contract` | 空 identity、能力不确定、缺少 recipe/报价、报价低于 recipe、重复 reserve、prefix 越界 |
| `program_contract` | `validate_program` 接受别的 backend/model/precision/kernel id/dataflow |

尚未覆盖、必须由 backend 自己测：数值与 reference 的容差、KV 分页与 COW、prefix 实际命中、
取消/排空、checkpoint 恢复、连续批处理。后续迭代会把它们并入套件；在此之前不要把这些项目标成“通过”。

## 9. CLI 与文档

- 实现 `crates/service/cli/src/backend/<name>.rs`，默认值/利用率/覆盖检查保持与 Metal adapter 同形；
- 在 `SelectedBackend` / `SelectedTicket` 增加 variant 并补齐转发（当前是逐方法 match，S4 宏化后只需一个 variant）；
- 更新 `tools/check/layout.py` 的 `BACKENDS`、README 支持表、
  [Backend 能力与扩展](../architecture/backends.md) 的限制表；
- 在支持矩阵里区分“已编译”“已设备验收”“性能未验收”，不要把编译通过写成生产支持。

## 10. 完成定义（Tier-0）

```text
[ ] 能力/枚举/CLI/layout.py/文档支持表已补齐
[ ] KernelProvider 只注册真实可执行的算子与精度
[ ] 权重走 load_weights + WeightTarget；状态走 StateRecipe + KvCacheManager
[ ] BackendProvider 的 6 个必需方法与状态/prefix/checkpoint 契约已实现或显式 Unsupported
[ ] 目标算子子集有 reference-ops 数值对照
[ ] 至少一个仓库夹具的 golden 输出通过
[ ] conformance 的 manifest/state/program 检查通过，失败项逐条有原因
[ ] 无 CPU fallback；未知模型/算子/精度明确报 Unsupported
[ ] 支持矩阵、已知限制、性能未验收声明已更新
```

## 11. 常见错误

| 现象 | 原因 | 修正 |
|---|---|---|
| 编译通过但加载后每一步都退化 | 依赖了 `BackendProvider` 默认实现 | 实现状态/prefix/checkpoint，或显式 `Unsupported` |
| 报价与真实分配不一致 | 自写状态字节公式 | 用 `StateRecipe::layout`，按 region 分配 |
| 换模型就改 backend | 在 backend 内按模型名/形状分支 | 图由模型 provider 提供；backend 只看 `TensorOp` 与能力 |
| attention 在长上下文失败 | 只按输出元素数限制容量 | 用 head_dim/KV 长度与真实 workspace 校验 |
| 测试覆盖很多但换设备就崩 | 只测了单个形状 | 用 reference-ops 做算子对照，再补夹具与边界 |

## 12. 本地命令

```sh
cargo fmt --all --check
cargo clippy -p <backend-crate> --all-targets -- -D warnings
cargo test -p <backend-crate>
python3 tools/check/layout.py
python3 tools/check/test-inventory.py
```

设备套件沿用现有 `make test-<backend>` 约定；没有设备时只运行 host 可检查组，并在 PR 中注明
未执行项与原因。
