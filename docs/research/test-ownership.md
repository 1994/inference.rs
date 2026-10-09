# 测试归属与执行器消费者清单

本文是 [测试组织方案](../plans/engineering/tests.md) 与
[删除测试后端方案](../plans/engineering/remove-test-backends.md) 的第一、二步产出：迁移前登记用例
身份、平台/feature 门控、ignore 原因，并给每个执行器消费者标记删除时的归属。

清单由 [test-inventory.py](../../tools/check/test-inventory.py) 生成并**作为门禁执行**：基准快照存于
`tools/check/test-inventory.json`，`make check` 会拒绝下列变化。

## 用例现状（427 个，`src` 已清零）

| crate | src inline | src near-test | tests/ | benches/ |
|---|---:|---:|---:|---:|
| api | 0 | 0 | 6 | 0 |
| cuda | 0 | 0 | 67 | 0 |
| kernel-api | 0 | 0 | 7 | 0 |
| metal | 0 | 0 | 19 | 0 |
| observe | 0 | 0 | 8 | 0 |
| quality | 0 | 0 | 3 | 0 |
| runtime | 0 | 0 | 61 | 0 |
| scheduler | 0 | 0 | 34 | 0 |
| state | 0 | 0 | 16 | 0 |
| workloads | 0 | 0 | 7 | 0 |
| core | 0 | 0 | 18 | 0 |
| ir | 0 | 0 | 11 | 0 |
| spi | 0 | 0 | 3 | 0 |
| package | 0 | 0 | 80 | 0 |
| agent | 0 | 0 | 12 | 0 |
| cli | 0 | 0 | 17 | 0 |
| frontdoor | 0 | 0 | 58 | 0 |
| **total** | **0** | **0** | **427** | **0** |

起始 288 个正文在 `src`，现已全部搬到各 crate 的 `tests/unit/`：**`src` = 0**，
`tests/` = 427，仍以 `#[cfg(test)]` + `#[path]` 挂在被测模块下编译，私有访问与编译边界不变。

## 门控与 ignore

| 类别 | 数量 | 含义 |
|---|---:|---|
| `#[cfg(feature = "test-backends")]` | 3 | 只有启用测试后端才编译；删除执行器时必须一并迁移或改写 |
| `all(target_os = "linux", feature = "cuda")` | 0 | 生产设备用例 |
| `target_os = "macos"` | 1 | Metal 用例 |
| 需要 GPU/CUDA fixture 的 `#[ignore]` | 40 | 必须由具名 device suite 显式选择，不能靠"全 ignored"算通过 |

## 执行器消费者与删除归属（69 个文件）

| 归属 | 数量 | 该归属下断言的去向 |
|---|---:|---|
| `protocol` | 15 | 协议/状态场景：准入、背压、取消、完成身份、HTTP/SSE、actor。断言针对正式 runtime/state/actor 行为，用脚本化桩替代完整模型计算 |
| `service` | 25 | CLI 与服务路径：帮助、doctor、错误路径可无 GPU 测试；成功推理与网络服务移到真实设备 suite |
| `numeric` | 8 | 数值对照：单算子小型独立公式、官方 golden、真实设备验收；不允许用被测 kernel 生成 expected |
| `benchmark` | 2 | CPU benchmark：保留正式 CPU 协议与分配测量，明确排除模型执行 |
| `fixture` | 7 | 两个执行器自身与其测试，随删除一并移除 |
| `plumbing` | 12 | 构建与 feature 接线：manifest、IR feature、门禁与打包脚本 |

逐文件列表见快照的 `executor_consumers`（含 `role` 字段）。门禁比较的是**依赖面**（哪个 crate
提到哪些模式），不是文件路径或文件数：迁移把正文拆到 `tests/unit/` 不算新增依赖，只有新的
crate × 模式组合会被拒绝；同时**任何没有归属的消费者都会让门禁失败**，强制这份删除工作清单保持完整。

删除顺序按方案：先 `protocol` 场景（kernel registry、runtime、agent，再 frontdoor/CLI），再
`numeric`（CUDA examples 的 host 对照、CPU benchmark、Metal 验收对测试版 CLI 的调用），最后删
`fixture` 与 `plumbing`。`infer-backend-host`/`infer-backend-reference`、`test-backends`、
`TestCpu` 的残留只允许出现在明确拒绝它们的回归测试与历史说明里。

## 门禁拒绝的变化

1. **`src` 里的用例正文增加**（迁移方向只减不增）。
2. **`#[path]` 挂载消失或重复**。
3. **测试文件没有任何模块挂载**：`tests.rs`、`*_tests.rs`、`*_check.rs` 的用例会静默不运行。
4. **出现新的执行器依赖面**，或**消费者没有删除归属**。
5. **收集总数下降**：`--record` 会同时记下新总数，避免"文件少了"被当成进展而丢掉断言。
6. **集合被静默移出**：`tests/unit|support/` 存在却没有 `autotests = false`，或 `autotests = false`
   而 `tests/*.rs` 没有 `[[test]]` 声明。

## 迁移记录

| crate | 文件 | 收集数 |
|---|---|---:|
| `infer-spi` | 1 | 3 → 3 |
| `infer-ir` | 1（另补登 `hardware`、`tokens` 两个 `[[test]]`） | 28 → 28（含 core） |
| `infer-core` | 13 | 28 → 28（含 ir） |
| `infer-models`、`infer-scheduler`、`infer-workloads` | 16（11 + 2 + 3） | 121 → 121（13 target） |
| `infer-runtime`、`infer-frontdoor`、`infer-agent` | 13（6 + 6 + 1） | 131 → 131（16 target） |
| `infer-state`、`infer-observe`、`infer-quality`、`infer-gpu-api`、`infer-kernel-api`、`infer-cli` | 12（2+1+1+2+2+4） | 46 → 46（12 target） |
| `infer-backend-cuda`、`infer-backend-metal` | 36（33 + 3） | 67 → 67（`--features cuda --lib --list`） |

六个操作要点，后续批次沿用：

1. **移动后要用 `rustfmt` 直接格式化新文件**（`cargo fmt --all` 只报不写）。
2. **改 `autotests` 的 crate 要同时登记既有集成 target**，否则用例被静默移出收集。
3. **抽取内联块不能只数花括号**（测试数据里的 JSON 会打乱深度）。
4. **重写挂载要按解析结果比对 `#[path]` 的值**，不能只比文件名。
5. **`#[path]` 会把文件位置与模块树解耦**，找不到 owner 时要回退到全树扫描解析结果。
6. **内联块的 `cfg` 可能是 `all(test, ...)`**，抽取时要保留原属性并把 `#[path]` 插在 `mod` 前。

## 待处理的既有 lint：`check-cuda` 的 clippy

`make check-cuda` 跑 `cargo clippy -p infer-backend-cuda --features cuda --all-targets -- -D warnings`，
迁移前后都报同样 11 条（用 `git worktree` 在迁移前 HEAD 实测 12 条错误），全部在
`resident/recurrent_prefill` 的测试正文里：5 条 `too_many_lines`、2 条 `float_cmp`、
2 条 `redundant_clone`、2 条 `cast_precision_loss`。`policy.py` 禁止豁免 `too_many_lines`，那 5 条
只能拆函数；其余按其建议改。独立一批，不改断言。

## 替换第一批的实测阻塞：CPU benchmark 需要两份声明

按方案从 `benchmark` 与 `protocol` 入手尝试替换时，在最小的消费者（CPU benchmark，2 个文件）
上就撞到两个必须先解决的契约问题。尝试的改动已完整回滚，工作区保持绿色。

1. **`DeviceBackend` 没有非设备变体。** 删掉 `TestCpu`（feature-gated）之后，枚举只剩
   `Cuda(NvidiaCapabilities)` 与 `Metal(MetalCapabilities)`。CPU benchmark 的后端是一个不执行
   模型的"设备契约替身"，原来靠 `infer_ir::testing::reference_capabilities()` 报告
   `TestCpu`。方案里写的替代是"CUD/Metal 描述样例，只验证元数据与选择规则，不宣称设备存在"，
   仓库已有先例（`crates/backend/kernel-api/tests/unit/registry.rs` 手写 `NvidiaCapabilities`）。
   本轮验证过：把 `DeviceCapabilities` 逐字段写成 CUDA 样例可以编译，同时保留原来的
   `compute_dtypes`/`memory_bytes`/`unified_memory` 值，测量口径不变。
2. **还需要一份声明的 kernel 集合。** 用显式 `ModelIr` 描述符替换 `ReferenceModel::fixture` 后，
   bench 在运行时报 `no compatible kernel for TokenEmbedding with F32`——引擎会按注册表校验
   模型的算子。`ReferenceKernels` 之前同时提供了"模型"和"kernel"，所以这个消费者的替代要做两件事：
   显式模型描述符 + 一个只声明所需算子（不含执行逻辑）的 kernel provider 桩。

结论：`benchmark`/`protocol` 的替换不是逐文件改写，而是先落一个**声明式协议桩**（capability
样例 + kernel 声明 + 脚本化完成结果），再让这些消费者改用它。这也说明"先清点、再替换"的顺序是
对的：这两点在删除 `TestCpu` 与 `testing` 模块之前必须先定，否则消费者没有可声明的后端身份。
