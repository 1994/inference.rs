# 测试归属与执行器消费者清单

本文是 [测试组织方案](../plans/engineering/tests.md) 与
[删除测试后端方案](../plans/engineering/remove-test-backends.md) 的第一步产出：迁移前把现有用例
身份、平台/feature 门控、ignore 原因和两个 CPU 测试执行器的消费者登记下来，供逐 crate 迁移和对账。

清单由 [test-inventory.py](../../tools/check/test-inventory.py) 生成并**作为门禁执行**：基准快照存于
`tools/check/test-inventory.json`，`make check` 会拒绝下列变化。

## 现状（427 个用例）

| crate | src inline | src near-test | tests/ | benches/ |
|---|---:|---:|---:|---:|
| cuda | 10 | 1 | 56 | 0 |
| observe | 0 | 8 | 0 | 0 |
| api | 0 | 6 | 0 | 0 |
| state | 1 | 5 | 10 | 0 |
| quality | 0 | 3 | 0 | 0 |
| cli | 2 | 0 | 15 | 0 |
| kernel-api | 0 | 0 | 7 | 0 |
| metal | 0 | 0 | 19 | 0 |
| runtime | 0 | 0 | 61 | 0 |
| scheduler | 0 | 0 | 34 | 0 |
| workloads | 0 | 0 | 7 | 0 |
| core | 0 | 0 | 18 | 0 |
| ir | 0 | 0 | 11 | 0 |
| spi | 0 | 0 | 3 | 0 |
| package | 0 | 0 | 80 | 0 |
| agent | 0 | 0 | 12 | 0 |
| frontdoor | 0 | 0 | 58 | 0 |
| **total** | **13** | **23** | **391** | **0** |

`src` 合计 **36** 个（inline 13 + 就近测试文件 23），
`tests/` **391** 个。起始为 288 个正文在 `src`，现已完成 252 个。

### 仍留在 `src` 的 crate

| crate | inline | 就近测试文件 |
|---|---:|---:|
| `cuda` | 10 | 1 |
| `observe` | 0 | 8 |
| `api` | 0 | 6 |
| `state` | 1 | 5 |
| `quality` | 0 | 3 |
| `cli` | 2 | 0 |

## 门控与 ignore

| 类别 | 数量 | 含义 |
|---|---:|---|
| `#[cfg(feature = "test-backends")]` | 50 | 只有启用测试后端才编译；删除执行器时必须一并迁移或改写 |
| `all(target_os = "linux", feature = "cuda")` | 24 | 生产设备用例 |
| `target_os = "macos"` | 23 | Metal 用例 |
| 需要 GPU/CUDA fixture 的 `#[ignore]` | 40 | 必须由具名 device suite 显式选择，不能靠"全 ignored"算通过 |

ignore 原因原文全部记录在快照的 `ignored_reasons` 里，迁移时按方案要求逐条对应，不允许因为换目录
而丢掉断言或把设备用例降级成 host 用例。

## CPU 测试执行器的消费者（84 个文件）

| 类别 | 文件数 |
|---|---:|
| manifest | 11 |
| 源码 | 52 |
| 其它（脚本/工具） | 21 |

完整列表见快照的 `executor_consumers`。消费者按**crate + 提到哪些模式**登记而不是按路径：迁移本来
就会移动这些文件，搬走一个消费者不算新增依赖；只有同一 crate 里出现新的依赖面才会被门禁拒绝。

这批是删除 `crates/testing/cpu/host`、`crates/testing/cpu/reference` 与产品 `test-backends`
feature 之前必须逐个替换或删除的消费者，也是 [E1](../plans/README.md) "消费者替换及测试迁移" 一批
的输入。

## 门禁拒绝的变化

1. **`src` 里的用例正文增加**：迁移方向是只减不增，基准数只能通过 `--record` 在新一轮迁移后下调。
2. **`#[path]` 挂载消失或重复**：挂载必须解析到存在且唯一的文件。
3. **测试文件没有任何模块挂载**：`tests.rs`、`*_tests.rs`、`*_check.rs` 里的用例会静默不运行，
   编译器不会报错。
4. **出现新的执行器消费者**（按依赖面判定，不按路径）。
5. **收集总数下降**：`--record` 会同时记下新总数，避免"文件少了"被当成进展而丢掉断言。
6. **集合被静默移出**：`tests/unit|support/` 存在却没有 `autotests = false`，或 `autotests = false`
   而 `tests/*.rs` 没有 `[[test]]` 声明。

## 迁移记录

| crate | 内容 | 收集数 |
|---|---|---:|
| `infer-spi` | 3 个用例移到 `tests/unit/resource.rs` | 3 → 3 |
| `infer-ir` | `state_recipe/tests.rs`；补登 `hardware`、`tokens` 两个 `[[test]]` | 28 → 28（含 core） |
| `infer-core` | 8 个就近文件 + 5 个内联块 | 28 → 28（含 ir） |
| `infer-models`、`infer-scheduler`、`infer-workloads` | 16 个文件（11 + 2 + 3） | 121 → 121（13 target） |
| `infer-runtime`、`infer-frontdoor`、`infer-agent` | 13 个文件（6 + 6 + 1） | 131 → 131（16 target） |
| `infer-state`、`infer-observe`、`infer-quality`、`infer-gpu-api`、`infer-kernel-api`、`infer-cli` | 10 个文件（2+1+1+2+2+4） | 53 → 53（14 target） |
| `infer-backend-cuda`、`infer-backend-metal` | 29 个文件（26 + 3） | 67 → 67（`--features cuda --lib --list`） |

六个操作要点，后续批次必须照做：

1. **移动后要用 `rustfmt` 直接格式化新文件**（`cargo fmt --all` 只报不写）。
2. **改 `autotests` 的 crate 要同时登记既有集成 target**，否则用例被静默移出收集。
3. **抽取内联块不能只数花括号**（测试数据里的 JSON 会打乱深度）。
4. **重写挂载要按解析结果比对 `#[path]` 的值**，不能只比文件名。
5. **`#[path]` 会把文件位置与模块树解耦**：找不到 owner 时要回退到全树扫描 `#[path]` 的解析结果。
6. **迁移不改测试正文**：`infer-backend-cuda` 在 `--features cuda` 下的 lint 发现与本次迁移无关，
   迁移前后同一份正文、同样 11 条。分开处理，逐条记录在下面。

### 待处理的既有 lint：`check-cuda` 的 clippy

`make check-cuda` 会跑 `cargo clippy -p infer-backend-cuda --features cuda --all-targets -- -D warnings`。
在迁移前后都一样报 11 条（用 `git worktree` 在迁移前的 HEAD 上实测同样 12 条错误、EXIT=101），
全部落在 `resident/recurrent_prefill` 的测试正文里：5 条 `too_many_lines`、2 条 `float_cmp`、
2 条 `redundant_clone`、2 条 `cast_precision_loss`。

`tools/check/policy.py` 明确禁止豁免 `clippy::too_many_lines`（"strict lint checks cannot be waived"），
所以这 5 条只能拆函数，不能加 `#[expect]`；其余 4 类按其建议改（去掉多余 clone、按位比较浮点、避免
精度丢失的转换）。这是独立一批，不与目录迁移混做，且不改变任何断言。

## 迁移顺序

按方案与路线图：foundation/model/scheduler/workloads、runtime/frontdoor/agent 与 GPU 私有测试已完成；
接下来是 examples 与性能场景；之后是删除两个 CPU 测试执行器与 `test-backends` feature。每批只改目录
与收集，不改数值公式或产品行为；完成后用 `python3 tools/check/test-inventory.py --record` 下调基准
并刷新本文表格。
