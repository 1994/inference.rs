# 测试归属与执行器消费者清单

本文是 [测试组织方案](../plans/engineering/tests.md) 与
[删除测试后端方案](../plans/engineering/remove-test-backends.md) 的第一步产出：迁移前把现有用例
身份、平台/feature 门控、ignore 原因和两个 CPU 测试执行器的消费者登记下来，供逐 crate 迁移和对账。

清单由 [test-inventory.py](../../tools/check/test-inventory.py) 生成并**作为门禁执行**：基准快照存于
`tools/check/test-inventory.json`，`make check` 会拒绝下列变化。

## 现状（427 个用例）

| crate | src inline | src near-test | tests/ | benches/ |
|---|---:|---:|---:|---:|
| cuda | 33 | 34 | 0 | 0 |
| observe | 0 | 8 | 0 | 0 |
| api | 0 | 6 | 0 | 0 |
| state | 1 | 5 | 10 | 0 |
| metal | 2 | 1 | 16 | 0 |
| quality | 0 | 3 | 0 | 0 |
| cli | 2 | 0 | 15 | 0 |
| kernel-api | 0 | 0 | 7 | 0 |
| runtime | 0 | 0 | 61 | 0 |
| scheduler | 0 | 0 | 34 | 0 |
| workloads | 0 | 0 | 7 | 0 |
| core | 0 | 0 | 18 | 0 |
| ir | 0 | 0 | 11 | 0 |
| spi | 0 | 0 | 3 | 0 |
| package | 0 | 0 | 80 | 0 |
| agent | 0 | 0 | 12 | 0 |
| frontdoor | 0 | 0 | 58 | 0 |
| **total** | **38** | **57** | **332** | **0** |

`src` 合计 **95** 个（inline 38 + 就近测试文件 57），
`tests/` **332** 个。起始为 288 个正文在 `src`，现已完成 193 个。

### 仍留在 `src` 的 crate

| crate | inline | 就近测试文件 |
|---|---:|---:|
| `cuda` | 33 | 34 |
| `observe` | 0 | 8 |
| `api` | 0 | 6 |
| `state` | 1 | 5 |
| `metal` | 2 | 1 |
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
3. **出现新的执行器消费者**（按依赖面判定，不按路径）。
4. **收集总数下降**：`--record` 会同时记下新总数，避免"文件少了"被当成进展而丢掉断言。
5. **集合被静默移出**：`tests/unit|support/` 存在却没有 `autotests = false`，或 `autotests = false`
   而 `tests/*.rs` 没有 `[[test]]` 声明 —— 两者都会让用例从 `cargo test` 里消失而不报错。

## 迁移记录

| crate | 内容 | 收集数 |
|---|---|---:|
| `infer-spi` | 3 个用例移到 `tests/unit/resource.rs` | 3 → 3 |
| `infer-ir` | `state_recipe/tests.rs`；补登 `hardware`、`tokens` 两个 `[[test]]` | 28 → 28（含 core） |
| `infer-core` | 8 个就近文件 + 5 个内联块 | 28 → 28（含 ir） |
| `infer-models`、`infer-scheduler`、`infer-workloads` | 16 个文件（11 + 2 + 3） | 121 → 121（13 target） |
| `infer-runtime`、`infer-frontdoor`、`infer-agent` | 13 个文件（6 + 6 + 1） | 131 → 131（16 target） |
| `infer-state`、`infer-observe`、`infer-quality`、`infer-gpu-api`、`infer-kernel-api`、`infer-cli` | 10 个文件（2+1+1+2+2+4） | 53 → 53（14 target） |

五个操作要点，后续批次必须照做：

1. **移动后要用 `rustfmt` 直接格式化新文件**（`cargo fmt --all` 只报不写）。
2. **改 `autotests` 的 crate 要同时登记既有集成 target**，否则用例被静默移出收集。
3. **抽取内联块不能只数花括号**（测试数据里的 JSON 会打乱深度）。
4. **重写挂载要按解析结果比对 `#[path]` 的值**，不能只比文件名。
5. **`#[path]` 会把文件位置与模块树解耦**：`kernel-api` 的 `src/attention_tests.rs` 由
   `src/attention.rs` 挂载，找不到 owner 时要回退到全树扫描 `#[path]` 的解析结果。

## 迁移顺序

按方案与路线图：foundation/model/scheduler/workloads 与 runtime/frontdoor/agent 已完成；接下来是
GPU 私有测试（`cuda` 剩 67 个、`metal` 剩 3 个）、examples 与性能场景；之后是删除两个 CPU 测试执行器
与 `test-backends` feature。每批只改目录与收集，不改数值公式或产品行为；完成后用
`python3 tools/check/test-inventory.py --record` 下调基准并刷新本文表格。
