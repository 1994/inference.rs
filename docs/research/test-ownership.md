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
| package | 62 | 3 | 15 | 0 |
| frontdoor | 13 | 37 | 8 | 0 |
| runtime | 8 | 5 | 48 | 0 |
| cli | 4 | 9 | 4 | 0 |
| agent | 0 | 12 | 0 | 0 |
| observe | 0 | 8 | 0 | 0 |
| kernel-api | 2 | 5 | 0 | 0 |
| workloads | 1 | 6 | 0 | 0 |
| api | 0 | 6 | 0 | 0 |
| state | 1 | 5 | 10 | 0 |
| scheduler | 1 | 3 | 30 | 0 |
| metal | 2 | 1 | 16 | 0 |
| quality | 0 | 3 | 0 | 0 |
| core | 0 | 0 | 18 | 0 |
| ir | 0 | 0 | 11 | 0 |
| spi | 0 | 0 | 3 | 0 |
| **total** | **127** | **137** | **163** | **0** |

`src` 合计 **264** 个（inline 127 + 就近测试文件 137），
`tests/` **163** 个。方案要求用例正文最终只存在于 `tests/`，因此这 264 个是本项迁移
的实际工作量。

## 门控与 ignore

| 类别 | 数量 | 含义 |
|---|---:|---|
| `#[cfg(feature = "test-backends")]` | 52 | 只有启用测试后端才编译；删除执行器时必须一并迁移或改写 |
| `all(target_os = "linux", feature = "cuda")` | 25 | 生产设备用例 |
| `target_os = "macos"` | 25 | Metal 用例 |
| 需要 GPU/CUDA fixture 的 `#[ignore]` | 40 | 必须由具名 device suite 显式选择，不能靠"全 ignored"算通过 |

ignore 原因原文全部记录在快照的 `ignored_reasons` 里，迁移时按方案要求逐条对应，不允许因为换目录
而丢掉断言或把设备用例降级成 host 用例。

## CPU 测试执行器的消费者（84 个文件）

| 类别 | 文件数 |
|---|---:|
| manifest | 11 |
| 源码 | 52 |
| 其它（脚本/工具） | 21 |

完整列表见快照的 `executor_consumers`。这些是删除 `crates/testing/cpu/host`、
`crates/testing/cpu/reference` 与产品 `test-backends` feature 之前必须逐个替换或删除的消费者，
也是 [E1](../plans/README.md) "消费者替换及测试迁移" 一批的输入。

## 门禁拒绝的变化

1. **`src` 里的用例正文增加**：迁移方向是只减不增，基准数只能通过 `--record` 在新一轮迁移后下调。
2. **`#[path]` 挂载消失或重复**：挂载必须解析到存在且唯一的文件。
3. **出现新的执行器消费者**：删除过程中不允许新增依赖。
4. **收集总数下降**：`--record` 会同时记下新总数，避免"文件少了"被当成进展而丢掉断言。
5. **集合被静默移出**（本轮新增）：`tests/unit|support/` 存在却没有 `autotests = false`，或
   `autotests = false` 而 `tests/*.rs` 没有 `[[test]]` 声明 —— 两者都会让用例从 `cargo test`
   里消失而不报错。

## 迁移记录

- **`infer-spi`**：3 个用例移到 `crates/foundation/spi/tests/unit/resource.rs`。
- **`infer-ir`**：`src/state_recipe/tests.rs` 移到 `tests/unit/state_recipe.rs`。该 crate 本来就有
  `tests/hardware.rs`、`tests/tokens.rs` 两个平台集成 target，设 `autotests = false` 后必须显式
  `[[test]]` 声明，否则它们被静默移出收集（实测收集数从 28 掉到 20，正是第 5 条要防的情况）。
- **`infer-core`**：8 个就近测试文件与 5 个内联 `mod tests` 块移到 `tests/unit/`，模块仍由被测模块
  用 `#[path]` 挂载；迁移后 `cargo test -p infer-core -p infer-ir` 收集数仍为 28。

两个操作要点，后续批次必须照做：

1. **移动后要用 `rustfmt` 直接格式化新文件**：`cargo fmt --all` 会把 `#[path]` 挂载的
   `tests/unit/*.rs` 报进 `--check`，但不会重写它们，必须 `rustfmt --edition 2024 <file>`。
2. **改 `autotests` 的 crate 要同时登记既有集成 target**（见上）。

## 迁移顺序

按方案与路线图：先 foundation/model/scheduler/workloads，再 runtime/frontdoor/agent，最后 GPU 私有
测试、examples 与性能场景；每批只改目录与收集，不改数值公式或产品行为。每批完成后用
`python3 tools/check/test-inventory.py --record` 下调基准并刷新本文表格。
