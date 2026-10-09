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
| core | 11 | 7 | 0 | 0 |
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
| ir | 0 | 3 | 8 | 0 |
| spi | 0 | 0 | 3 | 0 |
| **total** | **138** | **147** | **142** | **0** |

`src` 合计 **285** 个（inline 138 + 就近测试文件 147），`tests/` **142** 个。

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

| 类别 | 文件 |
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
4. **收集总数变化未经记录**：`--record` 会同时记下新的总数，避免"文件少了"被当成进展而丢掉断言。

## 迁移顺序

按方案与路线图：先 foundation/model/scheduler/workloads，再 runtime/frontdoor/agent，最后 GPU 私有
测试、examples 与性能场景；每批只改目录与收集，不改数值公式或产品行为。每批完成后用
`python3 tools/check/test-inventory.py --record` 下调基准并在本文更新表格。


已完成的第一批：`infer-spi` 的 3 个用例正文移到 `crates/foundation/spi/tests/unit/resource.rs`，由被测模块用 `#[path]` 挂载，该 crate 设 `autotests = false`；收集数量不变，用例仍编译在原模块的测试子模块里。
