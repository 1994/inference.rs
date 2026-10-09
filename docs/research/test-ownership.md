# 测试归属与执行器消费者清单

本文是 [测试组织方案](../plans/engineering/tests.md) 与
[删除测试后端方案](../plans/engineering/remove-test-backends.md) 的第一步产出：迁移前把现有用例
身份、平台/feature 门控、ignore 原因和两个 CPU 测试执行器的消费者登记下来，供逐 crate 迁移和对账。

清单由 [test-inventory.py](../../tools/check/test-inventory.py) 生成并**作为门禁执行**：基准快照存于
`tools/check/test-inventory.json`，`make check` 会拒绝下列变化。

## 现状（427 个用例，`src` 已清零）

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

起始为 288 个正文在 `src`，现已全部搬到各 crate 的 `tests/unit/`：**`src` = 0**，
`tests/` = 427。用例正文仍在被测模块的测试子模块里编译（`#[cfg(test)]` +
`#[path]` 挂载），私有访问与编译边界不变。

## 门控与 ignore

| 类别 | 数量 | 含义 |
|---|---:|---|
| `#[cfg(feature = "test-backends")]` | 3 | 只有启用测试后端才编译；删除执行器时必须一并迁移或改写 |
| `all(target_os = "linux", feature = "cuda")` | 0 | 生产设备用例 |
| `target_os = "macos"` | 1 | Metal 用例 |
| 需要 GPU/CUDA fixture 的 `#[ignore]` | 40 | 必须由具名 device suite 显式选择，不能靠"全 ignored"算通过 |

ignore 原因原文全部记录在快照的 `ignored_reasons` 里，后续删除执行器时按方案要求逐条对应，不允许
因为换目录而丢掉断言或把设备用例降级成 host 用例。

## CPU 测试执行器的消费者（85 个文件）

| 类别 | 文件数 |
|---|---:|
| manifest | 11 |
| 源码 | 53 |
| 其它（脚本/工具） | 21 |

完整列表见快照的 `executor_consumers`。门禁比较的是**依赖面**（哪个 crate 提到哪些模式），不是文件
路径或文件数：迁移会把一个消费者的正文拆到 `tests/unit/`，那不算新增依赖；只有出现新的
crate × 模式组合才会被拒绝。

这批是删除 `crates/testing/cpu/host`、`crates/testing/cpu/reference` 与产品 `test-backends`
feature 之前必须逐个替换或删除的消费者，也是 [E1](../plans/README.md) "消费者替换及测试迁移" 一批
的输入。

## 门禁拒绝的变化

1. **`src` 里的用例正文增加**：迁移方向是只减不增，基准数只能通过 `--record` 在新一轮迁移后下调。
2. **`#[path]` 挂载消失或重复**：挂载必须解析到存在且唯一的文件。
3. **测试文件没有任何模块挂载**：`tests.rs`、`*_tests.rs`、`*_check.rs` 里的用例会静默不运行。
4. **出现新的执行器依赖面**。
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

六个操作要点，后续批次（examples、性能场景）沿用：

1. **移动后要用 `rustfmt` 直接格式化新文件**（`cargo fmt --all` 只报不写）。
2. **改 `autotests` 的 crate 要同时登记既有集成 target**，否则用例被静默移出收集。
3. **抽取内联块不能只数花括号**（测试数据里的 JSON 会打乱深度）。
4. **重写挂载要按解析结果比对 `#[path]` 的值**，不能只比文件名。
5. **`#[path]` 会把文件位置与模块树解耦**，找不到 owner 时要回退到全树扫描解析结果。
6. **内联块的 `cfg` 可能是 `all(test, ...)`**，抽取时要保留原属性并把 `#[path]` 插在 `mod` 前。

## 待处理的既有 lint：`check-cuda` 的 clippy

`make check-cuda` 跑 `cargo clippy -p infer-backend-cuda --features cuda --all-targets -- -D warnings`，
在迁移前后都报同样 11 条（用 `git worktree` 在迁移前 HEAD 实测 12 条错误、EXIT=101），全部在
`resident/recurrent_prefill` 的测试正文里：5 条 `too_many_lines`、2 条 `float_cmp`、
2 条 `redundant_clone`、2 条 `cast_precision_loss`。`policy.py` 禁止豁免 `too_many_lines`，所以那
5 条只能拆函数；其余按其建议改。这是独立一批，不改任何断言。

## 下一步

`src` 清零后，E1 剩下的是方案里的后半段：examples 与性能场景归位、规则检查（已由本文的门禁覆盖
大部分）、然后按删除方案逐个替换/删除 85 个执行器消费者，最后删掉两个 CPU 测试执行器
与 `test-backends` feature。
