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

## 执行器消费者与删除归属（54 个文件）

| `protocol` | 5 | 协议/状态场景：准入、背压、取消、完成身份、HTTP/SSE、actor。断言针对正式 runtime/state/actor 行为，用脚本化桩替代完整模型计算 |
| `service` | 25 | CLI 与服务路径：帮助、doctor、错误路径可无 GPU 测试；成功推理与网络服务移到真实设备 suite |
| `numeric` | 5 | 数值对照：单算子小型独立公式、官方 golden、真实设备验收；不允许用被测 kernel 生成 expected |
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

## benchmark 消费者已替换：声明式协议桩的形状

`benchmark` 归属（`tools/bench/cpu` 的 2 个文件）已经改完，是 67 个消费者里第一个真正落地的，
表格里因此不再有这一行。它同时回答了两轮尝试暴露的契约问题，做法可复用到 `protocol` 归属：

| 原来借用的东西 | 替换成 | 为什么是诚实的 |
|---|---|---|
| `ReferenceModel::fixture(..).ir` | 显式 `ModelIr` 描述符（2 层 attention、hidden 8、intermediate 16、vocab 32、`max_sequence` 32768，与原来同尺寸） | 只声明协议用到的图元数据，不加载权重、不执行算子 |
| `ReferenceKernels` | `DeclaredKernels`：只列出模型会 lower 到的 15 个 `Operation`，`estimated_ns`/`workspace_bytes` 是声明值，无任何执行逻辑 | 引擎要用注册表校验计划，声明支持不等于实现推理 |
| `infer_ir::testing::reference_capabilities()` | 显式 CUDA `DeviceCapabilities` 样例（`compute_dtypes`/`memory_bytes`/`unified_memory` 保持原值） | 方案要求"只验证元数据与选择规则，不宣称设备存在" |
| `completion_timing()` 上报 `CpuWall` | **不再上报计时**（trait 默认返回 `None`） | 见下 |

最后一行是上一轮那个卡点的答案。[scheduling.rs](../../crates/foundation/ir/src/scheduling.rs) 的
`ExecutionTiming::matches_backend` 只接受 `(Cuda, CudaGpu)`、`(Metal, MetalGpu)` 与被删除的
`(TestCpu, CpuWall)`，[feedback.rs](../../crates/engine/runtime/src/pipeline/scheduling/feedback.rs)
又用它和 `q.backend != self.program.backend` 一起校验成本反馈。既然这个替身**什么都不执行**，
它就不该上报任何计时：报 `CpuWall` 会因为 `TestCpu` 被删而失效，报设备计时则是伪造设备遥测。
不再上报后引擎走静态成本模型，替身也不需要时间来源——这样"非设备后端种类"这个契约问题就不必
靠新增枚举变体来解决。

**验证**：`cargo run --locked --release` 的输出与替换前**逐字段一致**（忽略 `*_ns` 墙钟），即分配
计数、stage 分解与请求口径都没变；`cargo fmt --check` 与严格 clippy（`-D warnings`）通过；
`infer-backend-reference` 从 manifest 与 `Cargo.lock` 中消失。

### 已替换：`runtime/tests/providers.rs`（`protocol` 归属第一个）

`providers.rs` 现在用 `crates/engine/runtime/tests/support/mod.rs` 里的声明式替身：显式 `ModelIr`
描述符、15 个 `Operation` 的声明、CUDA capability 样例，加上一个不执行模型的后端。替身需要额外
声明三件事才能覆盖这个场景，都是协议管道而不是推理：

- `supports_control_checkpoint() -> true`：否则 `Engine::restore` 直接拒绝；
- `capture_execution_state`/`restore_execution_state`：替身的全部"设备状态"就是它发放过哪些
  state，序列化这份清单即可（参考实现序列化的是 token 历史）；
- **每个 token 一行 hidden**：引擎只在 `rows == 计划里的 context 长度` 时才把 hidden 输出交给
  workload 的 `postprocess`，所以替身必须像参考实现那样 commit 任务的 token 并按其历史长度给出
  行数。这一条是替换过程中最容易被忽略的：行数不对时请求会"完成"但没有输出，且不报错。

`infer-runtime` 的 7 个集成 target 与单测全过，clippy 干净。

### 已替换：`runtime/tests/capacity.rs` 与 `runtime/tests/unit/observation.rs`

两个场景复用上一轮的 support 模块。`capacity.rs` 的 `Backpressure` 装饰器保留，只是内层从
`ReferenceBackend` 换成 `ProtocolBackend`，并把 `reserve_state_for`、`capture/restore_execution_state`、
`recycle_output/recycle_batch` 一并转发（引擎走的是 `reserve_state_for`，只转发 `reserve_state`
会让替身的 state 表为空）。`unit/observation.rs` 用 `#[path = "../support/mod.rs"]` 从
`tests/unit/` 指到同一份支撑模块。

这一轮补上了替身缺的两条协议行为，都是"替身不执行模型"这个前提下的正确实现：

- **`reset_state` 必须清空 token 游标**。参考实现会在 reset 时清空该 state 的 token 历史；替身
  用默认空实现时，被注入拒绝后重试的同一次 prefill 会因为游标不匹配而失败
  （`InvalidInput: incremental token cursor mismatch`），整个请求以 `Failed` 结束、`output` 为
  `None`，而测试只表现为"没有输出"。
- **不要租借输出缓冲**。替身改为每次 `submit` 复制一份声明输出：引擎可能在取回上一次输出之前就提交
  下一个 unit，用缓冲池会把这种情况判成 `readback still leased`。真正需要按分配计数的那份替身在
  `tools/bench/cpu`，它保留了自己的缓冲池实现。

### 已替换：`runtime/tests/checkpoint_owner.rs` 与 `runtime/tests/cpu_storage.rs`

两个场景的正文都不用改，只换引擎的 backend：`checkpoint_owner` 验的是异步 checkpoint owner 的
确认与恢复后重新入队，`cpu_storage` 验的是主机侧保留量、指针身份与指标导出——都不是数值对照。
因此**归属表的分类也据实修正**：`cpu_storage` 与 `checkpoint_owner` 归到 `protocol`，`runtime`
里只有 `runner.rs` 仍是 `numeric`（它逐 token 对照参考数值）。

两条替身细节：

- **logits 用声明式斜坡而不是常数**。`checkpoint_owner` 断言生成 3 个 token；常数 logits 会让
  采样一直选中同一个下标（可能正是 stop token），斜坡让"选到非 stop token"变成确定的。
  斜坡用累加构造，避免 `index as f32` 触发 `clippy::cast_precision_loss`。
- **state 表容量要按场景给**。`cpu_storage` 的候选窗口场景提交的请求数远超 4，替身容量太小会
  以 `fixed map exhausted` 失败。

### 已替换：`runtime/tests/scheduling.rs`（11 个用例）

装饰器 `TimedBackend` 保留，内层换成声明式替身，并补两项：`reserve_state_for` 转发（引擎走的是
它，只转发 `reserve_state` 会让替身 state 表为空），以及把注入计时的 `source` 从 `CpuWall` 改成
`CudaGpu` 以匹配声明的 backend 种类；`measured_cost_feedback_is_replayed_and_checkpointed_deterministically`
里那处断言也跟着改成 `CudaGpu`（该用例要验的是"测量成本被确定性重放"，不是来源名字）。

替身还差一条能力声明才让页面压力场景不再卡死：**`supports_recompute_preemption() -> true`**。
没有它，`logical_page_pressure_recomputes_and_replays_without_losing_generated_tokens` 会在
"等调度器收敛"上超时。

### 已替换：agent 与 frontdoor 的四个测试文件（本轮 5 个消费者）

| 文件 | 用例 | 说明 |
|---|---:|---|
| `agent/tests/unit/cases.rs` | 12 | 为替身实现 `AgentBackend`（`fresh`/`registry`/`inspection`/`traces`/`probes`/`profile`/`execution_stats`），断言不变 |
| `frontdoor/tests/output.rs` | 1 | 投影阻塞场景，直接构造 |
| `frontdoor/tests/isolation.rs` | 3 | `Gated` 装饰器保留闸门语义 |
| `frontdoor/tests/unit/actor.rs` | 4 | 直接构造 |
| `frontdoor/tests/unit/http.rs` | 8 | `HeldBackend` 装饰器；指标断言里的 `backend="test_cpu"` 跟随声明改成 `backend="cuda"` |

**跨 crate 复用同一份替身**：这几个 crate 用
`#[path = "<相对路径>/engine/runtime/tests/support/mod.rs"] mod support;` 引用同一份源码，而不是
各自复制一份（方案允许"多个 crate 复用的少量 fixture/脚本/断言共享测试源文件"，且不为 helper 新建
crate）。kernel 声明里的来源信息用 `env!("CARGO_PKG_NAME")` 与 `file!()`，所以别的 crate 引用它时
不会自称是 runtime。副作用有两条：

- 前端 crate 的 manifest 需要 `infer-model-recipes` 作为 dev-dependency（替身用
  `decoder::lower` 从描述符生成 dataflow）。
- frontdoor 的 `actor.rs` 与 `http.rs` 同属一个测试二进制，同一份源码被引入两次，需要一处
  `#[allow(clippy::duplicate_mod, reason = ...)]`。

**装饰器必须转发 `reserve_state_for`**：引擎通过它创建序列，只转发 `reserve_state` 会让替身的
state 表为空，几轮下来这是最常踩的坑（`capacity`、`scheduling`、`isolation`、`http` 都遇到）。

### 已替换：runtime 的两个大文件，`infer-runtime` 已零消费者

| 文件 | 用例 | 关键点 |
|---|---:|---|
| `runtime/tests/control_path.rs` | 16 | `FaultyBackend`/`DelayedBackend` 两个装饰器 |
| `runtime/tests/runner.rs` | 10 | `ReserveGate`（走 `into_threaded`，状态经资源命令到达）、`InlineReservationIntent` |

两个文件都**只用补一处**就全过：装饰器必须把 `reserve_state_for`（以及 `reset_state`/
`release_state`/`recycle_output`/`recycle_batch`）转发给替身，并把闸门/计数逻辑挂在
`reserve_state_for` 上而不是 `reserve_state`。之前失败时表现为"tick 返回 `Ok([])`、请求静默不完成"，
根因是替身的 state 表为空、`submit` 直接失败。这也是**第五次**踩同一个坑。

`runner.rs` 原分类为 `numeric`，实际验的是保留预算、取消与资源命令语义，因此按实情改回 `protocol`。
至此 `infer-runtime` 的消费者清零，manifest 里的 `infer-backend-reference` dev-dependency 已删除。

### 仍未替换：需要真实 token 内容的场景

`frontdoor/tests/unit/http_openai.rs`（15 个用例）与 `http.rs` 里那一处 host 用法需要**真实模型
生成内容**：它们加载 `examples/qwen-hybrid-tiny` 包、跑实际推理并断言流式输出里的 token 文本。替身
能提供"固定 token/logits 与可控完成时机"，但给不出与 tokenizer 一致的文本，因此这些用例要么按
"参数映射/顺序/背压/中断"与"内容断言"拆开，要么整体归到真实设备 suite，不能靠替身糊过去。

### 下一步：`control_path` 需要替身支持在途票据语义

`control_path.rs`（16 个用例）已试改：13 个可以直接跑通，3 个需要替身补在途票据的语义，本轮
**已完整回滚该文件**，但过程中确认了两条对后续批次有用的事实，并已落到 support 模块里：

- **快照必须带上每个 state 的 token 历史**（原先只存 state id）。引擎从提交过的游标继续，替身
  恢复后历史为空时，下一次 decode 会被判成 stale cursor（`incremental token cursor mismatch`）。
  现在 `capture_execution_state` 序列化 `(state, history)`。
- **注入计时的 `source` 必须与声明的 backend 种类匹配**，否则引擎直接丢弃该样本。`FaultyBackend`
  用 `CpuWall` 配 CUDA 声明就对不上，须改成 `CudaGpu`。
- **替身需要一个可声明的权重身份**：快照按 `backend.identity()` 比对"权重指纹"，所以
  `ProtocolBackend::tagged(tag, ..)` 让"不同权重"场景用不同 tag 表达，而不是真的加载权重。

剩下 3 个用例（`NeverComplete` 超时诊断、零长度计时拒绝、取消后的在途状态保留）依赖引擎对
在途票据的判定，需要在替身里显式建模"票据已提交但未完成"的状态之后才能迁移。

### 其余 `protocol` 消费者的要求

同样三件事：显式模型描述符、声明的 kernel 集合、显式 capability 样例；需要成本反馈的场景让替身
不上报计时。`protocol` 的 15 个消费者里，`runtime/tests/{control_path,scheduling}.rs` 已有包装
`ReferenceBackend` 的装饰器桩，替换时把装饰器改为包装声明式替身即可，断言不需要改写。

## 跨消费者共享还是各自持有

方案要求"公共 helper 只复用样例、脚本、票据和断言；实际 SPI adapter 放在需要它的测试 suite 或
开发 benchmark 中"，且"不为少量 helper 新增 Cargo crate"。本轮按这条执行：`DeclaredKernels`、
`protocol_model()` 与 capability 样例都放在 `tools/bench/cpu/src/engine/` 内，还没有第二个消费者
需要它们。等 `protocol` 归属的第一个消费者真正需要同一份替身时再抽共享测试源文件，而不是提前建
crate。

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

### 第二轮实测：还缺一个"非设备"后端种类，否则成本反馈无法通过校验

把 capability 换成 CUDA 样例、并补上声明的 kernel 集合之后，bench 越过了"没有可用 kernel"，
随即失败在：

```
Error { code: InvalidInput, message: "cost feedback does not match loaded program/budgets" }
```

原因在 [scheduling.rs](../../crates/foundation/ir/src/scheduling.rs) 的 `ExecutionTiming::matches_backend`：

```rust
match (backend, self.source) {
    (BackendKind::Cuda, TimingSource::CudaGpu)
    | (BackendKind::Metal, TimingSource::MetalGpu) => true,
    #[cfg(feature = "test-backends")]
    (BackendKind::TestCpu, TimingSource::CpuWall) => true,
    _ => false,
}
```

也就是说**删除 `TestCpu` 之后，没有任何 backend 种类接受 `CpuWall` 计时**。而 CPU benchmark 的
全部意义就是测量 CPU 墙钟下的协议与分配（它的完成计时必然报 `CpuWall`），`feedback.rs` 又会用
`q.backend != self.program.backend` 与 `timing.matches_backend(...)` 双重校验。结果是：一个
"不执行模型、只测主机侧协议"的后端既不能声明 CUDA/Metal（会要求设备计时，等于伪造设备遥测），
也不能在删掉 `TestCpu` 之后通过成本反馈校验。

这不是某个消费者的局部问题，而是删除方案需要一个明确决定：要么保留一个**不带 `test-backends`
feature 的非设备种类**（它不再是产品可选后端，只是协议/基准的身份），要么让 `CpuWall` 对某个
声明的非设备种类合法。方案完成条件里的"实现和 manifest 不再包含 `TestCpu`"在这一条解决之前无法
落地，因此 E1 的删除半段建议先补这个契约，再逐消费者替换。

## CI 成本

文档类改动不再触发整条矩阵：`.github/workflows/ci.yml` 对 `docs/**` 与 `**/*.md` 加了
`paths-ignore`，这类改动走 `.github/workflows/docs.yml` 只跑 `make check-tools`（本地链接、布局、
用例清单）。原因是整条矩阵包含 4 次打包与 2 个 macOS runner，改一行 markdown 也要付这份成本；
文档校验本身很便宜。改动同时把 `actionlint` 纳入本地可跑（`check-tools` 已包含）。
