# 服务内 MTP 接线设计

诊断路径（`cuda-model-smoke --mtp`）已经跑通 draft propose → target verify → 接受前缀提交，并在真实模型上通过短验证；生产 Engine / CLI / HTTP 路径完全没有 MTP。本文给出把同一套算法接到生产路径的方案、契约改动与验收口径。实现进度见[实现状态](status.md)，性能背景见[性能路线](performance-plan.md)。

## 0. 进度

| 阶段 | 状态 |
|---|---|
| P1 draft 加载：`LoadOptions.mtp_depth`、draft 图 + fusion 权重 + 共享 embedding/lm_head、`LoadedModel::draft()`、预算计入、CLI `--num-speculative-tokens` | **已完成**，真实模型 GPU 验证：`cuda-provider-check --num-speculative-tokens 2` 能捕获 draft 程序且输出与关闭时逐位一致（`391`） |
| P2a 契约：`ExecutionTask.sampling`、`ModelOutput.tokens`、`ProcessedOutput.tokens`、`complete_generation` 多 token 提交 | **已完成**，workspace 测试与 golden 执行通过 |
| P2b executor 推测链：`Sequence` 持有 draft、propose → target 批量 verify → 接受前缀提交、draft 回退重放 | **已完成**：`--num-speculative-tokens 2` 下服务真实模型逐 token 与 `--num-speculative-tokens 0` 完全一致（`391` / 4 token / 默认采样请求同样一致），7 项服务检查全过；探针确认每次 decode 都走了 3-lane 批量 verify |
| P3 批量 verify（`step_batch` + `commit_batch`，`verification_width = FUSED_VERIFY_LANES`） | **已完成**（`--num-speculative-tokens > 0` 自动启用 3 lane） |
| P3b draft 批量预热（`build32`/`capture32` 支持 fusion + per-lane hidden） | 待实现，**净加速的前提** |

> 顺序 verify 的正确性成立但**没有净加速**：每个被接受的候选都需要一次 target 单步才能验证下一个候选，步数与不推测时相同，只是多了 draft 开销。这与[性能路线](performance-plan.md)记录的“当前 MTP 仍无净加速”一致。要真正加速必须走 `step_batch`：一次 target 前向同时计算输入 token 与全部候选，再按 `commit_batch` 提交接受前缀。

## 1. 现状契约

| 层 | 事实 |
|---|---|
| 调度 | 一次 decode step = 1 个输入 token；`ready.rs` 里 decode 的 `remaining` 恒为 1 |
| 输入 | `ExecutionInput::Decode { position, token }`，且必须满足 `position == 后端已提交 token 数`（`tokens.rs` 的 `delta`/`commit`） |
| 输出 | 输出 stage 用 `infer_workloads::sample_with_history` 从 logits 采 1 个 token，`ModelOutput` 只表达 `logits` + `hidden` |
| 状态 | `executor::Sequence` 只持有一个 target `DeviceProgram`，没有 draft、checkpoint、回退 |
| 采样历史 | `SamplingHistory { prompt, generated }`：重复惩罚跨 prompt+generated，存在惩罚只看 generated |

第一个生成 token 由 **prefill 的 logits** 在输出 stage 采样，所以第一次 decode 的 `position` 恰好等于 prompt 长度。

## 2. 决策：采样下沉 executor，后端返回已决定 token

- 方案 A（采样留在引擎）：draft 逐个候选都必须回引擎采样，一次 target step 需要 `depth` 次引擎↔后端往返，且批量 verify 需要先集齐候选。不可行。
- **方案 B（采用）**：后端在 executor 内完成 propose/verify/采样，把「本次 step 新决定的 token 列表」返回引擎；引擎只负责追加、长度/EOS 判定与统计。采样数学全部复用 `infer_workloads`（`probabilities`、`draw_distribution`、`sampling_uniform`、`verify_draft`），不重写。

方案 B 也为后续「设备侧采样」（P3）留了同一个切入点：把 executor 内的 CPU 采样换成设备 argmax，其余不动。

## 3. 契约改动

1. `infer_ir::ExecutionTask` 增加 `sampling: Option<Sampling>`：调度层仅在 `ExecutionRole::Decode` 填充；不支持推测的后端忽略它。
2. `infer_ir::ModelOutput` 增加 `tokens: Vec<u32>`（`#[serde(default)]`）：后端**已经接受**的 token 前缀。后端同时返回「这些 token 之后那个位置」的 logits，引擎把它们补进采样历史后自己采出下一个 token。这条约定保证 `ExecutionInput::Decode.position == 后端已消费 token 数` 的不变式继续成立。
3. `stages/output.rs`：`ProcessedOutput.tokens` 取代单个 `token`；`OutputJob::process` 先取 `output.tokens`、把它们并进 `SamplingHistory.generated`，再采样一个 token 追加到列表尾部。`OutputShape::validate` 不变（logits 仍是完整词表长度）。
4. `pipeline/completion`：`complete_generation` 接受 token 列表，逐个追加、逐个判定 EOS 与 `max_new_tokens`；终态之后的尾部直接丢弃。
5. `executor::Sequence`：增加 draft `DeviceProgram`、`prompt_len`（第一次 decode 时记录，用于切分 `SamplingHistory`）与 draft 位置簿记。

**范围限制**：推测只在 `temperature == 0`（贪心）时启用。非贪心时被拒绝的候选必须从**残差分布**采样，而残差采样只发生在 executor 内部；让引擎从原始 logits 采样会得到错误的分布。贪心时残差就是 target 的 argmax，与引擎自己的采样结果一致，因此贪心是唯一能保持「逐 token 与不推测完全一致」的路径。

## 4. CUDA executor 推测循环

对一次 `Decode` 输入（`token` 已在历史中，`position == history.len()`）：

1. target 单步：`step_readout(token, position, position, None, logits=true, hidden=true)` 得到 `L(position+1)` 与上一 token 的 hidden。
2. `task.tokens.commit(&mut history)` 提交输入 token；`prompt_len` 在第一次 decode 时记为 `position`，此后 `SamplingHistory { prompt: &history[..prompt_len], generated: &history[prompt_len..] }`。
3. 贪心采出 `t1`（`probabilities` + `draw_distribution`，`sampling_uniform(seed, 1, index)`）；EOS 即结束本轮。
4. draft checkpoint = `draft.position()`；propose 第 i 个候选时在 `kv_position = index - 1` 处执行 draft 单步（MTP 层 `kv_offset = 1`，draft 图必须 attention-only），用 `sampling_uniform(seed, 2, index)` 采出候选。
5. verify：target 在候选位置单步得到分布，`verify_draft` 判定接受/替换；被替换的 token 结束本轮。顺序 verify 用 `step_readout` 逐个候选；批量 verify 用 `step_batch`/`commit_batch`。
6. `draft.rewind_attention(checkpoint)` 后把已接受 token 重放进 draft，保持 draft KV 与 target 对齐。
7. 结束时 target 已消费最后一个决定 token，返回 `ModelOutput { tokens: 决定的 token, logits: 该位置之后的分布, hidden: vec![] }`；引擎据此再采一个 token，下一次 decode 的 `position` 与后端游标继续相等。

容量、EOS、`max_new_tokens` 与「greedy 等价」是这一步的四条硬约束。

### 4.1 已确认的实现约束（下一步按这些写）

**a. draft 必须先用 prompt 预热。** draft 的 `kv_position` 永远是 target 下标减一（`kv_offset = 1`），而 `step_readout` 要求 `kv_position == next_position`。若第一次 decode 就想在 `kv_position = P` 处喂 `d1`，draft 的 `next_position` 必须已经是 `P`，因此**每个 prompt token 都要预先过一遍 draft**。诊断路径正是这么做的（`model_smoke/prefill.rs:37-49`，逐 token `draft.step`）。
代价是 prompt 长度个 draft 单步；draft 目前 `prefill_width = 1`、`batch_width = 0`，只能单步。**优化方向**：给 draft 配 `prefill_width = PREFILL_LANES (32)` 的 prefill 图，把预热批量化——这需要在 `loaded.draft()` 里按 prefill 宽度构造 draft 程序并验证数值不变。

**b. `step_batch` 语义**（`resident/program.rs:301-331`）：一次前向跑 `tokens`（≤ `batch_width`）个 lane，内部**已经** `commit_batch(tokens.len())` 并把 `next_position` 推 `width = batch_width`（不是 `tokens.len()`），同时记录 `last_batch = (position, tokens.len())`。因此接受前缀比 lane 数少时，必须再调 `commit_batch(accepted)` 把 KV 回退/重提交到真正的接受长度，并同步 `next_position`（`TargetSteps::finish` 就是这一步）。

**c. hidden 供给**：decode 用 `step_readout(..., read_hidden = true)` 拿 target hidden；prefill 的 `BatchOutput` 每 lane 都带 hidden。draft 的 fusion 需要「**被喂 token 的前一个 token 的 target hidden**」（与 `propose` 里 `previous = history.last()`、`hidden = model.hidden` 一致），重放时按同一规则取。

**d. 容量守卫**：`step_batch` 要求 `position + batch_width <= capacity`，推测前必须检查，不足时回落到普通单步路径。

**e. 贪心下的 verify 可以不做分布采样**：`verify_draft` 在 `temperature == 0` 时退化为「候选 == target 贪心 token」——`p_target` 是 one-hot，接受概率 `min(1, p_target[u]/p_draft[u])` 只有 `u == argmax` 时为 1。所以生产实现只需比较 draft 候选与 target 的贪心 token（两者都用 `infer_workloads::sample_with_history` 在同样的 `SamplingHistory` 下算出），被拒绝时用 target 的贪心 token 替换。省掉每个候选两份全长词表分布（248320 × f64 ≈ 2 MB/份）。这也保证了「与不推测的贪心逐 token 一致」。

**f. 返回值**：无论接受/替换，结束时 target 必须**已经消费最后一个决定 token**，返回该位置之后的 logits；引擎据此再采一个 token（见 3.2 的契约）。拒绝分支里替换 token 不在 batch 里，需要补一次单步把它消费掉。

## 5. 加载与预算

- `LoadOptions.mtp_depth`（0 = 关闭）决定是否加载 draft 图与 fusion 权重；`LoadedModel::draft(capacity)` 构造 draft `DeviceProgram`。
- draft 与 target 共享 `embed_tokens` 与 `lm_head` 的**设备张量**（`Arc`/`Projection` clone），不重复占显存。
- draft KV 固定 F32、attention-only；`sequence_budget` 计入 draft 的 KV、激活、projection workspace 与 readback staging。

## 6. 验收口径

1. **greedy 等价**：`temperature = 0` 时 `--num-speculative-tokens 0` 与 `--num-speculative-tokens 2` 必须给出**完全相同的 token 序列**（推测解码只有在接受时才跳过 target 步骤，被替换时用 target 的 argmax，因此贪心序列不变）。
2. 数值回归：`cuda-resident-check` 63 项与 `cuda-provider-check`（F32/FP8）保持通过。
3. 静态门禁：`fmt`、`policy.py`、四种 feature 组合的 clippy 与 workspace 测试。
4. 真实模型：`infer run`/`serve` 在 `--num-speculative-tokens 2` 下端到端出 token，并记录 proposals/accepted 统计。

## 7. 实测约束与优化优先级（RTX 5090 / Qwen3.8-27B-NVFP4）

`cuda-provider-check --num-speculative-tokens 2` 实测（artifacts/mtp-prime.json）：

| 量 | 实测 |
|---|---:|
| draft 预热 | **650 µs / prompt token**（21 token 的 prompt 预热 20 步共 13.0 ms） |
| 生产 resident decode 单步 | ~15 ms（`cuda-pinned-mtp2.json`：6 个 target 步 0.089 s） |
| 输出与关闭 MTP 时 | 逐位一致（`391`），draft 程序捕获成功 |

**决定性约束：fusion 程序不能建批量图。** `resident/batch.rs:75-81` 的 `BatchBuilder::build` 对 `weights.fusion.is_some()` 直接报错，`capture32` 也把 `fusion: &mut None` 写死——批量路径没有 per-lane external hidden 的通道。因此：

- **draft 永远只能单步**（`prefill_width = 1`、`batch_width = 0`），预热就是 O(prompt) × 650 µs：1000 token ≈ 0.65 s，8192 token ≈ 5.3 s。这是 TTFT 级别的代价，不是可以忽略的噪声。
- **target 侧批量 verify 仍然可行**（target 无 fusion），`step_batch` 一次前向覆盖 3 个位置，这是唯一的净加速来源。

由此优化优先级：

1. **P3 target 批量 verify**（净加速本体）：把每 3 个位置一次 target 前向接进 `Sequence`，配 `commit_batch` 接受前缀。按上表估算，decode 段每个 token 省约 2/3 次 target 前向（约 10 ms/token），几百 token 的生成即可超过一次 prefill 级别的预热点。
2. **draft 批量预热**（TTFT 优化）：让 `build32`/`capture32` 支持 fusion + per-lane external hidden，把预热从 650 µs/token 降到 ~20 µs/token 量级。该项不改数值语义，但需要改 capture 路径和一个逐 lane hidden 的输入通道，属于 kernel/capture 层的独立改动。
3. **共享 KV 方案**（可选，架构级）：若 draft 能复用 target 的 KV 而不是自己再算一遍，预热成本可以直接消失；但这会改变 draft 的注意力语义，需要重新做数值验证。

**结论**：MTP 在当前架构下只对「生成长度远大于预热点摊销」的请求有净收益。P3a（target 批量 verify）已经接通，因此**现在打开 `--num-speculative-tokens` 是功能正确但未必更快**：实测 21-token prompt / 4-token 生成时，`--num-speculative-tokens 2` 每个请求 131 ms，`--num-speculative-tokens 0` 106 ms——draft 预热（20 × 650 µs ≈ 13 ms）与额外 draft 前向超过了省下的 target 步。要让 MTP 真正变快，必须做 P3b（批量预热）并面向长生成场景。

### 7.1 实测证据（RTX 5090 / Qwen3.8-27B-NVFP4，`tools/bench/check-cuda-service.py`）

| 运行 | 输出 | 7 项服务检查 | 顺序请求 wall_ms |
|---|---|---|---|
| `--num-speculative-tokens 0` | `391`，4 token | 全过 | 1369.8 / 106.8 / 106.9 / 106.4 |
| `--num-speculative-tokens 2` | `391`，4 token | 全过 | 1891.3 / 131.6 / 133.8 / 135.3 |

两次运行的响应体（除时间戳、临时端口、命令行）逐字段相同；探针显示每次 decode 都执行了 `proposals_max=2` 的 3-lane 批量 verify，说明一致性来自真实的推测链而不是回退。首个数字是含 JIT/图捕获的冷启动，不参与比较。

### 7.2 端到端 A/B（`tools/bench/mtp-ab.py`，137-token prompt，temperature 0）

| 场景 | `--num-speculative-tokens 0` | `--num-speculative-tokens 2` |
|---|---|---|
| 1 请求 / 128 token（实际 72 token 后 EOS） | TTFT 1.48–1.56 s，TPOT 17.3–17.5 ms，**57.1–57.8 decode tok/s**，76 step | TTFT 2.03 s，TPOT 18.9–19.4 ms，**51.6–52.8 tok/s**，24 step（**3.00 token/step**） |
| 1 请求 / 64 token | TTFT 1476 ms，TPOT 17.38 ms，57.53 tok/s，68 step | TTFT 2078 ms，TPOT 18.17 ms，55.03 tok/s，22 step |
| 2 请求 / 64 token | TTFT 1851 ms，TPOT 35.37 ms，56.5 tok/s，68 step | TTFT 2539 ms，TPOT 36.46 ms，54.9 tok/s，22 step |
| 4 请求 | 准入失败：每序列 2.75 GB 状态、总预算约 8 GB | 准入失败：每序列 **3.44 GB**（+25%） |

所有对比里 **greedy 序列完全一致**（逐 token 相同），推测接受率 **100%**（3.00 token/step，76 → 24 步）。

**为什么还是更慢**：当前实现每个 decode 步要做**两次 target 全量前向**——Phase A 消费输入 token，随后 3-lane 批量 verify。3 个 token 花 2 次权重读取（~2×17 ms）再加 draft 的 3 次 propose + 3 次 replay 单步，收益被吃光。而 batch=1 时 57 tok/s 已经接近带宽上限的 61%（roofline 93.8 tok/s，见[性能路线](performance-plan.md)），所以并发也提不上吞吐。

**下一个真正的优化**：把输入 token 折进 verify 批量（批量直接跑 `[input, u1, u2]`，由 draft 提议第一个 token，删掉 Phase A），接受时不再 rewind/replay 整段。这样 3 个 token 只需 **1 次 target 前向**。

**已实现并实测**（`artifacts/mtp-ab-folded.json`）：折叠后 76 step → 30 step，TPOT 18.9 → 18.1 ms，decode 51.6–52.8 → 54.2–55.2 tok/s，greedy 序列仍逐 token 一致。但离 depth 0 的 57.4 tok/s 仍差 4%，原因是逐步计时显示瓶颈已经换人：

| 阶段（每步均值，22 步） | 耗时 |
|---|---|
| `step_batch`（3-lane verify） | **40.9 ms** |
| draft replay（1 步） | 4.6 ms |
| draft propose（2 步） | 3.7 ms |
| commit + 批量 hidden 收集 | 0.9 ms |
| 2 次 CPU 贪心采样（各 248320 logits） | 0.8 ms |

**3-lane 批量 verify 40.9 ms 是单步 17.4 ms 的 2.3 倍**——在带宽受限的 decode 里，批量本应只读一次权重（~10.6 ms 理论值）。所以现在真正的杠杆是让批量 verify 接近单步成本（权重读取跨 lane 共享 / kernel 效率），而不是 CPU 采样（0.8 ms）或 draft（3.7 ms）。在这一项解决之前，`--num-speculative-tokens` 的正确收益上限约为 3 token / 41 ms ≈ 73 tok/s。

**verify GEMV tile 实验**：`resident/batch_projection.rs` 的 `VERIFY_OUTPUT_TILE = [4, 8]`（BN=8、BK=256）不是随手写的。把 BN 提到 32 后 decode 直接掉到 **26 tok/s**（TPOT 38.5 ms，`artifacts/mtp-ab-bn32.json`），因为 `dense`/`fp8`/`nvfp4` 用的是 `sum: Tile<f32, [4, BN, BK]>` 三维累加器、k 循环结束后才 `reduce_sum`，`[4,8,256]` 已在 32 KB/block 量级，放大即溢出。**结构性修法的尝试与结论（已实测）**：把 k 循环内做成 `[4, BN]` 二维累加器在 cuTile 里**无法表达**——
循环内 `reduce_sum(product, 2)` 返回的是**具体形状**的 tile（`Tile<f32, [4, 8]>`），与 const-generic 的
累加器（`Tile<f32, [4, BN]>`）无法统一：加类型标注编译失败，改用 `.reshape(shape![4, BN])` 则在 JIT 时报
`binary Add requires operands of the same type`。已回退到三维累加器并复测通过（depth 0 62.8 tok/s、
depth 2 61.9 tok/s，无错误）。**另一条路也被格式数学堵住**：NVFP4 的 `BP = BK/2` 且 `BS = 16` 写死在 kernel 里
（`broadcast(shape![BN, BS, 16])`），所以不能靠缩小 BK 来减累加器。下一步要提高 3-lane verify 的效率，得换
思路（例如让批量 GEMV 直接吃三个独立 hidden、省掉每节点 2 次 pack/unpack 启动），而不是改累加器形状。这会改变归约顺序，需要用 `cuda-resident-check` 的容差校验兜住，而不是要求逐位一致。

### 7.2b 批量 draft 预热（fusion × 批量捕获）

`BatchBuilder::build` 原先直接拒绝任何 fusion 程序，导致 draft 只能单步运行、预热 O(prompt)。现已支持：

- `allocate_lane_fusion` 为每个 lane 分配 external hidden 与 fusion workspace，由 `DeviceProgram` 持有；
- `capture`（2–9 lane）与 `capture32`（32 lane）的 Row 分支都按 lane 传入各自的 hidden，融合段
  `embedding → fusion_norm → mtp.fc` 因此读到本 lane 的 hidden；
- `BatchGraph` 记录 state 相对 RoPE 的滞后（draft = `-MTP_KV_OFFSET`），`prime_batch` 一次上传
  各 lane hidden 并跑 32-lane prompt 图；`prime_draft` 把 prompt 按 32 切块（不连续时逐 token 回退）。

实测（137-token prompt，1 请求，64 新 token）：depth 2 TTFT **2051 → 2020 ms**，TPOT 17.78 → 17.84 ms，
**greedy 逐 token 等价仍为 True**。

**下一个杠杆**：剩余 MTP TTFT 开销（depth 0 1614 ms vs depth 2 2020 ms ≈ 406 ms）主要落在每 lane 的
staging —— 每个 chunk 要 32 次 `upload` + 32 次 `graph.update`。改成一个 "scatter lanes" kernel
（一次 upload + 一次 launch 把 `[lanes, hidden]` 摊到各 lane 缓冲）即可把这块压到接近 0。

### 7.2c 并发被"每序列一套图"锁死（实测）

同一份 3 请求配置在真实模型上的对照（`--gpu-memory-utilization 0.95`）：

| `--max-num-batched-tokens` | 准入 | TTFT |
|---|---|---|
| 32（默认，32-lane prompt 图） | **拒绝**：`StateBytes required=2710378496, available=2358104064` | — |
| 1（不捕获 prompt 图） | 3 条请求全部准入并完成 | **6777 ms**（prefill 退化为逐 token） |

结论：32-lane prompt 图的**每序列工作内存约 2 GB**，它既是 prefill 速度的来源，也是并发被锁在 1 的原因
（`state.rs::reserve` 每序列 `loaded.sequence(capacity)` 新建 `DeviceProgram`；`budget.rs` 按
`lanes = 1 + verify(3) + prompt(32)` 计费）。禁用 prompt 图能放 3 条并发，但 TTFT 6.8 s，不可取。

**因此下一步唯一正确的方向是把捕获好的图与工作内存上移到模型级共享**（KV 仍每序列）。障碍是捕获时
kernel 参数里烧进了每序列的 state 指针，所以需要：要么按序列重绑 state 实参（`GraphUpdate` 机制已有
先例），要么让 state 常驻一份 per-model 池、序列只持块表。

### 7.2d 每序列内存分解与可执行的收敛路径

`sequence_budget` 的记账口径（[budget.rs](../../crates/backend/cuda/src/loading/budget.rs)）：

```
lanes = 1(flat) + verify(3) + prompt(32) = 36
bytes = graph_headroom + program(graph, capacity, lanes) + program(draft, 1) + capacity*F32
```

实测每序列 2.71 GB ÷ 36 lane ≈ **75 MB/lane**，因此分解约为：

| 组成 | 占比 | 说明 |
|---|---|---|
| prompt（32-lane 批图工作集） | **~2.2 GB（81%）** | 只在 prefill 期间使用，decode 期间纯占位 |
| verify（3-lane） | ~0.2 GB | decode 推测用 |
| flat + KV + headroom | ~0.3 GB | 主路径 |

**可执行路径**：把 prompt 图改成**按需构建、prefill 完成后释放**（`DeviceProgram::build_prompt_batch`
惰性重建；`prefill_width()` 已有回落到 `batch_width()` 的行为）。这样：

- prefill 期间：仍按现状占 ~2.2 GB（prefill 是短阶段，并发受限可接受）
- decode 期间：每序列降到 **~0.5 GB** → 准入并发从 **1 提到 ~4-5**

风险点只有一个：`ResourceCommand::Prefix` 之类的重前缀路径会让已释放 prompt 图的 state 再次 prefill，
必须走"惰性重建"而不是报错。

**硬件边界（必须同时说明）**：本机 32.6 GB 显存装下 22 GB 权重后只剩 ~2.4 GB 可用。即使每序列降到
0.5 GB，并发上限也只有 ~5；要真正对标 vLLM 的几十并发，需要 fp8 KV / 张量并行 / 更小的权重占用——
这与"每序列 2.71 GB 的开销"是两个独立的限制，不能混为一谈。

### 7.2e TTFT 的真实构成：图捕获，不是 prefill（事件流实测）

同一次 137-token prompt 请求的事件流（`tools/bench` 的 events.jsonl）：

| 事件 | 时间 |
|---|---|
| `StateReserved` | t = 0 |
| `Deferred` / `ResourceAcknowledged` | **t = 1168 ms** |
| 首个 `TokenProduced` | t = 1462 ms |

即 **TTFT 的 1168/1462 = 80% 花在"状态预留"**，也就是每序列的 `DeviceProgram` 图捕获 + 分配；真正的
prefill 只有 ~294 ms。池（`StatePool`）的键已正确归一化到 `next_power_of_two(capacity)` + readout，所以
**同 bucket 的后续请求是热的**——但每个新 bucket、以及并发准入的第一条，都要在请求路径上付这 1.17 s。

这是与 vLLM 的又一个设计差异：vLLM 在 **init 阶段**捕获 CUDA graph（启动慢、请求快），我们在请求路径上捕获。

**两个可落地的修法**：

1. **load 时预热池**（对应 vLLM 的 init-time capture）：按 bucket 从小到大 reserve+release，直到池预算用尽——
   把 1.17 s 从请求路径挪到启动路径。唯一要处理的是"预热哪些 bucket"（后台按需预热最理想）。
2. **减少捕获本身**：prompt（32-lane）图当前是 eager 捕获的最宽图，改惰性构建可省掉一大块；前置条件是
   KV states 归 `Sequence`/`DeviceProgram` 所有 + 保留 weights/IR（`DeviceProgram<'a>`）。

### 7.2f 修正：稳态 TTFT 是 ~107 ms，1.37 s 只是"每个 bucket 的第一次"

Round 8 用事件流得出"TTFT 的 80% 是图捕获"，但那是**冷启动**读数。服务验收产物里的 4 次**串行**请求
（`artifacts/mtp-svc-d0.json` 的 `sequential_request_wall_ms`）给出了完整对照：

| 请求 | depth 0 | depth 2 |
|---|---|---|
| 第 1 次（冷：付图捕获） | **1369.8 ms** | **1891.3 ms** |
| 第 2/3/4 次（热：命中 `StatePool`） | 106.8 / 106.9 / 106.4 ms | 131.6 / 133.8 / 135.3 ms |

结论修正：

- **稳态 TTFT ≈ 107 ms**（MTP 开启 +25 ms，来自 draft 预热），已经很接近可用水平；1.37 s 是每个
  `next_power_of_two(capacity)` bucket 的**一次性**开销，池会把它缓存住（键含 pow2(capacity)+readout）。
- 因此"load 时预热池"的价值不是提升稳态，而是**别让用户的第一条请求承担这 1.37 s**（vLLM 在 init 做
  同样的捕获）。代价是启动时间 +1.37 s/bucket，收益是首请求 TTFT −1.27 s——这是运维取舍，值得做成
  可配（并默认按 bucket 数上限约束）。
- Round 8 提出的"惰性 prompt 图以减少捕获"依然成立（减少的是那一次冷启动与每序列内存），优先级低于
  "把捕获挪到启动路径"。

### 7.2g 实测：naive 池预热是净伤害

按 7.2f 的取舍实现了一版"load 时预热 N 个 capacity bucket"（`warm_pool`，用 `--warmup-buckets 2`
驱动），服务验收实测（4 次串行请求）：

| | 第 1 次 | 第 2/3/4 次 |
|---|---|---|
| 不预热（`mtp-svc-d0.json`） | 1370 ms | 106 / 107 / 106 ms |
| 预热 bucket 32+64 | **240 ms** | **390 / 390 / 391 ms** |

首请求确实不再付冷启动（1370 → 240 ms），但**稳态从 107 ms 恶化到 390 ms**：池的容量只够放少量
program，被预热进去的 32/64 条目挤掉了流量真正复用的那个 bucket，于是每个请求都要重新捕获
（390 ms ≈ 缓存已热时的捕获成本，1370 ms 则含首次 cuTile JIT）。

结论：**`StatePool` 的价值恰恰来自"只缓存流量真正用的 bucket"**；在池容量有限的前提下预热多个 bucket
是自伤。已回退该实现（不留开关，避免运维踩坑）。下一步若要真正解决冷启动，必须让池容量与预热
bucket 数联动（按 bucket 数分配池预算），而不是无条件多缓几个。

### 7.2h 回退：decode 期释放 prompt 图是回归（服务验收抓到）

Round 6 让 `Sequence` 进入 decode 时释放 32-lane prompt 图，动机是省内存。**服务验收（4 次串行请求）
暴露了它没被测到的代价**：

| depth 0 | 第 1 次 | 第 2/3/4 次 |
|---|---|---|
| 有释放（Round 6–10） | 1356 ms | **388 / 391 / 386 ms** |
| 回退后（本轮） | 1342 ms | **110 / 109 / 110 ms** |

原因：释放后序列回到 `StatePool` 时**不再带 prompt 图**，复用它的下一个请求 `prefill_width()` 回落到
3（verify 宽度），prefill 变成每 3 个 token 一次。TTFT 从 107 恶化到 388 ms —— 而稳态请求才是服务
的绝大多数流量。Round 6 的验证只测了"新序列一次请求"，没测池复用路径，因此漏掉了。

**回退后并发不受影响**：3 请求仍然全部准入（TTFT 2178 ms 并发完成，无 `Capacity`），因为并发是
Round 7 的**精确 arena 记账**带来的，与释放无关。

回退后的服务验收基线（`artifacts/svc-r11b-d0.json` / `svc-r12-d2.json`，均 `passed=true` 6 项检查）：

| | 第 1 次（冷 bucket） | 第 2/3/4 次（热） |
|---|---|---|
| depth 0 | 1342 ms | 110 / 109 / 110 ms |
| depth 2（MTP） | 1962 ms | 125 / 125 / 125 ms |

即 MTP 的稳态 TTFT 开销只有 **+15 ms**（此前被释放 bug 放大到 +233 ms）。

教训：任何影响 `Sequence` 生命周期/池复用的改动，必须用**服务验收的串行请求序列**验证，单次 `run`
或新进程基准都会漏掉池路径。

### 7.2k 回归防线：池化状态必须保持 prefill 宽度（Round 26）

Round 6 的回归（decode 期释放 prompt 图 → 池化状态失去 32-wide prefill 路径 → 热请求 108→388 ms）当时
**没有任何自动检查能发现**：单次 `run` 与新进程基准都走不到池复用路径，而服务验收只比对响应、不看耗时。

现已把它变成断言：`CudaBackend::prefill_width_for(state)`（[state.rs](../../crates/backend/cuda/src/executor/state.rs)）
暴露某状态所用程序的 `prefill_width()`；`cuda-provider-check` 的池探针在**释放前**记下新程序的宽度，
在**池复用后**断言两者一致：

```rust
let fresh_width = backend.prefill_width_for(states[0])?;
...
if backend.prefill_width_for(id)? != fresh_width {
    return Err(Error::invariant("pooled state lost its wide prefill path"));
}
```

用"新程序宽度"作参照而不是硬编码 32，因此对任何 `--max-num-batched-tokens` 配置都成立。实测 GPU 通过
（`cuda-provider-check` passed，输出 `391` 不变）。

### 7.2l 状态池的可观测入口（Round 29 勘察，已存在，无需新增）

TTFT 的冷/热差异全部由 `StatePool` 决定，因此它的计数是最该被观测的指标。勘察结论：**已经暴露**，不需要
新增 SPI 方法——

| 入口 | 内容 |
|---|---|
| `crates/service/cli/src/backend/selected.rs:146` | catalog JSON 里的 `state_pool: PoolInspection`（`active_sequences` / `cached_sequences` / `cached_admission_bytes` / `sequence_allocations` / `sequence_reuses` / `sequence_evictions`） |
| `agent` 的 `state` 命令（`commands/state.rs:59`） | 页池 + 抢占 + `execution_stats`（与 CUDA 的 `StatePool` 是两层，别混用） |
| `cuda-provider-check` 的池探针 | 断言级检查（Round 26 新增了"池化后 prefill 宽度不变"） |

判断"是否在反复重捕获"的判据：`sequence_allocations` 随请求数线性增长而 `sequence_reuses` 不涨 ⇒ 池键不命中
（容量 bucket/readout 不匹配），这正是 Round 6/11 那类回归的**早期信号**。

### 7.2m **回归（已修）**：MTP 推测路径静默失效（Round 36 发现 → Round 37 修复）

用同一工具、同一 prompt 对比两档的**步数**（`mtp-ab.py` 的 `steps` 来自 events）：

| 配置 | max_new_tokens | depth 0 steps | depth 2 steps | depth 2 TPOT |
|---|---|---|---|---|
| Round 16（当时） | 64 | 68 | **27**（2.4 token/步） | 18.23 ms |
| Round 36（现在） | 64 | 68 | **68**（1 token/步） | 17.40 ms |
| Round 36（现在） | 16 | 20 | **20**（1 token/步） | 17.40 ms |

判据：depth 2 的 **TPOT 与 depth 0 几乎相同**（17.40 vs 17.48）。若推测在跑，每步是 3-lane verify
（约 2.3× 单步成本），即便接受率为 0，**TPOT 也该是 ~40 ms**；实测 ~17.4 ms ⇒ **verify 批量根本没执行**。
同时 depth 2 仍比 depth 0 多付 **+566 ms TTFT**（2010 vs 1444）⇒ draft 仍在加载与 prime，只是不再投机。

即 **MTP 现状是纯开销**（早前"break-even"的结论应修正为"未生效"；Round 18 之后两档数值一直近似相等，
与此一致）。

**当时已排除**（Round 36 逐一核对，均正确，说明问题在包装层而非这些环节）：
- `BackendProvider::speculation_capability` 的 CUDA 覆盖在 trait impl 内且返回 `self.capabilities.speculation`
- `CudaBackend::new` 里 `capabilities.speculation = { draft_depth: loaded.mtp_depth(), greedy_only: true }`
- `pipeline/dispatch.rs:80` 仍按 `speculation_capability().draft_depth > 0` 决定是否传 `sampling`

**根因（Round 37 定位并修复）**：`SelectedBackend`——CLI 里包裹具体后端的**枚举包装层**——没有委托
`BackendProvider::speculation_capability`。该方法是后加的、**带默认实现**（返回 `draft_depth: 0`），于是
包装层走了默认值 ⇒ 引擎认为后端不支持推测 ⇒ `pipeline/dispatch.rs` 从不传 `sampling` ⇒ CUDA executor 的
`speculate()` 第一个守卫（`task.sampling.as_ref() else return Ok(None)`）直接跳过。**全程零报错**，
greedy 等价照旧成立，只有步数/耗时暴露问题。

修复：在 `SelectedBackend` 里显式委托（`forward!(self, speculation_capability)`），并顺手审计了**所有**带默认
实现的 trait 方法（35 个），补齐了另外 4 个同样缺失的委托：`requires_async_checkpoint`、
`pending_resource_releases`、`resource_epoch`、`set_waker`（其中 `set_waker` 正是异步通路所需）。
另外 3 个**有意不委托**并写明原因：`submit_shared`/`submit_shared_borrowed` 的默认实现会走包装层自己的
`submit`（已经正确包 ticket），`launch_accepted` 需要按变体转换 ticket 类型。

**修复后实测**：

| | steps（64 token） | TPOT | greedy 等价 |
|---|---|---|---|
| depth 0 | 68 | 17.65 ms | — |
| depth 2 | **27**（2.4 token/步） | 17.75 ms | **64/64 token 完全一致** |

服务验收（depth 2）`passed=true`、6 项检查、热请求墙钟 110–112 ms。即在恢复推测的同时没有牺牲稳态延迟。

> 教训（第三次同类）：**"不报错的静默降级"最危险**。它会同时骗过数值检查（greedy 仍等价 ✓）、服务验收
> （响应仍正确 ✓）和单元测试，只体现在**步数/耗时**这类结构性指标上。因此对 MTP 这类"可开可关"的特性，
> 必须把**结构性判据**（token/步）纳入验收，而不是只比响应。

### 7.2n 冷启动（#4）**受显存容量约束**，不是策略问题（Round 38 收口）

`StatePool::make_room` 的准入条件（`pool.rs:70`）：

```
active_budget + pool.bytes + needed <= self.budget     // 记账预算（limit − 权重）
&& states.len() + pool.entries.len() < maximum_states  // 64，不构成约束
&& needed <= self.available()                          // 实际空闲显存
```

本机：权重 ~27 GB / 32.6 GB ⇒ 可用 **~2.4 GB**；每个 program 的记账约 0.7 GB（Round 7 精确 arena 之后），
实际含 32-lane prompt arena（实测释放约 324 MB）+ 各 lane arena + KV。结论：**池只能容纳 1–2 个 program**。

于是 Round 10 的实测是**必然而非实现失误**：预热 bucket 32+64 后，流量自己的 bucket 无法被保留（容量已满），
每个请求重新捕获 ⇒ 热请求 107 → 390 ms。

**因此 #4 的真正解法只有"降低每 program 内存"，而那正是跨序列批处理计划里的步骤 C（图与工作内存按模型共享）**
——即 #4 与本轮被排除的那一项**同源**。在不动共享化的前提下，可做的只有：

- 接受冷启动（每 bucket 一次 ~1.35 s，之后同 bucket 热到 108 ms）；本机显存不支持"预热多个 bucket 而不挤掉
  流量自己的条目"。
- 或换更小的权重占用（fp8 KV / 张量并行 / 更小模型），那是部署选择而非引擎改动。

**不要再尝试 naive 池预热**——两次分析（Round 10 实测、本轮算术）都指向同一结论。

### 7.2o 剩余项的真实体量（Round 39 定性）

**#2 异步通路：接口卫生，无运行时效果。** CUDA 后端**没有**实现 `set_waker` / `pending_resource_releases` /
`resource_epoch` / `requires_async_checkpoint`（`grep crates/backend/cuda/src/executor` 无匹配）⇒ 包装层补的 4 个
委托是**正确的接口契约**（避免将来实现这些方法时又被静默吞掉），但今天不改变任何行为。真正修好的是
`speculation_capability`（已实测）。

**#7 prefix cache：CUDA 后端完全未实现，是"新特性"而非"差个开关"。**
- 引擎侧机制齐备：`pipeline/scheduling/resources.rs:36 reuse_prefix`、`runner/control.rs:114 reuse_prefix_shared`、
  `ExecutionStats.prefix_hits/prefix_entries/prefix_bytes` ✓
- 后端侧：`crates/backend/cuda/src/executor` 里 **0 个** `reusable_prefix*`/`reuse_prefix*` 实现 ⇒ trait 默认
  （不可复用）生效，前缀缓存对 CUDA 不生效。
- 实现体量：需要前缀哈希/匹配 + 把已有序列的 KV 快照复制进新序列（后端已有 `capture_execution_state`/
  `restore_execution_state`/checkpoint 复制设施可复用），属**中等偏大**特性。

**#5 融合内核：体量已知偏大、收益上限 3–4.5%**（需新 IR 算子 + CUDA kernel + 编译器发射条件，Round 34 查明）。

⇒ 在"除跨序列批处理外都搞完"的口径下，真正的剩余是**两个特性**：#7（prefix cache，中等偏大、收益依赖流量）
与 #5（融合，偏大、收益 3–4.5%）；而 #4 已被证明与跨序列批处理同源。其余（#1 MTP 推测、#3 准入排队、
#6 诊断替代、#2 接口卫生）已完成。

### 7.3 优化前台（按实测重新排定）

实测画像（RTX 5090 32.6 GB / Qwen3.8-27B-NVFP4，服务验收与基准）：

| 指标 | 实测 | 参照 |
|---|---|---|
| 稳态 TTFT | **110 ms**（MTP 125 ms） | 冷启动 1342/1962 ms，每 capacity bucket 一次 |
| 单步 decode | 57.3 tok/s，TPOT 17.5 ms | roofline 93.8 tok/s（61%） |
| 并发 | **3**（可用显存 2.7 GB 限制；原为 1） | 精确 arena 记账解锁 |
| 聚合吞吐 | ≈ 单流（调度交错，非真批处理） | 真批处理应 ≈ ×并发数 |
| 数值正确性 | resident 63/63、greedy 逐 token 等价 | — |

| 优先级 | 前台 | 现状与依据 |
|---|---|---|
| **1** | **跨序列批处理**（可执行计划见[连续批处理实现计划](continuous-batching-plan.md)） | lane 机制齐备（3-lane verify、32-lane prompt、`Dispatch32` 的 Batched/Row、`update_metadata` 的 per-lane token/position/state_pos）；缺的是让不同序列的 KV 能被同一张图寻址——KV 张量前导 `1` 维度即天然 lane 位。收益 = 吞吐 ×并发数 |
| 2 | 单步 decode 61% 带宽 | 1155 kernel/步 的启动开销；**已排除**：三维累加器改二维（cuTile 表达不了，§7.2b）、缩小 BK（NVFP4 的 `BP=BK/2`/`BS=16` 写死）。方向剩 kernel 融合 |
| 3 | 准入排队化 | `admission.rs` 把拒绝直接变成 `Err(Capacity)`，瞬时内存不足即判死。**勘察结论（Round 24–25，已修正）**：提交在 agent 里是一个 **RPC 命令** ——
`crates/service/agent/src/commands/runtime.rs:124 fn submit` → `context.engine.submit(request)?`（配对的是
`runtime.tick`）。也就是说驱动这个 RPC 循环的**客户端**（HTTP frontdoor）才是该做"待提交队列 + 重试"的地方：
它拿到 `Capacity` 时不要立刻回错，而是保留请求、下一轮 tick 后重试。**不要**改引擎生命周期（会牵动大量
runtime 测试）。Round 24 说的"agent 层"不准确：agent 只是被动执行 RPC。**现成模板**：`crates/service/agent/src/experiment.rs:130` 已经用"submit → 反复 `tick` 直到 idle，超时预算封顶"的模式（那里把超时表达为 `ErrorCode::Capacity`）——把同一模式用于服务提交即可：拒绝时把请求留在待提交队列，下一轮 `tick` 后重试，超过 `max_queue_wait_us` 再以 Capacity 失败 |
| 4 | 冷启动按 bucket 捕获 | **已排除**：naive 池预热是净伤害（§7.2g）。要做需让池预算与预热 bucket 数联动 |
| 5 | prefix cache、异步执行 | 同步 provider 无法 overlap；`ResourceCommand::Prefix` 已有接口 |

**长 decode 的总线/占用实测（Round 22，补齐 Round 14 未做成的那次）**：用 `eos_token=0` 关掉早停，
强制生成 512 token，并让 `hardware-monitor`（1 s 粒度）覆盖整个 decode 窗口（tail busy 样本 10 个）：

| 指标 | 解码窗口实测 |
|---|---|
| `utilization.gpu` | **87.8%** |
| `utilization.memory`（显存控制器忙时占比） | 均值 **60.6%**，峰值 68% |
| `power.draw` | 360 W（上限 575 W，非功耗受限） |
| `memory.used` | 24.0 GiB（22 GB 权重常驻）+ ~1.4 GiB 状态 |

结论（修正 Round 14 的"没有 profiler 就不该投入"）：decode **不是**把 DRAM 打满的状态——显存控制器只有
~2/3 时间忙、GPU 88% 忙。结合"达成带宽 = datasheet 的 70%"，说明剩余缺口的性质是**每步的串行阶段与
kernel 启动间隙**，而不是"DRAM 已饱和"。因此**kernel 融合 / 削减每步启动次数仍是合法的第二杠杆**（1155
kernel/步），第一杠杆仍是跨序列批处理。

**第二杠杆（每步开销）的可行性边界（Round 23）**：把"削减每步开销"的几条低成本路径逐一验证后排除了——

| 候选 | 结论 |
|---|---|
| metadata 每步更新的分配 | **已最优**：`resident/metadata.rs` 用一次 `GraphUpdate` + 标量实参写入，模块文档明确"无临时设备分配"（且用 `f32::from_ne_bytes` 传位模式规避 cuTile 对整型标量的 DivHint 特化） |
| 直接 H2D 写入既有设备缓冲（绕开图更新） | **不可行**：cutile 0.4 只有 `api::memcpy`（设备→设备，仍是一次图操作）与 `copy_host_vec_to_device`（会分配），没有 write-into |
| 减少 Linear 之外的重复 kernel | 未发现重复；每个节点一次启动 |

所以第二杠杆**只能靠融合内核**（新 cuTile kernel），可列出的融合点与量级：

| 融合 | 每步省下的启动 |
|---|---|
| `rope` + KV `append` | ~64 |
| MLP `silu` + `mul` | ~64 |
| `norm` + 残差 `add` | ~64 |

合计约 **192 / 1155 次启动（~17%）**，按 2–3 µs/启动估算 ≈ **0.4–0.6 ms / 17.5 ms（2–3%）**。也就是说第二杠杆
量级是"几个百分点"，而控制器空闲的其余部分来自**小内核之间的串行依赖**（融合能缓解但不能消除）。这进一步
说明：**跨序列批处理才是数量级的那一项**（吞吐 ×并发数）。

**关于"decode 61% roofline"的修正**：22 GB 权重 / 17.5 ms = **1257 GB/s = datasheet 1792 GB/s 的 70%**。
对 batch-1、每个权重只读一遍的 GEMV 来说 70% 属于正常区间（datasheet 值本身不可达），所以"还有 39% 余量"
是我早期的乐观口径，**不应据此投入**。本机没有 `ncu`/`nsys`，1 s 粒度的 `hardware-monitor` 也无法隔离
17 ms 的 decode 步（实测里混入了权重加载阶段）；要判定剩余缺口来自 kernel 启动间隙还是访存效率，
需要 profiler 或时间对齐的长 decode 采样。实测旁证：busy 窗口内 SM 2460 MHz、功耗 144 W（上限 575 W）
——不是功耗/温度受限。

**verify tile 几何：纠正 + 实测（Round 16）**。此前写"BK 被 NVFP4 的 `BP=BK/2`、`BS=16` 锁死在 256"是
**错的**：`BS` 是**每 k 块的 scale group 数** = `BK/16`，不是常量（BK=256 时恰好也是 16，所以当时看不出
来）。据此试了合法且未试过的组合 **BN=16 / BK=128**（累加器体积与现状相同，仍 8192）：

| depth 2 | BN=8 / BK=256（现用） | BN=16 / BK=128 |
|---|---|---|
| TPOT | 17.84 ms | **20.37 ms** |
| decode | 55.8 tok/s | **49.1 tok/s** |

即**更宽的 BN 换更浅的 k 是净负**（-14%），说明该 kernel 对 **k 循环深度**比输出宽度更敏感。已回退并复测
（resident 63/63、depth 2 TPOT 18.23 ms / 54.9 tok/s）。`VERIFY_SCALE_DEPTH` 的推导已修正为
`VERIFY_TILE_DEPTH / NVFP4_GROUP_SIZE`（对任意 BK 成立，不再是巧合）。

**已被实测否决的方向（勿重走）**：verify GEMV 二维累加器（cuTile 类型系统）、缩小 BK、naive 池预热、
decode 期释放 prompt 图（池复用回归，§7.2h）。

**验证纪律（两轮教训）**：任何影响 `Sequence` 生命周期 / 池复用 / 捕获时机的改动，必须用
`tools/bench/check-cuda-service.py` 的**串行服务请求序列**验证——单次 `run`、新进程基准都会漏掉池路径。

