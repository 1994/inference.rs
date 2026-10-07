# 当前续接：continuous batching 与 MTP 联合验证（2026-10-07）

接续 Kimi 会话 `9b39a747-be26-491d-b720-e4f20bc92406` 中断时的实现。
以下历史章节保留当时的测量；其中“draft 加载时不建池”“投机过的序列永久退出批路径”
已由本节实现取代。

- 引擎继续使用已有的跨后端 `StepPlan` / `ExecutionTask` 批契约；CUDA 将每条序列的
  草稿链展平为 `slot × verify_width`，共同运行 target 验证图。
- 每个槽位分别维护 KV 游标和 conv/delta 检查点；接受前缀按槽位提交，拒绝部分独立回滚。
  替换 token 由引擎使用最后接受位置的 logits 采样，下一步再提交，避免额外 target 回放。
- 草稿仍逐序列生成；整条草稿被接受时只补跑最后一个 draft token。非贪心采样或接近
  容量边界时，联合图只运行该序列的有效前缀，其余候选 lane 屏蔽。
- 曾经串行执行 MTP 的序列可以迁入共享槽位；已租用槽位的序列始终使用共享图，
  不会误回放已经过期的私有 target 状态。释放和复用沿用原有状态池生命周期。
- 修复 `tools/bench/mtp-ab.py` 只比较最后一个 depth 的错误：现在所有 depth、每个
  request 的 token 都参与比较；缺失、失败或不一致会输出失败并返回非零退出码。

当前边界：槽位宽度仍为 3、容量仍为 4096；请求仍持有原有私有程序，槽位图另占显存。
显存不足时回退串行，不代表 S2/S3 的共享程序与准入计费重构已经完成；也没有取得与
vLLM 同负载的性能对比结论。更长请求仍使用原有串行路径。

验证（RTX 5090 / Qwen3.8-27B-NVFP4）：

| 检查 | 结果 |
|---|---|
| CUDA resident 原有数值检查 | 63 项通过 |
| 三槽位 GPU 专项 | 接受 0/1/2 个草稿、独立回滚、空闲掩码、乱序任务、短链、重新绑定均与私有验证图一致 |
| 双并发、每请求 64 token | depth 0/1/2 逐 token 一致；提交步数 66/39/35 |
| 实际图路径 | `slot_verify` 回放 68 次：36 次 4 个有效 lane、32 次 6 个有效 lane；确认不是串行回退 |
| 单序列、64 token | depth 0/2 逐 token 一致；提交步数 66/27 |
| HTTP 服务 | depth 0/2 各 7 请求验收通过，覆盖四次串行复用、双并发、默认采样、正常关闭 |
| 静态/CPU 检查 | 完整 Rust 门禁、CUDA feature 全目标 Clippy、CUDA 非硬件单测、脚本 Ruff 全部通过 |

双并发取证运行开启图内 profiling，不能把该组耗时当作无 instrumentation 的性能基线。
此次完成的是 MTP 与 CB 同时执行及状态正确性；固定三槽位和私有程序冗余的成本仍需后续优化。
原始报告和门禁日志保存在 `artifacts/validation/cb-mtp-20261007/`。

复现专项 GPU 测试（须设置本机适用的 CUDA/Clang 环境）：

```sh
bash tools/bench/safe-run.sh --memory-gib 4 cargo test --locked --release \
  -p infer-backend-cuda --features cuda --lib pooled_verify_preserves -- --ignored --nocapture
```

---

# 从这里继续（Handoff）

> 本文件是全项目优化的唯一入口。下面是**当前状态 + 下一步**，按"已实测"的顺序排列。
> 所有数字都有出处；被实测否证的假设也列在最后，**不要重复**。

## 已完成（本会话，全部经 GPU 验证）

| 项 | 结果 | 验证 |
|---|---|---|
| MTP 推测静默失效 | `SelectedBackend` 未委托 `speculation_capability`（默认 0 ⇒ 引擎从不传 `sampling`） | 修复后 27 步/2.4 token 每步、64/64 token 等价、热请求 110 ms |
| 包装层默认方法 | 补齐 4 个委托（含 `set_waker`） | 接口审计 35 个默认方法 |
| 准入排队重试 | `submit` 遇 Capacity 入队，`tick` 排空；`query` 上报 pending/expired | agent 12 测试 + 服务验收 6 项 |
| **prefix cache** | 默认开启（容量 `state_budget/4`，下限 128 MiB）、设备侧分行拷贝、哈希去重、LRU | 13 单测 + 63/63 + 服务验收 6 项 |
| op 取证 | provider check 输出 `operations`/`graph_nodes`/`op_census` | 实测 1155 派发；linear 497 / norm 161 / add 128 / mul 80 / silu 64 |

## 下一步（按性价比）

### A. ~~融合轻量算子~~ **已被实测否证，不要做** ✗
`silu+mul` 融合**已完整实现并跑通**（录制层 + `capture32` + 竞技场布局感知，见 §5f），
`cuda-resident-check` 63/63 通过、真实提示路径不再报错，但 **5 次重复测量显示慢 ~1.2%**（噪声地板 ±0.8%）
⇒ **无收益**，已回退。由此否证 Round 71 的"per-kernel 31 µs 均匀分布"推论：
**36 ms 集中在少数重 kernel 上，elementwise 小 kernel 极便宜** ⇒ 减少派发数不是杠杆 ✗
（"norm+add+silu+mul 一批可拿 20%"的说法同样被否证 ✗）

### B. **给 prefill 上 instrument（当前唯一正确的下一步）** → 见 §5g
先用 black-box 已穷尽推论（7 个假设被否证）。必须实现 **CUDA event 计时段**（Round 44 勘察：
两个常驻 event + `cuEventElapsedTime`，约 30–50 行；`execute` 是 `&mut self` 且同步 ⇒ 事件可常驻），
才能知道 **36 ms 落在哪类 kernel** 上。**在此之前不要再提新假设**（§5g 的四条约束必须同时成立）。

### C. 跨序列批处理（数量级，用户曾暂缓）→ 见 §1–§5
- 现状：并发 3（准入已精确记账），但**聚合吞吐 ≈ 单流**
- 新证据：prefill **延迟受限**（GPU 99% 忙但显存控制器仅 ~20%）⇒ 并发填充收益**比原估计更大**

### D. 长 prompt 的既有收益（**已完成**）
prefix cache 已默认开启（§5b）：分区拷贝、哈希去重、LRU。按实测 prefill 1.8 ms/token，
复用 1k token 前缀约省 **1.8 s** ✓

## 关键基线（RTX 5090 / Qwen3.8-27B-NVFP4）

| 指标 | 值 |
|---|---|
| 稳态 TTFT | 110 ms（MTP 113 ms） |
| 冷启动 | 1260 ms/进程（图捕获）；服务首请求 1340 ms |
| decode | 17.5 ms/token = 57.3 tok/s = datasheet 带宽 70% |
| prefill | 57 ms / 32-token chunk（= 带宽下限 4.7×）；1.8 ms/token |
| 并发 | 3；聚合吞吐 ≈ 单流 |
| 引擎稳态分配 | 0（`gate.sh cpu` 强制） |

## 验证纪律（每次改动必须）

1. **先看 GPU 检查再跑门禁**：`gate.sh rust` 的 release 构建**会覆盖** `target/release/infer`（不带 cuda feature）⇒ 之后任何 GPU 检查都会报 "CUDA execution requires Linux and a build with --features cuda"（误导性）。
2. 改动若触及 `Sequence` 生命周期 / 池复用 / 捕获时机 ⇒ 必须跑 `tools/bench/check-cuda-service.py`（4 条串行请求 + golden 校验），单次 `run` 会漏掉池路径。
3. kernel 改动 ⇒ **先看 63/63**（`cuda-resident-check`），再看性能。
4. 实验脚本要带**自动回退**（改坏了立即 `cp` 备份恢复并重建）——Round 69/70 靠这个没把回归留在树里。
5. 收尾跑 `bash tools/check/gate.sh rust`。

## 已否证的方向（不要重复）

| 假设/尝试 | 否证方式 |
|---|---|
| cuTile 2-D 累加器 | 编译/类型错误（Round 2） |
| 缩小 decode BK / BN=16+BK=128 | 实测更慢（Round 2） |
| naive 池预热 | 热请求 107→390 ms（Round 10） |
| decode 期释放 prompt 图 | 热 TTFT 107→388 ms（Round 11） |
| 快照整块拷贝是长 prompt 元凶 | 关/开快照只差 133 ms（Round 58） |
| prefill 走 FP32 SIMT | 读码：走 `gemm::nvfp4`（Round 63） |
| 每 chunk 同步+读回是元凶 | 读码：中间 chunk 不读回（Round 66） |
| prefill 占用率不足 | 实测：N tile 减半反而慢 2.8×（Round 69） |
| N tile 加宽到 128 | cuTile 类型约束编不过（Round 70） |
| cuTile `load_pipelined` hint | 图内 −3%、墙钟在噪声内，已回退（Round 87） |

# 连续批处理（跨序列批处理）实现计划

本文是第 1 优先优化项的可执行计划。**为什么是它**：decode 是带宽受限的——batch-1 每生成一个 token 都要
把全部 22 GB 权重读一遍。因此并发 N 条序列时，调度器交错执行 N 次完整权重读取，聚合吞吐 ≈ 单流
（实测 57 tok/s，与单序列一致；3 条并发时每条各得约 1/3）。只有让 N 个 token **共享一次权重读取**，
聚合吞吐才会变成 ≈ N × 57 tok/s。vLLM 的 continuous batching 正是这一点。

## 1. 已具备的条件（不要再造）

| 机制 | 位置 | 现状 |
|---|---|---|
| per-lane metadata（token / `RoPE` 位置 / state 位置 / override） | `resident/batch.rs::update_metadata` | 已按 lane 写 4 个字段，prompt 尾部 lane 的 state 位置会置 `-1` 掩码 |
| 节点分流 Batched / Row / Linear | `resident/batch.rs::dispatch32` | shape-agnostic 辅助算子一次覆盖整批；需要标量输入的（embedding、state 写入）逐 lane |
| 32-lane 批图与 per-lane arena 行 | `ActivationArena::new_batched`、`CaptureMode::Row(lane)` | 已能寻址"第 lane 行" |
| state 位置滞后（MTP `kv_offset`） | `BatchGraph::state_offset` | 已按程序类型记录，随 metadata 生效 |
| 后端任务串行执行 | `BackendProvider::execute(&mut self, …)` | 单流、同步 → 批内不需要锁 |

## 2. 缺的一块：KV 不可跨序列寻址

KV 张量形状为 `[1, CAP, D]`（`attention::append` 的 `keys: &mut Tensor<E, { [1, CAP, D] }>`），
那个前导 `1` 就是天然的 lane 位，但**目前被烧进每序列自己的捕获图**——图中记录的是该序列 state 张量的
指针，所以一张图只能服务一条序列。这正是"每序列一套 `DeviceProgram`"的根因（也解释了 1.34 s/bucket
的捕获成本）。

### 步骤 A：KV lane 化
1. `allocate_states` 把 `AttentionKv` 张量按 `[MAX_LANES, CAP, D]` 分配（`MAX_LANES` 取 `max_num_seqs`
   或直接的 3/32 复用现有常量），其余 state 保持原样。
2. `attention::append` / flash-attention 读取 KV 的路径接受一个 **lane 标量参数**（kernel 已支持标量入参，
   如 `epsilon`/`offset`），由 `Capture` 用当前 lane 传入；`CaptureMode::Row(lane)` 已经是逐 lane 记录的，
   所以只需把 lane 传给内核，而不是新造机制。
3. metadata 增加/复用一个字段承载 lane（当前 4 个字段全用满：rope 位置、token、state 位置、override；
   `METADATA_FIELDS` 加 1 是常量改动，append 与注意力内核同步读取）。

### 步骤 B：引擎批契约
1. `ExecutionInput` 增加"批 decode"变体：`Decode { position, token }` → 增补
   `DecodeBatch { start: usize, tokens: &[u32] }`（每 lane 一条序列的当前 token，position 由各序列自身推进）。
2. `ModelOutput.tokens` 的既有语义（后端可决定 token 前缀）要扩展为**每 lane 各自的 token**；引擎侧
   `stages/output.rs` 已经能把 `output.tokens` 追加进 `sampling.generated`，但需要按 request 分派而不是
   按单序列。
3. 调度器：每 tick 从 ready 队列取**不同序列**的 decode，凑成一个 batch（lane 数 ≤ 图宽），一起提交。
   现有 `max_num_seqs` 配置即批宽上限。

### 步骤 C：内存与准入
- 每序列的预留随之降到 KV 之外的少量结构（arena 行/元数据），`sequence_budget` 需同步改为按 lane 计费。
- 图与工作内存**按模型共享**（一张批图 + 一个批 arena），这是 1.34 s/bucket 捕获成本与 2.71 GB/序列
  内存的最终解法；`StatePool` 的键随之简化（不再需要按 capacity 分桶，CAP 固定为池大小）。

## 3. 必须保持的不变量（违反即回归）

1. **贪心等价**：`temperature == 0` 时输出序列必须与单序列路径逐 token 一致（现有 `tools/bench/mtp-ab.py`
   的 `greedy-equivalent` 检查，以及 `check-cuda-service.py` 的响应比对）。
2. **尾部 lane 掩码**：批内不足宽的 lane，其 state 写入位置必须是 `-1`（`update_metadata` 现有行为），
   否则会污染未参与序列的 KV。
3. **state 位置 vs `RoPE` 位置分离**：MTP 路径的 state 位置滞后 `MTP_KV_OFFSET`，不得混用
   （`BatchGraph::state_offset`）。
4. **池键语义**：`StatePool::take` 依赖 `next_power_of_two(capacity)` + readout；CAP 固定后要一并调整，
   否则命中率骤降（第 7.2g 节的教训：池的价值来自"只缓存流量真正复用的形状"）。
5. **readout 语义**：`OutputReadout::Full` 与 `None` 的分支（hidden/logits 回读）不得因批处理而丢失。

## 4. 验证协议（每一步都要跑）

| 目的 | 命令 | 通过标准 |
|---|---|---|
| 数值正确性（63 项） | `cuda-resident-check`（经 `safe-run.sh`） | 63/63 PASS |
| 贪心等价 | `tools/bench/mtp-ab.py --depths 0,2` | `greedy-equivalent True` |
| **池复用路径**（两轮教训） | `tools/bench/check-cuda-service.py`（内部 4 次**串行**请求） | `passed=true`，且热请求墙钟不劣于 ~110 ms |
| 并发 | 3 请求的 `run`（`/tmp/ttft-probe/long-n3-t16-rep0-d0` 形态） | 全部准入；**聚合吞吐应 ≈ 并发数 × 单流**（这是本项的核心指标） |
| 启动/冷路径 | 同上，看第 1 次墙钟 | 不应劣化（共享图后应显著改善） |

### ⚠ 验证顺序陷阱（Round 30 实测）

`tools/check/gate.sh rust` 的 release 构建**不带 `--features cuda`**，会**覆盖** `target/release/infer`。
之后任何 GPU 检查都会以一句很有误导性的话失败：

```
{"code":"Unsupported","message":"CUDA execution requires Linux and a build with --features cuda"}
```

所以顺序必须是：**先跑 GPU 检查（需要 cuda feature 的二进制），或每次 GPU 检查前先
`cargo build --release -p infer-cli --features cuda`**。`gate.sh rust` 放最后。

另：`--op-trace <path>` 实测产出**空数组**，已查明原因——**CUDA 后端的 `traces()` 是写死的空实现**：

```rust
// crates/service/cli/src/backend/selected.rs:264
Self::Cuda(_) => vec![],
```

（`ExecutionStats` 里有 `trace_dropped`，说明 trace 机制存在，但 CUDA 路径没有导出。）所以：

- 要做"每步 kernel 构成"的取证，需要二选一：给 CUDA 实现 `traces()`，或给 `ExecutionStats` 加一个**每步
  启动计数**（更便宜，够用于验证融合收益）。
- **已取证（Round 32）**：不必实现 `traces()`——`cuda-provider-check` 手里就有 `ExecutionProgram`，
  直接报出编译算子数与图节点数即可。真实模型实测：

  ```
  operations: 1155    graph_nodes: 1155    (Qwen3.8-27B-NVFP4, 1 请求)
  ```

  即"每步 1155 次派发"从**估算**升级为**实测**。
  （注意：op 与 node 一一对应，每个 op 可能含 1 个以上 kernel，所以 1155 是**下界**。）

- **融合机会的精确计量（Round 33，同一实测）**：provider check 现在还报 op 直方图：

  ```json
  {"linear":497, "norm":161, "add":128, "multiply":80, "silu":64,
   "gated_norm":48, "split":16, "embedding":1}      // 合计 995，其余 ~160 为 attention/rope/state
  ```

  据此可算的融合上限：`silu`+`multiply`（MLP 门控）~64 次、`norm`+`add`（残差）~128 次、
  `rope`+KV `append` ~64 次 ⇒ 合计 **~256 / 1155（22%）**，按 2–3 µs/启动 ≈ **0.5–0.8 ms / 17.5 ms（3–4.5%）**。
  `gated_norm` 已有 48 个，但**查明它不是通用融合框架的产物**（Round 34）：它由模型编译器在
  **特定架构模式**处手工发射（`crates/model/recipes/src/decoder.rs:427`，linear-attention 的
  delta+z+weight 门控归一化），对应的 IR 是 `TensorOp::GatedNorm { head_dim, epsilon }`。因此要做
  `norm+add` / `silu+mul` 融合，必须**新增 IR 算子 + 对应 CUDA kernel + 编译器发射条件**——是编译器+kernel
  的联合改动，而不是"加一个 pass"。以 3–4.5% 的收益衡量，性价比低于跨序列批处理（数量级）。

  **结论（证据等级：实测计数 + 估算换算）**：第二杠杆上限是个位数百分点，跨序列批处理仍是唯一数量级项。

## 5. 已确立的死路（勿重走）

- verify GEMV 三维累加器改二维：cuTile 的 `reduce_sum` 在 k 循环内丢失 const-generic 形状。
- 缩小 `BK` 减小累加器：NVFP4 的 `BP = BK/2`、`BS = 16` 写死在 kernel 里。
- naive 池预热：挤掉流量真正复用的条目，热请求 107 → 390 ms。
- decode 期释放 prompt 图：序列回池后无 prompt 图，复用它的请求退化成 3-wide prefill，热请求 107 → 388 ms。

## 5b. #7 prefix cache 实现计划（**默认开启、设备侧高效实现**）

### 定位
prefix cache 是**基本特性**（不是可选项）：默认开启，容量由显存预算推导，不新增用户配置项。

### 已定的关键事实（勘察所得）
- **引擎侧已完成**：`pipeline/scheduling/resources.rs::reuse_ready_prefixes` → `ResourceCommand::Prefix { state, tokens: prompt, maximum }`
  → 后端 `reuse_prefix_shared`；接口语义：`reusable_prefix*` 返回可复用前导 token 数，`reuse_prefix*` 挂载并返回实际长度。
- **参考实现**：`crates/backend/metal/src/executor/prefix.rs`（97 行）——`CachedPrefix{tensors,tokens,hidden,logits}` +
  `candidate_prefix`（按 block 边界匹配）+ `encode_pending_prefix`（设备拷贝）+ `cache_boundary`（快照）。
- **高效实现的底座已有**：`cuda_core::api::memcpy_dtod_async<T>(dst_ptr, src_ptr, num_elements, stream)`
  ⇒ 快照与回填都是**流上的一次设备间拷贝**，零 host 往返。
- **CUDA 差异**：KV 在每序列 `DeviceProgram` 内（`allocate_states` 产生 `BTreeMap<TensorId, Vec<Tensor<f32>>>`），
  且该 map **当前不是字段**（只被图 retain）⇒ 必须先让状态张量可从 `DeviceProgram` 访问。

### 设计（默认开启 + 高效）
- 容量**推导**：`prefix_cache_bytes = state_budget / 4`（下限 128 MiB；设备预算过小时自动为 0），
  **不新增 CLI 开关**；命中/字节数走既有的 `PoolInspection` 风格计数以便观测。
- 存储：`VecDeque<CachedPrefix>`（LRU，按字节淘汰）；`CachedPrefix { tokens, states: BTreeMap<TensorId, Vec<Tensor<f32>>>, bytes }`，
  **设备常驻**，不做 D2H。
- **去重**（高效的关键）：按前缀 token 的哈希索引；同一热门前缀只保留一份快照（避免 N 份重复拷贝）。
- 匹配：按 block 边界比较 token 向量，取最长命中且 ≤ `maximum`。
- 回填：对每个 state 张量一次 `memcpy_dtod_async`（流上顺序执行，随后 submit 自然有序）。
- 边界条件：仅对 `OutputReadout::Logits`（Generate 主路径）缓存/回填；`Full` 读出需要 hidden 边界快照，
  首版直接**不缓存**（安全回退，引擎会走完整 prefill）。

### 切片计划（每片都可独立编译 + GPU 验证，不留半成品）
- **A. 原语**：
  - ✅ **已完成**：`DeviceProgram` 持有 state 张量并暴露 `states()` / `states_mut()` / `fp8_states()`
    （[program.rs](../../crates/backend/cuda/src/resident/program.rs)，声明在图之后保证 drop 顺序；
    `States` 别名提为 `pub`）。**GPU 验证：`cuda-resident-check` 63/63 PASS**——这一步必须验，因为 state
    张量从"仅被图 retain"变成"结构体持有"。
  - ⏳ **待完成**：`CudaDevice::copy_d2d`（封装设备间拷贝）。**已查明的坑**：`cuda_core::api` 是私有模块
    （不能用 `memcpy_dtod_async`）；只能直接调 `cuda_core::sys::cuMemcpyDtoDAsync_v2`，但需要 **context 的
    stream 句柄**：`readback.rs:216` 的写法是 `context.get_cuda_stream().cu_stream()`，其中 `context` 是那个
    函数的**入参**——所以下一步要先找它的**调用方**，看 `CudaContext` 是从哪里取的（`Device::bind_to_thread()`
    只做绑定、返回 `()`，**不是**它）。找到后 `copy_d2d` 就是 15 行，并用 provider check 的 d2d 逐位相等探针验证。
- **B. 缓存与匹配**：`CachedPrefix` + LRU + 哈希去重 + `reusable_prefix_for`。
  **此片刻意让 `reusable_prefix*` 仍返回 0**（匹配结果先只计数/断言），⇒ 默认行为零变化，可用 63/63 + 服务验收锁死。
- **C. 快照与回填**：prefill 消费完 prompt 时做快照；`reuse_prefix_shared` 设备拷贝回填并返回长度。
  此片完成后特性真正生效（默认开启）。

  **集成细节（Round 49 查清，写代码前必读）**：
  1. 引擎路径**只调用 `reuse_prefix_shared`**（`pipeline/scheduling/resources.rs:40` 直接发
     `ResourceCommand::Prefix`，从不先问 `reusable_prefix*`）⇒ 那两个「查询」变体保持 trait 默认即可，
     **不要**为了它们写返回非 0 的匹配逻辑（会导致"报了长度但不恢复"的危险组合）。
  2. 返回值的语义与后果（`engine/runtime/src/resource/mod.rs:253`）：
     `ResourceReply::Prefix(tokens)` → `self.state.ensure_tokens(state, tokens)` + `self.state.commit(state, tokens)`
     ⇒ **引擎会把 tokens 记为"已 prefill"**，后续只发剩余 token 的 prefill 任务。
     因此后端必须让执行器侧一致：**把 `Sequence.history` 填成命中的前缀 token**，并把
     `DeviceProgram` 的 `next_position` 设为命中长度（需要一个 `set_position` 之类的小接口），
     否则 `position != history.len()`、KV 位置从 0 开始 ⇒ 静默错误输出。
  3. 只有 `OutputReadout::Logits`（Generate 主路径）可复用：`Full` 还需要边界处的 hidden/logits 快照，
     首版**直接返回 0**（安全回退，引擎走完整 prefill）。
  4. 快照分配用 `api::zeros::<f32>(&[elements])`（例：`resident/program.rs:49`），随后逐张量
     `CudaDevice::copy_d2d`（A 片已完成）；容量已在 C1 落地（`state_budget / 4`，下限 128 MiB，默认启用）。
  5. **借用结构的坑（Round 51 读 `execute` 后得出）**：快照点必须写在 `executor/execution.rs::execute` 的
     `tasks.iter().map(|task| { let state = self.states.get_mut(..)?; state.run(task) })` 里，但该闭包同时要
     用 `self.loaded.device()`（拷快照）与 `self.prefix.insert(..)`：
     - `self.states`（可变）与 `self.loaded` / `self.prefix`（另一个字段）在 Rust 2021 的**不相交字段捕获**下
       可以共存，但 `state` 这个 `&mut` 存活期间再调 `self.prefix.insert` 仍会 E0502。
     - 正确写法：**在循环内先构造 `Vec<CachedPrefix>`（只借用 `self.states` 与 `self.loaded`），循环结束后再逐个
       `self.prefix.insert(...)`**。这样借用不重叠，也不需要 `RefCell`。
     - 同样的顺序要求适用于 `reuse_prefix_shared` 的回填（先 `self.prefix.take_match` 拿到拥有所有权的
       `CachedPrefix`，再借用 `self.states` 写回）。
  6. **C2 的未知量已全部关闭（Round 52）**：
     - `TokenBuffer` 实现 `Deref`（`foundation/ir/src/tokens.rs:23`）⇒ `reuse_prefix_shared` 里直接用
       `&tokens[..]` 当 token 切片，无需额外访问器。
     - 快照分配 `api::zeros::<f32>(&[elements])`；元素数取 `Tensor::size()`。
     - 命中/未命中：`PrefixCache::take_match(&[u32], maximum)`，未命中或 `readout != Logits` 一律
       `Ok(0)`（引擎随后照常完整 prefill，安全）。
     - 回填后必须：`Sequence.history` 填成命中前缀 + `DeviceProgram::set_position(matched)`（C1 已加）。
     - 容量推导已有单测（`prefix_capacity_tests`：四分之一、下限 512 MiB 预算起效）。
  7. **C2b 的效率关键点（Round 54 得出，必须照此实现）**：state 张量按 **`capacity`** 分配，若整块拷快照
     会是数百 MB/次 ⇒ **只拷"已覆盖的 token 行"**：
     - 快照粒度：`history.len() % PREFIX_GRANULARITY == 0`（建议 256，与 §5c 的 chunk 尺度一致），
       即每若干 chunk 一个复用点；`tokens` 取**当时的 `Sequence.history`**（正是快照覆盖的前缀，诚实定义）。
     - 每张量只拷 `covered_rows * row_elements` 个元素 ⇒ 256 token × 每行元素 ≈ 数 MB/快照 ✓
     - **因此 `copy_d2d` 需要支持元素偏移**（当前只有 `(dst, src, elements)`，从 0 开始）：
       扩展为 `copy_d2d_at(dst, src, dst_offset, src_offset, elements)`，或给现有函数加两个 offset 参数
       （`cuMemcpyDtoDAsync_v2` 本身接受裸指针，加偏移就是指针加法 ✓ 廉价）。
     - 回填同理：只写回覆盖行，且必须**先确认目标 `capacity` ≥ 命中长度**（否则 `Ok(0)` 回退）。
     - `CachedPrefix.bytes` 按**实际拷贝字节**记账（不是整个张量），这样容量推导的 `state_budget/4`
       才能容纳多个前缀条目 ⇒ 命中率与内存同时合理。
  8. **C2b 其余事实（Round 55 查清）**：
     - `ExecutionInput` 只有三种（`foundation/ir/src/tokens.rs:100`）：`Full(TokenBuffer)`（整段 prompt，
       prompt 图路径）与 `Prefill { span, readout }`（分块）都是 prefill，`Decode { position, token }` 是解码
       ⇒ 快照钩子的判据：`matches!(task.tokens, ExecutionInput::Full(_) | ExecutionInput::Prefill { .. })`。
     - 快照的 `tokens` = **`Sequence.history`**（run 之后已包含本次 chunk 提交的 token）✓，与回填时写入的
       `history` 语义一致。
     - `copy_d2d_at(dst, src, dst_offset, src_offset, elements)` 已实现（含"范围必须在张量内"检查）✓。
     - 仍需在运行时**校验布局假设**：state 张量 dim0 是否等于 `capacity`（即 `shape()[0] == capacity`）。
       成立才按 `covered_rows = history.len()` 分行拷贝；不成立就**不缓存**（`Ok(0)` 回退）——
       绝不假设布局（这是本项目反复吃过的亏）。
     - **快照张量的分配方式（Round 56 查清）**：`api::zeros` 返回的是 **DeviceOp**（必须在图/Builder 内执行），
       **不能**当独立分配用。独立分配用 `CudaDevice::upload(vec![0.0f32; elements], &[elements])`
       （`device.rs:192`）→ `Arc<Tensor<f32>>`，再用 `Arc::try_unwrap`（新建时引用计数为 1）拿到拥有所有权的
       `Tensor<f32>` 存入 `CachedPrefix.states`；写入前需要 `&mut Tensor`，用 `Arc::get_mut(..)`
       （或先 `try_unwrap` 再 `copy_d2d_at(&mut tensor, src, 0, src_offset, elements)`）。
       代价：`upload` 会做一次 zeros 的 host→device 拷贝（数 MB/快照，每 256 token 一次，可接受）；
       若日后要更省，可加一个纯 device 分配原语替换它。
- **D. 验收**：同一长 prompt 前缀、两种不同后缀连发 ⇒ 第 2 条 prefill 大幅缩短（对照 5c 的量化表）；
  greedy 逐 token 等价；resident 63/63；服务验收 6 项；`PoolInspection` 里命中数增长。

## 5c. #7 的收益量级：按 prompt 长度算（Round 42）

prefill 是**带宽受限**的：每个 32-token chunk 都要把所有常驻权重读一遍 ⇒
`chunk ≈ 22 GB / 1792 GB/s = 12.3 ms`，即 **≈ 0.38 ms/token**。
prefix cache 命中的部分可跳过这些 chunk：

| prompt tokens | chunks | prefill 权重读取耗时 |
|---|---|---|
| 137 | 5 | 61 ms |
| 256 | 8 | 98 ms |
| 512 | 16 | 196 ms |
| 1024 | 32 | 393 ms |
| 2048 | 64 | 786 ms |
| 4096 | 128 | 1571 ms |

⇒ 收益判据：
- **< ~200 token**：收益在几十 ms 量级，不值得为它增加复杂度（相对 110 ms 稳态 TTFT 是噪声级）。
- **1k token 级**：省 ~0.4 s / 请求 —— 对"长系统提示 + 多轮"的服务形态很可观。
- **4k token 级**：省 ~1.5 s / 请求 —— 决定性。

所以 #7 的优先级**取决于目标流量**：短提示为主 ⇒ 优先级低（#5 融合的 3–4.5% 甚至更稳定）；
长共享前缀为主 ⇒ 优先级高。这也是把它与 #5 并列而非排前者之一的原因。

## 5d. CUDA 后端未实现的默认 trait 方法审计（Round 43）

把 Round 36 的教训（"默认实现会静默降级"）系统化应用到**后端侧**：`BackendProvider` 共 35 个带默认实现的
方法，CUDA 有 **22 个**未实现。与 metal 对比后分类：

| 方法 | CUDA | metal | 性质 |
|---|---|---|---|
| `completion_timing` | ✗ | ✓ | **静默降级**：引擎成本模型拿不到真实耗时（`completion/mod.rs:111` 用它喂调度）。**无捷径**（Round 44 查证）：`TimingSource` 只有 `CpuWall/MetalGpu/CudaGpu`，而 `matches_backend` 对 CUDA **只接受 `CudaGpu`** ⇒ 不能用墙钟冒充；且 CUDA 侧当前**零** elapsed 数据。实现路径（~30–50 行）：`device.rs` 加一对常驻 CUDA event（`execute` 是 `&mut self` 且同步 ⇒ 至多一个 in-flight ⇒ 事件可常驻，无需 per-ticket 生命周期），`execute` 前后 record、返回后 `cuEventElapsedTime`；把 `elapsed_us: Option<u64>` 记到 `CudaTicket`（`mod.rs:69`）上，`provider.rs` 的 `completion_timing` 直接返回 `Some(ExecutionTiming { elapsed_us, source: CudaGpu })`。每步开销 ~2–5 µs。**先确认 `cuda_core::sys` 的 event 符号名**（`cuEventCreate/Record/ElapsedTime/Destroy`）再动手，避免编译期返工 |
| `supports_recompute_preemption` | ✗ | ✓ | 能力缺口：引擎无法抢占-重算，内存紧张时只能拒绝（正是 Round 13/21 突发失败的机制）；已由 #3 排队重试缓解 |
| `kv_cache` / `state_page_growth` | ✗ | ✓ | 页级记账未接入（当前用字节级准入，属设计选择而非缺陷） |
| `reuse_prefix*` / `reusable_prefix*` | ✗ | ✓ | 特性缺口 = [#7](#5b-附7-prefix-cache-的移植计划round-40-勘察) |
| `set_waker` / `resource_epoch` / `pending_resource_releases` / `requires_async_checkpoint` | ✗ | 部分 | 异步/发布追踪未接（包装层已能转发 ✓ 后端未实现 ⇒ 无行为变化） |
| `supports_control_checkpoint` | ✗ | ? | 控制面检查点能力未接 |
| `capture_execution_state` / `restore_execution_state` / `begin_resource` / `recycle_*` / `launch_accepted` / `submit_shared*` | 0 匹配 | — | **多数是审计假阳性**（多行签名/其它命名）；`submit_shared*` 属正常默认转发 |

**结论**：真正值得做的 CUDA 后端补全有三项——`completion_timing`（让调度器拿到真实 GPU 耗时，中小改动）、
`reuse_prefix*`（#7，中大）、`supports_recompute_preemption`（大）。其余或为设计选择、或为假阳性。

**审计方法**（可复用）：从 `foundation/spi/src/backend.rs` 提取带默认实现的方法名，检查各后端源码是否出现
`fn <name>`；对差异项逐个判断"默认值是否静默降级"。

## 5e. ⚠ 长 prompt 的 prefill 远低于带宽上限（Round 58 发现；数字已被 Round 62/85 修正）

> **阅读提示**：本节 Round 58 的原始数字（5.4 ms/token、"14 倍"、prefix cache 省 5.9 s、36 ms/chunk）
> **已被后续测量修正** ✗：正确值是 **prefill ≈ 1.8 ms/token**（带宽下限的 4.7 倍）、每 chunk
> **~44–50 ms**（Round 85 无混淆证据）、prefix cache 复用 1k 前缀 ≈ **1.8 s**。
> 下文保留原始记录以说明诊断过程；**以 §5g 的约束表为准** ✓

用 1096-token prompt 跑一次单请求（`max-num-batched-tokens 32`，即 prefill 宽度 32）：

| prompt tokens | TTFT | 折算 |
|---|---|---|
| 137 | 1.34 s（含冷启动 1.33 s） | 与带宽模型一致 ✓ |
| **1096** | **5.88 s** | **5.4 ms/token ≈ 带宽模型（0.38 ms/token）的 14 倍** ✗ |

（另一组：开快照 6015 ms / 关快照 5882 ms ⇒ 4 次快照仅 ~133 ms，**快照不是原因**；此测量同时证明了
前缀快照的开销约为 33 ms/次，可接受。）

**含义（这是本会话最有价值的发现）**：
1. 长 prompt 场景下 **prefill 效率是数量级问题**，比融合（3–4.5%）与短前缀缓存都重要得多——"高性能推理
   服务器"的 TTFT 在长 prompt 上目前被 14 倍地浪费掉。
2. §5c 的收益表（1k token 省 ~0.4 s）**偏小 14 倍**：按实测，复用 1k token 前缀可省 **~5.9 s** ⇒ prefix cache
   的价值远高于此前估算（对长提示流量）。
3. 与 decode 对比：decode 17.5 ms/token 达 datasheet 带宽 70% ✓ 正常；**prefill 与 decode 差了 300 倍**
   （5.4 vs 0.0175 s/token）⇒ 极可能是分块/派发问题，而不是带宽问题。

**已定位（Round 59 实测）**：1137-token prompt 下 **`steps = 37`** ⇒ **每个 prefill step 确实处理 32 token**
（token 预算生效，**不是**退化成 1 token/step）。用两个数据点线性拟合：

```
TTFT(5 chunks)  = 1340 ms = cold(~985) + 5×K
TTFT(36 chunks) = 3557 ms          ⇒ K = (3557-1340)/31 ≈ 71 ms / 32-token chunk
```

即 **每个 chunk 71 ms，而权重读取下限只有 12.3 ms ⇒ 5.8 倍差距**，且它与 prompt 长度无关（固定 per-chunk
开销）。

**候选原因（按可能性排序，下一步逐一排除）**：
1. **每 chunk 的同步/host 往返**：1155 次派发若夹杂 per-layer 同步（64 层 × ~0.5 ms ≈ 32 ms）或每步一次
   `sync_on`，量级吻合。
2. **每 chunk 的 logits/hidden 读回**：32×150k×4 B = 19 MB ⇒ ~10.6 ms + 一次同步（若**每个** chunk 都读，
   而不只是最后一个 chunk，则白花）。检查 `Prefill { span, readout }` 里每个 chunk 的 `readout`。
3. **host 侧每 op 的准备开销**：1155 op × ~50 µs ≈ 58 ms（若 lane 绑定/元数据准备在 host 上逐 op 做）。
4. 权重读取（12.3 ms）与激活流量（<1 ms）都不是主因。

**判别结果（Round 60 实测，1 Hz 采样）**：

| 阶段 | GPU 利用率 | 显存控制器 | 功耗 |
|---|---|---|---|
| 权重加载 | 20–21% | 2% | 88 W |
| **prefill + decode（3.5 s 窗口）** | **99%** | **18–24%** | 190–317 W |
| 对照：decode 基线 | 88% | **60–68%** | 360 W |

⇒ **prefill 期间 GPU 满载但显存控制器只有 ~20%** ⇒ **host 侧同步/往返（候选 1/3）被否证**；瓶颈在
**kernel 侧的计算效率**（候选 2/4 或 kernel 结构）。

**推算冗余量**：3.5 s × 约 30–50 TFLOP/s ≈ 100–175 TFLOP，而 1137-token prefill 的"有用"算力只有
`2 × 5.5e9 × 1137 ≈ 12.5 TFLOP` ⇒ **做了 ~10 倍的多余计算**。结合"每 chunk 71 ms 而权重读取只需 12.3 ms"，
最可能的结构性原因是：**prefill 复用了为单 lane 设计的 kernel/分块，32 个 token 没有真正并行起来**
（例如每个 lane 各自处理全部 32 个 token，或 tiling 浪费了 31/32 的行）。

**长度阶梯实测（Round 61，单进程 4 条请求，各生成 1 token）**：

| prompt | chunks | TTFT |
|---|---|---|
| 4 | 1 | 2998 ms |
| 36 | 2 | 3101 ms |
| 68 | 3 | 3260 ms |
| 100 | 4 | 3474 ms |

⇒ 边际 chunk 成本 **103 / 159 / 214 ms（随位置递增）**，即 prefill 里存在**位置相关**的分量
（注意力扫描/append 随已覆盖长度增长，或分块边界效应）。

**已澄清（Round 62，全新进程阶梯，冷启动恒定）**：

| tokens | chunks | TTFT | 边际 |
|---|---|---|---|
| 4 | 1 | 1259.7 | — |
| 36 | 2 | 1312.4 | 52.7 |
| 68 | 3 | 1381.8 | 69.4 |
| 100 | 4 | 1459.6 | 77.8 |
| 132 | 5 | 1476.8 | 17.2 |
| 164 | 6 | 1537.7 | 60.9 |
| 196 | 7 | 1603.3 | 65.6 |

⇒ **边际 ≈ 57 ms / 32-token chunk，基本恒定**（不是位置相关）；拟合 `TTFT = 1260 + 57×(chunks-1)`，
外推到 1137 token（36 chunk）= **3255 ms，与实测 3557 ms 相差 8%** ✓ 模型自洽。
⇒ Round 61 的两个疑点因此都解决：**(1)** 固定成本 1260 ms（全新进程）/ 1340 ms（服务首请求）本就一致，
Round 61 的 2998 ms 是**同进程多请求下池/图复用被扰动**的结果；**(2)** "边际递增"是同上的池污染伪象，
不存在 O(n²) 项 ✓。

**因此修正两个量（之前写错过，以此为准）**：
- prefill 实际 = **57 ms/32-token chunk = 1.8 ms/token**，是带宽下限（0.38 ms/token）的 **4.7 倍**；
- 每个 chunk 里 **45 ms 不是权重读取**（权重读只需 12.3 ms）⇒ 这是要打掉的目标。
- 相应地，prefix cache 复用 1k token 前缀的收益 ≈ **1.8 s**（不是我 Round 58 说的 5.9 s；那个数字来自被污染的
  固定成本）。
- 与 decode 对比：chunk（32 token）57 ms vs decode step（1 token）17.5 ms ⇒ prefill 每 token 效率是 decode 的
  **4.8 倍**；若 32 lane 真并行应达 ~32 倍 ⇒ **实际只有约 7 倍的有效并行度**。

**原疑点记录（保留作方法论警示）**：
1. **固定成本 2998 ms** 与早先服务测量的 1340 ms（Round 41，热服务器首请求）**不一致** ⇒ 一次性的
   `run` 冷路径比服务首请求多约 1.6 s，原因未查明（可能与该配置的 capacity/池条目数有关，不是 prefill）。
2. 把"chunk 成本随位置线性增长"外推到 1137 token 会得到 38 s ✗，与实测 3.5 s **矛盾** ⇒ 该递增**不能**
   简单外推（可能只在最短长度区间成立，或受池/图复用影响）。**在解释清楚这两点之前，不要据此改 kernel。**

**Round 63 的两项证据（一条否证、一条定位）**：
1. **否证"prefill 用了 FP32 SIMT"**：`resident/prefill_projection.rs` 显示 32-lane 线性层按权重类型分派
   （`ProjectionWeight::Fp4 → gemm::nvfp4`）⇒ **与 decode 用同一套量化张量核 kernel** ✗ 不是 dtype/分派 bug。
2. **否证"长上下文注意力是主因"**：1137 上下文下 TPOT = **18.07 ms**（137 上下文 17.49 ms，仅 +3%）⇒ 注意力
   与 KV 读取在长上下文下依然很便宜（KV 149 MB ≈ 83 µs）。

**因此每 chunk ~50 ms 的特征是**：与 **prompt 位置无关**（阶梯边际基本恒定）、与 **capacity 只有弱相关**
（capacity 64→256（4×）时 chunk 成本 52.7→65.6 ms，仅 +25%）、与 **dtype/kernel 选择无关** ⇒ 它正比于
**chunk 的 32 行本身**，即 32 行 kernel 的**有效并行度只有约 7×（而非 32×）**，或存在**每 chunk 固定的一次性
GPU 工作**（GPU 99% 忙、显存控制器 20% ⇒ 是真实计算，不是 host 侧）。

**Round 64 判别实验（固定 196 token，改变 chunk 宽度，无需改代码）**：

| chunk 宽度 | chunks | TTFT | 每 chunk | 每 token |
|---|---|---|---|---|
| 32 | 7 | 1587 ms | **46.7 ms** | 1.67 ms |
| 3 | 66 | 3705 ms | **37.1 ms** | 12.5 ms |

（每 chunk 已扣除 1260 ms 冷启动。）

⇒ **总 prefill 之比 7.5×**，接近 chunk 数之比 9.4×，远非 1.0× ⇒ **prefill 由"每 step 的固定开销"主导**。
3 token/step 与 32 token/step 的每 step 成本只差 **9.6 ms**（≈0.33 ms/token ≈ 权重读取下限 ✓）。

**因此模型修正为：`prefill_step ≈ 36 ms 固定 + 0.33 ms/token`**，且：

- **固定项 ~36 ms 是 decode step 固定项（~5 ms）的约 7 倍**（decode step 总 17.5 ms，其中权重 12.3 ms）；
- 它**与 token 数无关**（32 vs 3 token 只差 9.6 ms），也**与 capacity/位置无关**（Round 62/63）；
- GPU 采样显示 prefill 期间显存控制器仅 ~20% ⇒ 这 36 ms **不是显存流量**（权重只需 12.3 ms）⇒ 是**每 step 的计算/同步开销**。

**最可能的原因（按可能性）**：
1. **prefill 路径每 step 有 ~20 ms 的额外固定开销**（1155 个 kernel ⇒ ~17 µs/kernel，远高于图重放的典型值 ~1–4 µs）
   ⇒ 怀疑 prompt 图**没有被重放而是每 step 重新录制**（host 侧 ~20 ms）或存在**每 step 一次的全设备同步/读回**。
2. 若属实，修法是让 prefill 走与 decode 相同的"一次捕获、多次重放"路径 ⇒ **长 prompt 可省 ~3–7 倍**。

**Round 65 代码定位（读到 prefill 的实际调用链）**：

```
Sequence::run (executor/execution.rs:159)         // 逐 chunk 的 while 循环
  └─ program.prefill_batch_readout(tokens[chunk], position, read_logits, read_hidden)
       └─ prompt_batch.run(device, tokens, position, Some(read_logits), read_hidden)   // resident/program.rs:506
            └─ if width == PREFILL_LANES { self.run32(...) }                            // batch.rs:402
```

⇒ 确实是**已捕获图的执行**（不是每 step 重新录制）✓，所以 36 ms 在 `run32` 内部。已看到非 32-lane 的
`run` 分支每步会**逐 lane 构建 readback 源**（`ReadbackSource::whole(hidden[lane])`、`whole(logits[lane])`）
并调用 `self.readbacks.run(...)`（batch.rs:421–428）。

**最强假设（能同时解释所有观察）**：**prefill 每个 chunk 都做一次"提交 + 全设备同步 + 读回"**，
而不是只有最后一个 chunk 才读回：
- 每步同步（等 GPU 干完）⇒ 无法流水，GPU 在两次同步之间忙 ⇒ 1 Hz 采样看到 99% 忙 ✓
- 读回量按 **lane 数**算（32 lane × logits 600 KB ≈ 19 MB ≈ 10.6 ms）✓
- 因此与 **step 里的 token 数几乎无关**（3-lane 的 66 步只比 32-lane 的 7 步便宜一点：37 vs 47 ms）✓✓
- 与 capacity/位置无关 ✓

**据此的修法（若确认）**：**中间 chunk 不需要任何读回**（只需推进 KV）⇒ 让 prefill 变成"发完再同步"，
只在最后一个 chunk 读 logits/hidden ⇒ 长 prompt 可省 **3–7 倍**。

**Round 66 验证结果：假设被否证** ✗（读 `BatchGraph::run32`，batch.rs:491）

```rust
let sources = [
    read_hidden.then(|| ReadbackSource { tensor: hidden[0], skip: 0, len: tokens.len()*hidden_dim }),
    (prefill == Some(true)).then(|| ReadbackSource { tensor: logits[0],
        skip: (tokens.len()-1)*vocabulary, len: vocabulary }),
];
```

- logits 读回**只在 `prefill == Some(true)`**（引擎只在最后一个 chunk 传 `read_logits = last && ...`）⇒
  **中间 chunk 不读 logits** ✓
- hidden 读回只在 `read_hidden`（Generate + MTP 关闭 ⇒ false）⇒ **中间 chunk 不读 hidden** ✓
- 因此"每 chunk 同步 + 19 MB 读回"的假设**不成立** ✗

⇒ 那 36 ms 就是 **chunk 的 GPU 执行时间本身**（与 Round 60 的"GPU 99% 忙、显存控制器 20%"一致：是 GPU 侧
计算，不是 host 侧、不是读回）。

**剩下的唯一解释空间**：chunk 内的 **GPU 计算效率**。已知：32-lane 线性层走 `prefill_projection` →
`gemm::nvfp4`（张量核 ✓）。量级估算：32 token × 2 × 22e9 ≈ **1.4 TFLOP/chunk**；若 M=32 的 tile 只跑到
~40 TFLOP/s ⇒ **35 ms** ✓ 与实测 36 ms 吻合。但 3-token chunk 只需 0.13 TFLOP 却仍要 37 ms ✗
（其中权重读取 12.3 ms）⇒ 仍有 ~25 ms 无法用"token 数"解释 ⇒ **必须进 graph 内部计时**才能定位。

**Round 67：用"功耗 + 并发"两条已有数据收敛到"延迟受限（低占用）"**

证据 A（Round 60 采样）：

| 阶段 | GPU 利用率 | 显存控制器 | **功耗** |
|---|---|---|---|
| prefill | **99%** | ~20% | **190–317 W** |
| decode | 88% | 60–68% | **360 W** |

⇒ GPU 利用率 99% 却**功耗低于 decode** ⇒ SM 显示"忙"但在**停顿**（不是在做带宽或 FP 吞吐工作）
⇒ 典型是**访存延迟受限 / 占用率过低**的 kernel。

证据 B（Round 58 的 2 并发长 prompt 请求，各 1096 token）：两条请求**各 6.0 s**，而单条独跑是 **3.5 s**
⇒ 2 倍工作量只花了 1.7 倍时间（部分重叠）⇒ **也符合延迟受限**（停顿可被并发填充；若是纯吞吐受限，
两者应各自保持不变或显著劣化）。

**这解释了此前所有"无法用 token 数解释"的现象**：延迟受限的 kernel，其耗时由**串行依赖链**决定，
而不是由一次处理多少 token 决定 ⇒ 3 token 与 32 token 的 chunk 耗时相近（37 vs 47 ms）✓✓

**据此的修法方向（顺序）**：
1. **提高 prefill kernel 的占用率**（Round 68 已算出具体数字）：`resident/prefill_gemm.rs` 的 `nvfp4` kernel
   用输入 tile `[32,64]`、权重 tile `[64,64]`、scale `[64,4]`，K 维串行循环；`PREFILL_GEMM_TILE_N = 64`
   ⇒ 网格 = `(1, N/64)`。以 N=13824 的层为例：**216 个 CTA vs 170 个 SM ⇒ 每 SM ~1.3 个 CTA**
   ⇒ 若每 CTA 4 warp 即约 **8% 占用率** ✗ ⇒ 与"99% 忙、317 W（低于 decode 360 W）"完全吻合的**延迟受限** ✓
   **改法**：把 N tile 从 64 缩到 32（或 16）⇒ 网格 CTA 数 ×2–4（432/864 ⇒ 2.5–5 CTA/SM）⇒ 并行度上去，
   **而权重总流量不变**（每个权重元素仍只读一次 ✓）⇒ 是"无额外显存代价"的收益 ✓✓
   **实测结果（Round 69）：假设被否证** ✗ 把 NVFP4 输出 tile 从 64 降到 32（仅动 FP4 分支，新增
   `PREFILL_NVFP4_TILE_N`），196-token 阶梯：**每 chunk 46.7 → 132.1 ms（慢 2.8 倍）**，3-lane 路径基本不变
   （37.1 → 33.2 ms，噪声）；**63/63 数值仍然通过**（说明改动本身是正确的，只是更慢）。
   ⇒ **prompt GEMM 受限于 M=32 下张量核的 MMA 形状效率，而不是 CTA 占用率**：更窄的 N 让每条 MMA 更小、
   指令数翻倍 ⇒ 更慢。**已回退到 64**（并把结论写进 `PREFILL_NVFP4_TILE_N` 的文档注释，防止再犯）。
   **反方向实验（Round 70）：加宽到 128 编译失败** ✗（不是数值问题，是 cuTile 类型约束）：
   ```
   error[E0308] at prefill_gemm.rs:73/76
     expected Tile_2<_, 128, _>, found Tile_2<_, 64, _>
   ```
   即 `unpack(shape![128,64])` / scale broadcast 链的类型推导不能通过 ⇒ **加宽不是改常量的活**，需要重构
   （例如用**两个 `[32,64]` 累加器**手工组成 128 宽的输出，或改 fp4 解包方式）。
   （本轮实验脚本带**自动回退**：编译失败即恢复备份并重建 ⇒ 树始终绿色 ✓ 上一轮也是测得变慢后立即回退 ✓）

   **因此"改 tile 尺寸"这条线到此收束**：64 是当前 cuTile 表达下已验证的好值；32 慢 2.8 倍、128 编不过。
   下一轮若要继续 prefill 优化，应换赛道（见下）。

   **Round 71：把"每 chunk 固定开销"的解释收敛到串行依赖链**

   读 `device/readback.rs::run`（每个 step 都走它）：
   ```rust
   self.prepare(device, sources)?;                       // prefill 中间 chunk 的 sources 全为 None
   device.stream.device().bind_to_thread()?;             // 上下文绑定（每步，~µs 级）
   graph.launch().then(copies).sync_on(&device.stream)?; // 发射 + **全设备同步**
   ```
   ⇒ **每 chunk 一次全设备同步** ⇒ 实测的 wall time **就是 chunk 的 GPU 执行时间**（不是 host 侧开销 ✓）。

   于是三条实测事实合起来只剩一个解释：
   | 事实 | 出处 |
   |---|---|
   | 每 chunk ~37–47 ms，**与 token 数无关**（3 vs 32 token 只差 9.6 ms） | §5e / Round 64 |
   | 与 capacity、prompt 位置无关 | Round 62/63 |
   | 是 GPU 执行时间（99% 忙、功耗低于 decode、显存控制器 20%） | Round 60/71 |

   ⇒ ~~**chunk 时间 ≈ 1155 个相互依赖的 kernel 的串行延迟链**：36 ms / 1155 ≈ **31 µs/kernel**~~
   **已被 Round 81 否证** ✗：删掉 64 个 kernel 测不到收益 ⇒ 31 µs **不是均匀分布**的 ⇒ 36 ms 集中在
   少数重 kernel 上（下文第 1 条据此高估了融合收益，同样作废 ✗）
   （图内相邻 kernel 的典型间隔是 2–5 µs）⇒ 这些 kernel 是**延迟受限**（占用率低、每级依赖等一次访存），
   而不是吞吐受限 ✓ 这与"功耗只有 317 W（< decode 360 W）"一致 ✓

   **两个（曾以为可用的）杠杆—— 第 1 条已被否证 ✗，仅第 2 条仍然成立**：
   1. **减少派发数与依赖深度（融合）**：Round 33 量到 1155 个 op、其中 norm 161 / add 128 / mul 80 / silu 64
      ⇒ 融合这 ~256 个可获得 ~22% 的 per-chunk 时间（**比我此前按 2–3 µs/launch 估的 3–4.5% 更值**，
      因为真实 per-kernel 代价是 ~31 µs 而非 2–3 µs）~~**✗ 已否证（Round 81 实测无收益）**
   2. **并发填充**：延迟受限 ⇒ 多条序列的 chunk 交错执行能填满停顿（跨序列批处理，步骤 A）✓

   **换赛道的候选**（都基于已确认的事实：M=32 下张量核形状效率受限、GPU 利用率高但功耗低于 decode）：
   1. **提高 M**：既然 M=32 是效率瓶颈，把 32 行拆成 2×16 行或 4×8 行会更差 ✗；反过来**合并多个 step**
      （例如把相邻 chunk 合并成 64 行）⇒ 需要 KV 位置连续且引擎允许更大 chunk ⇒ 与 §5c 的 chunk 收益直接相关。
   2. **并发填充**：既然停顿占主导，跨序列批处理（计划步骤 A）能填满停顿 ⇒ 对 prefill 的收益比此前估计大。
   3. **in-graph 计时**（CUDA event）确认瓶颈 kernel 后再定方向（Round 44 已勘察实现路径）。
   （注意：这与 Round 2 已证伪的"2-D 累加器"是不同问题——那条路卡在 cuTile 表达力，这条只是 tile 尺寸。
   也要注意：Round 2 的 `BN=16/BK=128` 实验针对的是 **decode** 的 `linear_batch`，与本 kernel 无关。）
2. **并发前置**：既然停顿可被其它序列填充，跨序列批处理（计划步骤 A）对 prefill 的收益会比此前估计更大
   ⇒ 与"延迟受限"直接对应。
3. 验证手段：改 tile 后重跑 §5e 的阶梯（196 token、32 vs 3 lane）与 1137 token 单请求，看每 chunk 是否
   从 ~46 ms 降到接近权重下限 12.3 ms。

**（可选）更精细的定位**：CUDA event 逐 kernel 计时（Round 44 已勘察）——若 (1) 的第一次尝试没有效果，
再上这个手段。**在此之前不要再猜**（本会话已否证 3 个假设）。

**原下一步（保留）**：CUDA event 逐 kernel 计时（Round 44 已勘察：两个常驻 event +
`cuEventElapsedTime`，~30–50 行），先给"每 chunk 的 1155 个 kernel"做**分段计时**（例如按 op 类型分组），
找出这 25–36 ms 落在哪一类 kernel 上。在此之前**不要再猜**（本会话已连续否证 3 个假设：快照整拷、FP32 SIMT、
每 chunk 读回）。

**下一步候选（按性价比）**：
- (a) 把 32 行 GEMM 的**权重流量**与理论值对比：理论每 chunk 只读一遍权重（12.3 ms）。若实际是它的 4–5 倍，
  说明 **K/N 维分块导致权重被重复读取**（或 8 个投影各自按 N 分块时重复读 K 维）⇒ 查
  `PREFILL_GEMM_TILE_N` 与 `prefill_gemm::gemm` 的 tiling 是否对 32 行最优。
- (b) 用 CUDA event 给"每个 chunk"打点（`completion_timing` 基础设施），先确认 45 ms 落在 GEMM 还是
  attention/norm 类（`Dispatch32::Row` 逐 lane 记录的那些 op）上。
- (c) 直接做**消融**：把 `PREFILL_LANES` 从 32 降到 16（编译期常量），若 chunk 成本减半 ⇒ 说明成本正比于 lane
  数（并行度不足）；若几乎不变 ⇒ 说明是每 chunk 固定项。**这是最省事、判别力最强的一步。**

**原下一步（保留）**：比较"32-token prompt 的一次 prefill chunk"与"一次 decode step"的耗时：
- 若 chunk ≈ decode step（~17 ms）⇒ 分块/派发没并行，问题在 lane 化；修法要动 prefill 的 kernel 选择。
- 若 chunk ≫ decode step ⇒ 逐 kernel 加 CUDA event 计时（`completion_timing` 的基础设施），定位到具体 kernel。

也可先用 op 直方图（provider check）对比 prefill 与 decode 的 kernel 构成，看 prefill 是否触发了不同的 kernel 路径。

**历史下一步（保留；数字见本节开头的修正提示）**：
- 先量"每个 prefill step 处理多少 token"（`max_num_batched_tokens=32` 是否真的按 32 走，还是退化成
  1–3 token/step）；用 events（`mtp-ab.py` 的 steps 计数）或 op 直方图对比 137 vs 1096 token 的步数。
- 若确实按 32/step，则 32-token chunk 耗时 ~172 ms（远超权重读取的 12.3 ms）⇒ 查 prompt 图是否被使用、
  是否每 chunk 都做了一次多余的同步/读回/arena 重建。
- 这个方向若修好，长 prompt TTFT 应回到 ~0.5 s 量级。

## 5f. 融合的第一刀：silu+mul（方案已精确到行，Round 72 勘察）

**关键事实**：融合所需内核**已存在**，不需要新 IR 算子、不需要编译器发射、不需要写新 kernel：
- `crates/backend/cuda/src/mlp/kernels.rs:27` `fn silu_mul<const B: i32>(...)` ✓
- `crates/backend/cuda/src/mlp/pdl.rs:17` `fn silu_mul(...)`（PDL 变体）✓
- 当前捕获把它们**分开记录**：`resident/capture.rs:113`（`TensorOp::Silu | Sigmoid`）与
  `resident/capture.rs:122`（`Add | Multiply`，用泛型参数区分 `Multiply`）✗

**改法（捕获层，约 60–80 行）**：
1. 在节点遍历里识别模式：某 `Silu` 节点的输出**只被一个 `Multiply` 消费**（扫一遍 `graph.nodes` 建
   `fused_silu: BTreeSet<NodeId>` 即可，无需消费者索引）。**融合记录放在 `Multiply` 节点上**——那里两个操作数
   （`gate` = `Silu` 的输入、`up` = `Multiply` 的另一输入）都拿得到，而 `Silu` 节点直接跳过：
   ```rust
   TensorOp::Multiply => {
       if let Some(gate) = self.fused_gate(node)? {   // input[0] 由已被融合的 Silu 产出
           self.scope.record(mlp::kernels::silu_mul(out_partition, gate, &self.input(node.inputs[1])?)
               .generics(vec![AUX_KERNEL_TILE.to_string()]))?;
       } else { /* 原 aux::binary 路径 */ }
   }
   ```
   注意 `silu_mul` 的签名是 `(out, gate, up)`（`mlp/kernels.rs:27`，`gate` 是 silu **之前**的值 ✓ 正好是
   `Silu` 节点的输入 ✓）；`AUX_KERNEL_TILE` 分区与现有 `aux::unary/binary` 一致 ✓。
2. 对该 `(Silu, Multiply)` 对只记录**一次** `silu_mul`（写入 `Multiply` 的输出槽；`Silu` 的中间槽变为未使用，
   由 arena 的活跃性分析自然回收 ✓）。
3. **跳过被融合的 `Silu` 节点**，共需在 **3 个节点循环**里加同一个判断（`if fused_silu.contains(&node.id) { continue; }`）：
   - `resident/batch.rs:243` `capture32`（提示/批量路径）——**主要受益路径**
   - `resident/batch.rs` 的 `dispatch32` 分派循环（验证/小宽度路径）
   - `resident/program.rs` 的平坦图捕获循环（decode 路径）
   三处都通过 `resident/capture.rs` 的记录层写 op ⇒ 融合逻辑只在记录层写一次，跳过判断复制三处（各一行）
   ⇒ 若只想先吃提示路径的收益，可**只改 `capture32` 循环**（decode 路径保持原样，行为完全不变、零风险）✓

**实现形状已确认（Round 76，读码结论）**：

- `Capture::operation(&mut self, node, mut output: Tensor<f32>, part: usize) -> Result<Tensor<f32>>`
  是按值进出的匹配体（`resident/capture.rs:99`），`TensorOp::Multiply` 分支就在其中 ✓
- `capture32` 的循环（`batch.rs:242`）已有 `for node in &self.graph.nodes { ... match dispatch32(node, ..) }`
  结构 ✓，且**能访问 `self.graph`** ✓ ⇒ 融合的两个判断都放这里最省事 ✓
- **选定实现**（最小改动、无需读更多代码）：
  1. `Capture` 加一个字段 `fused_gate: Option<TensorId>`（`resident/capture.rs:21`），在
     `TensorOp::Multiply` 分支里：`Some(gate)` ⇒ 调 `crate::mlp::record_silu_mul(self.scope, &mut output,
     &self.input(gate)?, &self.input(node.inputs[1])?)`；`None` ⇒ 原 `aux::binary` 路径 ✓
     （构造处需补 `fused_gate: None`，多为 `batch.rs` 内的几处 ✓ 机械改动）
  2. `capture32` 循环里：先扫一遍 `self.graph.nodes` 得「被融合的 Silu 集合 + 其 Multiply 的 gate 映射」；
     遇到被融合的 `Silu` ⇒ `continue`；遇到对应 `Multiply` ⇒ 构造 `Capture { fused_gate: Some(gate), .. }` ✓
  3. `mlp/mod.rs` 加 `pub(crate) fn record_silu_mul(scope, output, gate, up)`（8 行）✓
- 安全性：**必须**校验"该 Silu 的输出只被这个 Multiply 消费"（否则跳过 Silu 会让另一个消费者读到未初始化槽
  ⇒ 静默数值错误）✓ 扫描时一并完成 ✓

**Round 80 结论：融合已完整实现且正确，但未测出收益（已回退）** ⚠

Round 80 把三条线全部接通（录制层 + `capture32` 分派 + **竞技场布局感知**），并解掉两个新问题：
1. 布局改动**必须按作用域**：同一个 `DataflowGraph` 被两种竞技场共用（平坦 1-lane 用**未融合**录制 ✓，
   32-lane 提示图用**融合**录制 ✓）⇒ `layout_with(graph, fused)` 只在 `lanes == PREFILL_LANES` 时启用融合，
   `plan` 与 `required_bytes` 用同一条规则 ✓（否则平坦路径读不到 `Silu` 的槽 ⇒ `unknown device activation` ✗）
2. 布局变换本身：删掉被融合的 `Silu` 节点 + 把 `Multiply` 的 `inputs[0]` 换成 gate ⇒ `plan_lifetimes`
   自动把 gate 的活跃区间延长到 Multiply ✓（与融合算子的读写集合一致 ✓）

**验证结果**：

| 检查 | 结果 |
|---|---|
| 编译 / clippy | ✅ |
| `cuda-resident-check` | ✅ **63/63 PASS**（数值未变） |
| 真实提示路径（196 token，32 lane） | ✅ **不再报错**（Round 77 的 arena 冲突已解决） |
| 性能 | ⚠ TTFT **1634 ms vs 基线 1587 ms**（每 chunk 53.5 vs 46.7 ms） |

**Round 81 补测（噪声地板 + 修正判读）**：同一基线连跑 5 次（196 token / 32 lane）：

| rep | TTFT | per-chunk |
|---|---|---|
| 1 | 1625.1 | 52.2 |
| 2 | 1599.2 | 48.5 |
| 3 | 1613.1 | 50.4 |
| 4 | 1617.4 | 51.1 |
| 5 | 1617.7 | 51.1 |

⇒ **均值 1614.5 ms，σ≈10 ms = ±0.8%**；per-chunk 均值 50.7 ms（±2.6%）。
（我此前说的"±3–15% 噪声"是错的——那是把**小差值相除**放大的结果 ✗；直接测 TTFT 的重复性其实很好 ✓）

**因此 Round 80 的判读要更严厉**：融合实测 1634.3 ms vs 基线均值 **1614.5 ms** ⇒ **慢 ~1.2%**，
**不是"分辨不出"，而是"没有收益、略偏慢"** ✗ ⇒ 回退是对的 ✓

**并因此修正 Round 71 的关键推论** ✗：
- Round 71 由"36 ms / 1155 ≈ 31 µs"推出"per-kernel 代价 31 µs ⇒ 融合收益比原估高 10 倍"——
  **这个平均是误导性的**：如果 31 µs 是均匀分布的，去掉 64 个 kernel 应该省 4% ✗ **实测几乎为零** ⇒
  **36 ms 集中在少数"重" kernel（GEMM/attention）上，而 elementwise 小 kernel 极便宜** ✓✓
- ⇒ **融合轻量 elementwise 算子（silu/mul/norm/add）不是杠杆** ✗ "norm+add+silu+mul 一批可拿 20%" **被本轮数据否证** ✗
- ⇒ prefill 的 36 ms 要打，必须打**重的那些 kernel**（prompt GEMM 的 K 维依赖链 / 32-lane attention），
  而不是减少派发数 ✓

**这条线的教训**：`silu+mul` 融合在"派发数减少"上确实成立，但它的**单个改动收益太小**（4%），小于测量
噪声；**要吃到 Round 71 结论里的 20%，必须一次做掉 norm+add+silu+mul 一整批**（否则每个小融合都会被
噪声淹没、无法验收）。

**Round 77 早先实测：代码已写成并编译通过，但运行时报 arena 不变量错误（已回退）** ⚠

实现路径（三步，均已写出并通过编译 + clippy）：
1. `mlp/mod.rs` 加 `pub(crate) fn record_silu_mul(scope, output: &mut Tensor<f32>, gate: &TensorView<f32>,
   up: &TensorView<f32>)`（**注意**：该模块的 `Result` 被 `infer_core::Result` 遮蔽 ⇒ 必须写
   `std::result::Result<(), DeviceError>`；参数必须是 `&TensorView` 才能接 `self.input()` 的返回 ✓）
2. `resident/capture.rs`：`record` → `record_with_gate(node, None)`；新增 `record_fused_gate(node, gate)`；
   `operation(..., fused_gate: Option<TensorId>)` 里在 `Add | Multiply` 分支前置 `if let Some(gate)` 用
   `crate::mlp::record_silu_mul(...)` ✓
3. `resident/batch.rs` 的 `capture32`：`fused_silu_mul_pairs(graph)` 扫出「被融合 Silu 集合 + Multiply→gate 映射」；
   循环里跳过被融合的 `Silu`，对相应 `Multiply` 调 `record_fused_gate` ✓

**结果**：编译 ✓、clippy ✓、**`cuda-resident-check` 63/63 通过**（数值正确 ✓），但真实提示路径（1137 token 阶梯）
运行时报：
```
"code":"Backend","message":"kernel launch error: Invariant: activation is currently an output"
```
⇒ **跳过 `Silu` 节点破坏了 arena 的输出/活跃性状态机** ✗：arena 依赖"每个节点都被记录"来推进激活槽的状态
（`take()` 标记为输出、`get()` 要求不是输出），少记录一个节点后，后续 `self.input(gate)` 读到的是仍处于
"输出中"状态的槽 ✗。

**根因已查明（Round 78，读 `resident/arena.rs:133`）**：

```rust
pub(crate) fn get(&self, id: TensorId) -> Result<&Tensor<f32>> {
    self.buffers[self.slot(id)?].as_ref()
        .ok_or_else(|| Error::invariant("activation is currently an output"))
}
```
`take()` 把槽置 `None`；`get()` 见到 `None` 就报这个错。**真正的原因是活跃性布局，不是状态机调用缺失**：

- arena 的槽复用是按**未融合**图的活跃区间算的：`Silu` 消耗掉它的输入（gate 前值）后，该输入的槽就"死了"，
  于是**可以被 `Multiply` 的输出复用** ✓（布局把两者分到同一个槽）。
- 我跳过了 `Silu` 之后，融合算子在**写 Multiply 输出**的同时还要**读 gate（Silu 的输入）** ✗ ⇒ 两者是同一个槽
  ⇒ 读到的槽已被 `take()` 置空 ⇒ 报错 ✓✓ **完全自洽**。
- 这也解释了为什么 **63/63 通过**：`cuda-resident-check` 走的是平坦/decode 捕获路径（我没在那里启用融合），
  提示路径（32-lane `capture32`）只有真实阶梯运行才覆盖 ✓。

**修法（唯一正确做法）**：让**活跃性布局也感知这个融合** —— `ActivationArena::new` 的布局分析里应用同一个
`fused_silu_mul_pairs` 变换：
1. 布局时把被融合的 `Silu` 视为**不存在**（其输出从不产生 ✓ 与录制一致）；
2. 把该 `Silu` 的**输入（gate）的活跃区间延长到那个 `Multiply` 之后** ✓（否则它会被提前复用 ✗）。
两处都用同一个 pair 扫描器 ⇒ 录制与布局对融合的认知一致 ✓ 这是正确性的关键。

**精确改点（Round 79 读码得出，零未知量）**：`resident/arena.rs:47`

```rust
fn layout(graph: &DataflowGraph) -> Result<ArenaLayout> {
    let mut planned = graph.clone();
    planned.plan_lifetimes()?;        // ← 活跃区间就是在这里算出来的
    ...
}
```

⇒ 只需在 `plan_lifetimes()` **之前**对 `planned` 施加同一变换（约 10 行）：
```rust
let mut planned = graph.clone();
for (silu_out, multiply_out) in fused_pairs(graph) {          // 与录制共用同一个扫描器
    // 1) 去掉 Silu 节点（它的输出不再产生）
    planned.nodes.retain(|n| !n.outputs.contains(&silu_out));
    // 2) 让 Multiply 直接依赖 gate（= Silu 的输入）：把 inputs[0]（Silu 输出）换成 gate
    //    ⇒ plan_lifetimes 自动把 gate 的活跃区间延长到 Multiply 之后 ✓
    for node in planned.nodes.iter_mut().filter(|n| n.outputs.contains(&multiply_out)) {
        for input in node.inputs.iter_mut() {
            if *input == silu_out { *input = gate; }
        }
    }
}
planned.plan_lifetimes()?;
```
**为什么这样就自洽**（关键推理）：
- 布局里 `Silu` 的输出**不存在**（节点被删）⇒ 它不占槽；录制侧也跳过该节点、从不查它的槽 ✓
- `Multiply` 的直接依赖变成 `(gate, up)` ⇒ gate 的活跃区间自动覆盖到 Multiply ✓ 与融合算子的
  `silu_mul(out, gate, up)` 读写集合**完全一致** ✓
- 槽分配按 `TensorId` 查询（`arena.slot(id)` ✓ 非节点索引）⇒ 删节点不影响既有映射 ✓
- `Multiply.outputs`（及其下游）不变 ⇒ 其余布局不受影响 ✓
（次优退路：不跳过 `Silu`，只在 `Multiply` 处融合 ⇒ 多一次派发、收益减半，但零布局风险 ✓）

**（本轮改动已全部按备份回退，树保持绿色 ✓）**

**Round 77 早先记录（已澄清）：`silu_mul` 的参数类型问题实为 `Result` 遮蔽** ✗

```rust
// mlp/kernels.rs:27 —— 参数是**拥有所有权的 Tensor**
fn silu_mul<const B: i32>(out: &mut Tensor<f32,{[B]}>, gate: &Tensor<f32,{[-1]}>, up: &Tensor<f32,{[-1]}>)
// 而 resident/capture.rs 的 self.input(id) 返回 **TensorView<'_, f32>**
```
⇒ 在捕获层直接调 `silu_mul` **编译不过**（`aux::binary` 能接 view 是因为它自己的参数就是 view 友好类型 ✓）。
**解法（下一轮二选一）**：
- (a) 给 `mlp/kernels.rs` 加一个**接受 view 的孪生入口** `silu_mul_view(out, gate: &TensorView<f32>, up: &TensorView<f32>)`
  （复制现有 kernel 体，参数改 view ✓ 与 `aux::binary` 对齐 ✓ 最稳）；
- (b) 或把 `silu_mul` 的参数改成 view 类型（需同时改 MLP 路径的两处调用 ✗ 波及面更大）。
**推荐 (a)**：只新增、不改现有路径，风险最低 ✓（`mlp/mod.rs` 的 `record_silu_mul` 包装则改为调 view 版）。
（本轮三步改动均已按备份回退，树保持绿色 ✓）

**最后一块拼图（Round 75：调用约定 + 可见性障碍，照此写即可）**：

现有融合路径的写法（`mlp/mod.rs:115`，直接照抄）：
```rust
scope.record(
    fused::silu_mul((&mut activated).partition([crate::constants::AUX_KERNEL_TILE]), &gate, &up)
        .generics(vec![crate::constants::AUX_KERNEL_TILE.to_string()]),
)?;
```
- `mlp::kernels` 与 `mlp::pdl` 都是 **私有 mod** ⇒ `resident/*` 够不到 ✗
- **解法**：在 `mlp/mod.rs` 加一个 `pub(crate) fn record_silu_mul(scope: &Scope, output: &mut Tensor<f32>,
  gate: &Tensor<f32>, up: &Tensor<f32>) -> Result<(), DeviceError>`（约 8 行，内部就是上面那段；
  或把 `mlp/pdl.rs:31` 的 `record_product` 提为 `pub(crate)` 并 re-export）✓
- `Capture` 结构里**没有** `graph` 字段 ⇒ 不要在记录器里查消费者关系 ✗；改由**循环**决定：
  `capture32` 遍历前先扫一遍 `self.graph.nodes`（`inputs: Vec<TensorId>`、`outputs: Vec<TensorId>` ✓）
  建出「被融合的 Silu」集合与「哪个 Multiply 用到了它」，然后在循环里：
  - 遇到被融合的 `Silu` ⇒ `continue`
  - 遇到对应的 `Multiply` ⇒ 调 `Capture { ... }.record_silu_mul(node, gate_id)`（新增方法，内部调
    `crate::mlp::record_silu_mul`，写 `node.outputs[0]` 的 arena 槽）
  **不需要给 `Capture` 加字段** ✓

**收益估算（⚠ 已被 Round 81 否证：实测无收益，见 §5g / 交接块 A 条）**：
- 融合 `silu`+`multiply` 去掉 **64 次派发** ⇒ 每 chunk 省 ~2 ms / 36 ms ≈ **5.5%**
- 若再做 `norm`+`add`（161+128 个节点）⇒ 总收益可达 **~20%** per chunk
- 对长 prompt（1137 token = 36 chunk）：5.5% ≈ **0.19 s**，20% ≈ **0.7 s**

**验收**：`cuda-resident-check` 63/63（数值必须完全一致——`silu_mul` 是已存在的内核，理论上逐位相同）；
greedy 逐 token 等价；§5e 阶梯（196 token）每 chunk 从 46.7 ms 下降。

## 5g. prefill 每 chunk ~36–50 ms：black-box 推断已穷尽（Round 82 收口，Round 85 加固）

**Round 85 的无混淆测量（关键）**：四条 prompt 全部落在**同一 capacity 桶（256）**，比较"同 chunk 数、不同 token 数"
与"不同 chunk 数"：

| tokens | chunks | prefill |
|---|---|---|
| **130** | **5** | **223.6 ms** |
| **158** | **5** | **218.1 ms** |
| 190 | 6 | 308.9 ms |
| 222 | 7 | 324.7 ms |

⇒ **同为 5 chunks 时，多 28 个 token 的 prefill 时间完全相同（差 5.5 ms，在噪声内）** ✓✓
⇒ **成本由 chunk 数决定，不随 chunk 内 token 数变化** ✓（这**消除了** Round 64 的混淆：那次用 3-lane 对比，
窄宽度可能走不同 kernel ✗；本轮同宽度同 capacity，结论可靠 ✓）
⇒ 每 chunk ≈ **44–50 ms**，5 chunk 的固定部分 ≈ 220 ms



**必须同时满足的四条约束**（全部实测，任一新假设必须让它们同时成立）：

| # | 约束 | 出处 |
|---|---|---|
| C1 | 每 chunk ≈ **44–50 ms**，与 chunk 内 token 数**无关**（**无混淆证据**，Round 85） | Round 64/85 |
| C2 | 与 **capacity / prompt 位置**无关（注意力随位置增长 ⇒ 注意力不是主项） | Round 62/63 |
| C3 | **GPU 99% 忙、显存控制器 ~20%、功耗 317 W < decode 360 W** ⇒ 真实计算，非带宽、非 host | Round 60/71 |
| C4 | 去掉 **64 个 elementwise 派发**（silu+mul 融合）**测不到收益** ⇒ 36 ms **不是**均匀分散在 1155 个 kernel 上 | Round 80/81 |

**已排除的假设**（含否证方式）：带宽受限 ✗(C3)、host 侧同步/读回 ✗(Round 66/71)、注意力 ✗(C2)、
dtype/FP32 SIMT ✗(prefill_projection 用 nvfp4)、CTA 占用率 ✗(tile 32 更慢)、派发数 ✗(C4)。

**由 C1 与 C4 得到的唯一自洽图景**：chunk 里有**一个（或少数几个）与 chunk 内 token 数弱相关的重项**，
它既不是带宽也不是注意力。最可能落在 **prompt GEMM 的 K 维串行循环**（K 固定 ⇒ 与 chunk 内 token 数弱相关 ✓
与位置无关 ✓ 计算型 ✓ 且不因删 elementwise 而改变 ✓）——即 **M=32 下每条 K 迭代的延迟没有被隐藏**：
每 chunk `K/64` 次串行迭代 × 每层 × 64 层；若单次迭代暴露约数百 ns 的访存延迟，累加即为数十 ms 量级 ✓

**结论：black-box 推断到此为止，下一步必须上 instrument** ✓
唯一能定案的手段：**CUDA event 计时段**（在 chunk 内按阶段/按 kernel 分组打点），实现路径见
[Round 44 勘察](#)（两个常驻 event + `cuEventElapsedTime`，~30–50 行）。
在拿到"哪一类 kernel 占了多少 ms"之前，**不要再提出新的猜测**（本会话已否证 7 个）。

## 5h. instrument 定案：每 chunk 的 ms 落在 prompt GEMM（Round 86–88）

**Round 86：图内 CUDA event 分段计时（永久设施，env 门控）**。
`INFER_CUDA_PREFILL_PROFILE=<jsonl>` 打开：`resident/profile.rs` 在 `capture32` 给每个 node 记
boundary event，`run32` 在读回后按 node 聚合，每 chunk 追加一行 JSON
（`graph`/`tokens`/`position`/`total_ms`/`by_op`/逐段 `segments`）。关闭时零开销。
关键坑：capture 里裸 `cuEventRecord` 产生的是**非 external** 节点，回放后 `cuEventElapsedTime` 报
`invalid argument`；必须 `cuEventRecordWithFlags(..., CU_EVENT_RECORD_EXTERNAL)`。

**测量（196 token / 32-token chunk，profiling 开时 57.3 ms/chunk，墙钟关时 46–47 ms）**：

| op | ms/chunk | raw % | 校正 event 开销后 |
|---|---|---|---|
| **linear（prompt GEMM）** | **39.0** | **68%** | **70–83%** |
| delta（线性注意力） | 7.6 | 13% | 14% |
| attention | 4.2 | 7% | 8% |
| conv | 2.4 | 4% | 4% |
| elementwise 合计（silu/mul/add/norm/…） | ~4.6 | ~8% | **<2%** |

（校正按 event 自身开销 p5=4.1µs × 1154 boundaries 摊回；`prefill_last` 图的 linear 占比 80–83%，更高。）
⇒ §5g 的推断**定案**：每 chunk 的 2/3–4/5 是 prompt GEMM ✓；C4 再确认——elementwise 全部融合也
拿不到 2% ✓。
- 失真量化：TTFT 关 1592 ms vs 开 1683 ms（+91 ms，主要是 host 端 2310 次 `elapsed` 调用，GPU 侧
  event 本身 ~4µs/boundary）。

**机理定案（延迟受限，不是带宽受限）**：每 chunk 496 个 linear 节点、均值 79µs；最大 GEMM
（[13824×5120] fp4，权重 ~35 MB）一段 0.22 ms ⇒ **160–560 GB/s ≈ DRAM 峰值的 10–30%**
（decode GEMV 同权重能到 ~70%）。M=32 时 grid 只有 N/64 ≈ 216 CTA / 170 SM ≈ 1.27 波，
K 循环（K/64 = 80 次串行迭代）暴露的 DRAM 延迟没有第二波 CTA 可 hiding——C1–C4 全部同时成立。

**Round 87：cuTile `load_pipelined<4>` hint——无收益，已回退**。nvfp4 kernel 三个 load 换流水线 hint：
图内 linear −3%（39.0→37.9 ms/chunk），墙钟热态 1612/1632 ms vs 基线 1592 ms（噪声内）。
瓶颈是"延迟 × 串行依赖"，编译器 hint 改不动 ⇒ 源码恢复，只留测量记录。

**Round 88：split-K——linear −17%，TTFT −6.2%，保留**。`PREFILL_SPLIT_K=4`：
`nvfp4_split` 把 K 循环切 4 段、grid ×4（216→864 CTA ≈ 5 波），未缩放 partials `[4×32, N]` 由
`reduce_split` 顺序求和并乘 global scale；partials 按输出宽度去重分配、`BatchGraph` 持有，两图共享、
同 stream 串行复用（新增显存 ~80 MB）。实测（196-token 阶梯）：

| 指标 | 基线（Round 86） | split-K=4 | split-K=8 |
|---|---|---|---|
| linear ms/chunk（图内 raw） | 39.04 | **32.26（−17%）** | 33.16 |
| chunk 总计（校正后） | 52.6 ms | **46.1 ms（−12%）** | — |
| TTFT 热态 | 1592.2 ms | **1493.9–1499.7 ms（−6.2%）** | 1472.8–1518.7（噪声内打平） |

数值：63/63 PASS（容差吸收重结合，SPLIT=4 与 8 各自通过）。SPLIT=8 的 partial 流量与显存翻倍、
图内 linear 反而略差 ⇒ 定稿 4。残余 32 ms 的构成：单 split 内 20 次串行 K 迭代仍暴露
`MMA 依赖链 + DRAM 延迟`，继续加 split 的边际收益已被 partial 读写抵消——**prefill GEMM 的
廉价手段到此为止，剩下的要交给 TP/核内流水线（重投入）或接受现状**。

## 6. 工作量与顺序建议（Round 89 按勘察重写，取代旧 A/B/C 拆分）

**Round 89 勘察推翻的三个旧假设**：
1. ~~引擎缺批契约~~——引擎**已经在凑批**：`packing.rs` 把不同序列的 decode 选进同一 `StepPlan`，
   `submit_shared_borrowed` 一次性提交整个 `&[ExecutionTask]`，输出按 task 分派回 request。
   **断点只在 CUDA 执行层**（`executor/execution.rs` 逐 task 串行 replay）。⇒ 引擎与 IR **零改动**，
   所有 backend 共享同一批契约；CPU/Metal 维持批内循环即可，CUDA 加真并行。
2. ~~KV 需要 lane 标量 + metadata 加字段~~——所有 state kernel（`attention::append`、`decode`、
   `conv4`、`delta`）都已按 `metadata[2] >= 0` 门控写入，**空 lane 掩码是现成机制**；
   逐 lane metadata 张量在 `BatchGraph` 里也是现成的。⇒ **kernel 零改动**。
3. ~~池键按 capacity 分桶~~——共享图把 CAP 固定为槽位常量，桶随之消失（本条保留旧结论）。

**定稿架构：槽位烘焙（slot-baked）**。加载期一次性分配 W 个槽位状态集（KV/conv/delta/metadata），
捕获**一张 W 宽 decode 图**（lane i 烘焙槽位 i 的张量指针）；序列 = 槽位租约 + CPU 侧 history。
空 lane 以 `state_pos = -1` 掩码（现有不变量 #2）。`CudaGraph` 宽度 W = `max_num_seqs`。

**切片（每片可独立编译 + GPU 验证，不留半成品）**：

- **S1（CUDA 批 decode 快路径）**：`BatchBuilder` 支持逐 lane state 集 + 免 checkpoint 的 decode-only
  构建；`BatchGraph::run_lanes` 接受逐 lane `(token, position)`（空闲 lane 掩码）；`execute` 把
  无 sampling 的 decode task 分组成一次 replay，输出按 lane 拆回。MTP/带 sampling 的 task 维持串行旧路。
  验证：63/63 + `mtp-ab.py --depths 0,2` + `check-cuda-service.py` + 3 并发聚合吞吐。
- **S2（槽位化 reserve/release）**：序列从出生即绑定槽位（prefill 图按槽位懒捕获一次），
  `StatePool` 从"整条 Sequence"降级为槽位空闲表；1.34 s/bucket 捕获与 2.71 GB/序列随之消失。
  验证：`check-cuda-service.py`（池路径）热请求不劣于 110 ms。
- **S3（预算按槽位计费 + 长 prompt 回退）**：`sequence_budget` 改槽位计价；超出槽位 CAP 的请求
  走 legacy 单序列路径（两路共存，贪心等价不变）。

**MTP 与 CB 的共存**：`speculate` 逐序列用 width-3 verify 图；投机成功过的序列永久退出批路径
（串行执行），sampling 不影响批资格（token 决策在引擎侧，与 §1 批契约一致）；draft 加载时
不建池（§6a）。跨序列联合 verify 是后续项。

旧文 §1–§5 的勘察事实（机制清单、KV 寻址根因、不变量、验证协议、死路）**继续有效**；
其中"步骤 A/B/C"的具体做法以本节为准。

## 6a. S1 落地（Round 90）：槽位池 + 共享 decode 图，实测与 GEMV 定案

**实现**（全部在 CUDA 后端，引擎/IR 零改动，其他 backend 不受影响）：
- `resident/slot_batch.rs`：`SlotPool` = 3 槽位状态集（KV 按 head 块 D2D 拷入——**KV 是 head-major
  `[kv_heads, capacity, head_dim]`，两侧 capacity 不同必须逐 head 拷**；conv/delta 整拷）+
  一张共享 decode 图。池失败只永久禁用批路径，不毒化请求。
- `batch.rs::build_slots`/`SlotDecodeGraph::run_lanes`：Linear 走 `batch_projection`（3 lane + 1 padding
  共享权重 GEMV），其余节点逐 lane Flat 记录；空 lane 写 `metadata[2]=-1`。校验 cursor 顺序、
  只回读活跃 lane logits。profiling 钩子同 prefills（`INFER_CUDA_PREFILL_PROFILE`，图内 event）。
- `executor/execution.rs`：`run_grouped` 把连续 eligible decode 段（cursor 对齐、Logits readout、
  未投机、capacity ≤ `CB_SLOT_TOKENS=4096`）绑槽位一次 replay；已绑定必批，未绑定 ≥2 lane 才绑
  （单序列保留 MTP）。**draft 加载时（`mtp_depth>0`）永不建池**——投机序列从首个 decode step 就
  失去 eligible 资格，建池纯亏 ~880 MB（3 槽 × 293 MB：fp8 KV 134 + delta 151 + conv 6 MB/槽）。

**诊断与 GEMV 定案（n3-t64，63 步平均）**：初版 46 ms/step。分段计时：linear 39.26 ms（87%），
逐 lane 串行的非 Linear 节点合计仅 6 ms ⇒ 瓶颈是融合 GEMV 没吃到权重共享（39 ≈ 3× 单 lane 13）。
tile 扫描（VERIFY_TILE_COLUMNS × DEPTH，**加粗为定稿**）：

| (BN, BK) | linear ms | | (BN, BK) | linear ms |
|---|---|---|---|---|
| (4,256) | 59.5 | | (32,64) | 37.5 |
| (8,128) | 59.9 | | **(16,256)** | **28.6** |
| (16,128) | 43.7 | | (8,512) | 34.3 |
| (8,256)（旧值） | 39.3 | | (32,256) 及以上 | ≥88.9（寄存器溢出崖） |

机理：每块累加器 `[4,BN,BK]` 常驻寄存器，16k f32 是上限；kernel 既不带宽 bound
（0.49 TB/s ≈ DRAM 峰值 27%）也不算力 bound（7 TFLOPS ≈ 7%），是**延迟/占用率 bound**。
**split-K 否证**：S=4 逐节点对比——中小节点 +0.1 ms（块固定开销×4），lm_head +0.34 ms，
总计 33.0 vs 28.6 ms，已回退（与 prefill GEMM 的 split-K 收益结论相反，勿混用）。

**实测（CB_SLOT_TOKENS=4096，cb-prompt 92 tok + 64 生成）**：
n1-d0 单流 62.6 tok/s（串行旧路）；n3-d0 三并发：66 步完成 192 token（2.9 tok/步 ⇒ 批生效），
**聚合 79.9 tok/s ≈ 1.28× 单流**，步均 34.3 ms。全部验证：63/63 PASS；mtp-ab n1/n3
greedy-equivalent True；check-cuda-service passed=true。

**已知边界（非回归）**：d2-n3（3 条投机序列）准入失败 `required=2.13 GB > available=0.63 GB`——
两条 d2 序列已占满显存，**无池时也放不下**（旧行为相同），准住在正常工作。

**距"并发数 × 单流"的差距与下一步**（按 ROI）：
1. 非 Linear 节点逐 lane 串行记录（6 ms/步）：conv/delta/attention/norm 的 kernel 加 lane 维
   可压到 ~2 ms（CB-4）。
2. pack/unpack 每 Linear 节点 2 次额外派发（497 节点 × ~8 µs ≈ 4 ms 启动地板）：arena 让
   Linear 输入/输出直接落在共享 `[4, N]` 缓冲的行视图上可消除（CB-5）。
3. GEMV 每字节效率 0.49 vs 单 lane 0.9 TB/s：需要 kernel 级重设计（占用率/依赖链），
   与 prefill GEMM 同属"重投入"档（CB-6）。
4. S2（prefill 入槽杀 1.34 s/bucket 捕获）与 S3（预算按槽位计费）不变。
