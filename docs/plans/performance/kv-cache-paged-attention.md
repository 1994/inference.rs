# KV cache 分页重构方案（PagedAttention 的分层与落地）

状态：**待实施**；第①步（块表间接寻址）已落地。源码基点为 `d5afa9b`（2026-10-10）。

整体排期见[路线图](../README.md)的 C1；CUDA 侧其它优化见 [CUDA 方案](cuda.md)；全部实测数据与负面结果见
[性能实验记录](../../research/cuda-performance-experiments.md) 第十三、十四节。本文只维护本项的设计、阶段划分与验收。

## 一、结论先说：PagedAttention 应该分三层放，现在只对了最下面一层

| 部分 | 应住哪 | 现在住哪 | 判定 |
|---|---|---|---|
| **块表间接寻址**（`table[逻辑块] → 物理块`，设备侧一次查表） | **CUDA 后端**（设备） | CUDA：`resident/capture.rs` 的 `Capture.table`、`resident/slot_batch.rs` 的 `SlotPool.tables`、`resident/metadata.rs::identity_table`，被 5 处核使用 | ✅ 正确，且已落地验证（§13.16–13.17） |
| **块池 / 按需增长 / 前缀共享 / 引用计数 / 按块计价** | **engine + SPI 契约** | SPI 已定义：`foundation/spi/src/backend.rs` 的 `kv_cache()`、`state_page_growth()`、`reusable_prefix*()`；`foundation/ir/src/diagnostics.rs:65` 的 `KvCacheInspection`（`block_size`/`total_blocks`/`free_blocks`/`active_blocks`/`cached_blocks`/**`shared_blocks`**/`bytes_per_block`） | ✅ 契约已在，**但 CUDA 未实现** |
| 同一套策略的第二份实现 | —— | CUDA 又自建：`SlotPool` 的 per-slot arena + `free` + `bind`/`SlotCopy` | ❌ 重复，且 engine 看不见 |

**关键事实**：engine 已经在按"页"调度——`pipeline/scheduling/ready.rs:50,118` 与 `overlap.rs:95` 每个调度步都调用
`backend.state_page_growth(state)?` 写入 `cost_query.page_growth`；`admission.rs:51` 与 `resources.rs:11,23` 用
`kv_cache()` 做按块核算。而实现者只有 **Metal**（`backend/metal/src/executor/provider.rs:67,70`）与 runtime 测试替身；
**CUDA 全部走 SPI 默认值**（`kv_cache() → None`、`state_page_growth() → Ok(None)`）。

后果（全部已实测，见 §13.18/13.25/13.29/13.31/13.33）：

- 按块准入在 CUDA 上失效，退回配置里的 `num_gpu_blocks`；
- **按需增长不存在** ⇒ 只能按 `capacity` 一次性预留（即"每序列按最大长度预留"）；
- engine 的计价只能按**字节 × 满容量**（`state_reservation_bytes`），而池在 backend 内部另算一套
  ⇒ 两套词汇互不可见，池与请求抢同一份预算；
- 前缀共享只能停留在"整状态快照拷贝"，因为 engine 看不到"哪些块可以共享"。

**所以本方案的判断是**：把策略继续往 CUDA 里塞（例如"共享 arena + 偏移表"）方向是偏的；正确做法是让 CUDA
**实现 engine 已经依赖的那份分页契约**，把块池、增长、共享的词汇交回 engine 可见的层。

## 二、已落地的地基（不要重做）

| 项 | 状态 | 实测 | 提交 |
|---|---|---|---|
| ① 块表间接寻址：预填充写 + 预填充读（SIMT/张量核）+ decode 读（SIMT/split-KV）五处核 + bind + 预算 | ✅ 默认路径 | 恒等映射下官方 matrix **12/12 trial token 逐位相同**、越界 0 | `953c0ea`、`9fb692e` |
| `CB_SLOT_TOKENS` 2048 → 8192 | ✅ 默认 | 4×2560 并发 wall 10.59 → **9.21 s（−13%）**、TTFT 中位 −22%、`gpu_budget` 递延 −32% | `921abe3` |
| 池内计价（占槽请求不再二次预留） | ✅ 默认（对交付 profile 尚不生效） | matrix 16/16、27 单测、4 GPU 测试 | `9421a8c` 等 |

## 三、硬边界账目（决定了"为什么必须按块"）

| 量 | 值（capacity 8192） |
|---|---|
| 每槽 fp8 KV | **262 MiB**（16 个全注意力层 × 8192 × 4 头 × 256 × 2 × 1 B） |
| 每槽 f32 递推状态（48 层 delta/conv） | **149 MiB** |
| 每槽合计 / 引擎 `state_budget` | **405 MiB** / **4.8 GiB** ⇒ **并发 ≈12 槽** |

**槽容量与槽数互斥**（容量大→长会话能进池但 KV 线性增长；槽数多→并发高但每槽 ×405 MiB）。
只有"序列只持有实际用到的块"能消除这个互斥——这正是本方案要交付的东西。

## 四、分阶段计划

每阶段的硬性要求：默认路径不变、新路径挂开关、单测与 GPU 测试全绿、用官方 matrix 实测并与改动前对比、
数据写进[性能实验记录](../../research/cuda-performance-experiments.md)并提交。**不得让工作区处于编译不过的状态。**

### P1 · CUDA 实现 `kv_cache()`，让按块核算点亮（最小、最快见效）

- 做法：把已有的块池与表如实上报——`block_size = KV_BLOCK_TOKENS`、`total_blocks`、`free_blocks`、
  `active_blocks`、`cached_blocks`、`shared_blocks`、`pool_bytes`、`bytes_per_block`。
  现在这些量在 backend 内部都已经存在（`SlotPool` 的 `free`/`slots`/`capacity` + 表），只是没有暴露。
- 启动条件：无。可与任何其它线并行。
- 完成标准：`/native/v1/observability` 与 `engine/inspection.rs` 出现真实块数；
  `admission.rs:51` 与 `resources.rs` 走按块分支；官方 matrix **16/16 不变**、token 逐位一致。
- 风险：低（只读上报）。

### P2 · CUDA 实现 `state_page_growth()`

- 做法：对一个 state 报告"下一步会用/增长多少块"（当前 state 的 token 数 → 块数，与 `KV_BLOCK_TOKENS` 对齐）。
- 完成标准：`cost_query.page_growth` 不再恒为 `None`；长请求的准入/调度走按块分支；
  `deferred_reasons` 中 `state_capacity`/`gpu_budget` 显著下降（用 4×2560 与 2×8000 两组并发对照）。
- 依赖：P1。

### P3 · 计价改为块数（原第④步）

- 做法：`state_reservation_bytes`/`state_reservation_bytes_for` 与准入改为按块（块数 × `bytes_per_block`），
  并把"池内已预留"的部分从请求报价里扣掉（§13.29 的那一刀，这次减**正确的项**）。
- 完成标准：`CB_SLOT_TOKENS` 不再是墙——`CB_DECODE_SLOTS=16` 的 prewarm 成功、matrix 16/16；
  长上下文请求不再因为"池占内存"被拒。
- 依赖：P1、P2。
- 注意：§13.30 已更正——报价的大头是**固定激活 arena**，不是 KV；减免必须针对池已覆盖的**激活/工作区**与**块**两件事。

### P4 · 块级前缀共享 + COW（原第③步）

- 做法：`reusable_prefix*()` 按页暴露；块内容哈希 + 引用计数 + 写时复制；`shared_blocks`/`cached_blocks` 计数可观测。
- 完成标准：`hot_long` 一类共享前缀场景的 TTFT 与块复用计数同时改善；重复前缀不产生整状态拷贝。
- 依赖：P3。

### P5 · 共享 arena / 按需分配（原第②步的 backend 内部实现，**可选**）

- 只有当 P1–P4 之后 **per-slot arena 仍是状态侧浪费**时才做。注意：在 P1–P4 这条路上它是**可选实现细节**，
  因此 `attention.rs:113` 那个 draft 路径越界**不在关键路径上**。
- 若要做，先按 §13.23 的办法定位：在**回放时**打印 `(lane, position, block, 表项)`，并核对 draft 私有 program 的
  `identity_table` 与它 arena 的归属（不要再用推理；§13.26–13.28 已排除"表形状/取值与 arena 不匹配"整族假设）。
- 参考实现进度：`/tmp/shared-arena.patch`（`KvSpan { dst, src }` 双侧偏移 + `arena_capacity` 拆分 + `slot_tables`）。

### P6 · token-major 块内布局 + 张量核 SDPA（原第⑤步）

- 前置：P3 之后按块寻址稳定。目标是把已验证的张量核 attention（§13.9，长上下文 **6.2×**）接到分页布局上。
- 完成标准：长上下文（≥3072）attention 走张量核且 token 与 SIMT 路径在 1e-4 相对误差内一致。

## 五、并行边界与门禁

- 不触碰：`crates/model/package/**`、`Makefile`、`tools/package/**`、`docs/plans/speculation/**`（并行进行中）。
- 提交前：`make local-build`、`cargo clippy -D warnings`、`cargo check --all-targets`、27 单测、
  `--ignored resident::slot_batch resident::slot_verify`、`tools/check/gate.sh`。
- 测量：`tools/bench/safe-run.sh --memory-gib 12 python3 tools/bench/serve-compare.py …--run-id ID --profile-id 27b-mtp2`；
  并发对照用固定 N 与固定 token 数的并发脚本，**同一小时内的成对测量**，并复测（单次图总曾经骗过人，§12.22）。

## 六、不要再走的路（已实测否证）

| 路线 | 否证 |
|---|---|
| 加大 `CB_SLOT_TOKENS` 突破容量上限 | 16384 在 MTP 下准备挂起 30 s（§13.30/13.31）；不解决并发 |
| 加大 `CB_DECODE_SLOTS` 提并发 | 16 槽 prewarm 失败：12 槽是状态账目的硬上限（§13.33） |
| 池内减免 KV 计价 | 减错了项，报价大头是激活 arena（§13.30） |
| prefill rung 预算 | 扣掉池的 arena 后仍选最宽档，未解决 prewarm（§13.32） |
| "表/arena 错配"族假设 | 表的形状与取值都与 arena 自洽（§13.27/13.28） |
| cuBLASLt 委派 / GEMM tile 扫描 / SDPA TMA / delta 代数重写 / value-dim 拆分 / outlier 归因 | §12.22 等 |

## 七、未决问题

1. **MTP 池几何挂起**：`(2048, 16 槽)` 与 `(16384, 4 槽)` 在准备阶段挂起 30 s、0 token，而 `(8192, 4)` 正常。
   需要单独定位（怀疑与池化推测图捕获在特定 (capacity, width, verify) 组合下的行为有关）。
2. **状态侧并发上限**：149 MiB/槽的 f32 递推状态若改 bf16 可近乎翻倍并发，但有数值风险；
   官方 matrix 的 token 对比可判定，需单独立项。
3. **跨层契约的测试面**：P1–P3 的验收涉及 engine 调度/准入分支，需要设备窗口与观测口径确认。
