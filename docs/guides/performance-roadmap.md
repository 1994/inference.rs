# Inference 性能现状

**目标**：native CUDA serving 延迟对齐 vLLM/SGLang（≤1.10x），覆盖 Qwen3.8-27B-NVFP4 与
qwen3vl-2b 的 short/long/batch4/hot_long 四个用例。

**结论：未达成。**

**后续执行计划**：见 [performance-improvement-plan.md](performance-improvement-plan.md)。

## 一、当前差距

比值 = native / vLLM，**小于 1.0 表示我们更快**。完整矩阵，1 次预热 + 3 次测量，
`--gpu-memory-utilization 0.88`，原始报告在 `artifacts/perf-r40/`。

| 用例 | wall | TTFT | TPOT | |
|---|---:|---:|---:|---|
| 2B short | **0.974** | 1.005 | **0.975** | 三项全过 |
| 2B hot_long | 1.084 | 1.873 | 1.028 | |
| 2B long | 1.137 | **3.256** | **0.979** | |
| 2B batch4 | 1.490 | **2.630** | 1.383 | |
| 27B hot_long | **0.951** | **0.369** | 1.187 | |
| 27B short | 1.150 | 1.269 | 1.141 | |
| 27B long | 1.761 | **5.863** | 1.272 | |
| 27B batch4 | 1.201 | **2.451** | **1.897** | |

**解码已追平**：2B 四个用例中三个 TPOT 持平，27B 在 1.14–1.27。差距集中在 **prefill**，
以及**并发负载下的 decode**。

**生产优先级**：27B batch4 TPOT 1.897 > 两个 batch4 TTFT ≈2.5 > 2B batch4 TPOT 1.383。
TPOT 是每 token、每请求都交的税，直接等于吞吐差距；TTFT 只影响首次响应感受。

## 二、两个瓶颈

### 1. 27B 长 prompt TTFT 5.86x

511 token = **8 次 target prompt graph replay × 64 token**，每次 42.5 ms = 340 ms，实测
TTFT 398 ms。**执行墙时间的 94% 是设备端 replay**，host 只加约 3 ms。

同一进程、同一 profiler 下，相同权重：

| graph | 节点数 | 时间 | 权重带宽 |
|---|---:|---:|---:|
| prompt（64 token） | 1154 | 42.88 ms | **505 GB/s**（峰值 28%） |
| verify（3–12 lane） | 1155 | 24.71 ms | **947 GB/s**（53%） |

**融合度几乎相同（节点数差 1），时间差 73%** —— 差距不在融合，在 kernel 的分块效率。

**原因已确认**：量化 GEMM 的 output tile 是 `[16, 64]`、weight partition 是 `[64, 128]`，
`pid.0` 选 row block。64 行 → **4 个 row block → 每份权重读 4 次**；verify 只有 1 个
row block，读 1 次。

**但 `QUANT_GEMM_TILE` 这条路走不通**（两个实测）：全局改 64 行 → short wall **+20.8%**
（decode 只 1 行，63/64 空转）；按张量推导高度 → short wall **+394.6%** 且 TTFT 收益消失。

阻塞点：**kernel 有 16 行下限，而 prompt 与 decode 在同一 program 内共用同一个 workspace。**

> **已落地的修法**：按行数分派的特化 tile（`quant_gemm_tile`，prompt 64 行 / decode 16 行，
> `QUANT_GEMM_TILE` + `PROMPT_GEMM_TILE_ROWS`），FP8 路径镜像同修。实测：27B long 的 prompt
> replay 46.9 → 43.1 ms、`linear` 桶 27.3 → 23.1 ms；服务矩阵 TTFT 比值 short
> 1.30 → 1.22、long 5.91 → 5.45、batch4 2.53 → 2.39，token 序列逐字节一致。
> **不要**改成全局宽 tile：隔离 kernel 基准里 `[64, 128]` 每个行数都更快，服务侧却全局回退
> （short wall 1.14 → 1.30、batch4 TPOT 1.86 → 2.05）—— 单次 40-CTA 的 decode GEMM 填不满
> 设备，64 行 CTA 在单行 decode 上浪费 63 行 tensor-core 计算。分块改动必须按服务矩阵复测。

### 2. batch4 TTFT 2.45–2.63x

**不是 prefill 算得慢，是执行次数多。** 实测：

- 单独 48-token prefill：**8.4 ms**（vLLM 的 batch4 TTFT 是 19.5 ms —— 单看 prefill 我们更快）
- 含 prefill + 1 个 decode 的 step：**27 ms**（纯 decode step 只要 5.0 ms）
- 请求**串行准入**：四个同时到达的请求 TTFT 为 0.020 / 0.047 / 0.047 / 0.061，3 倍离散

每个请求要 **3 次 execution，1 次就够**。

> **未做的修法**：prefill / decode 阶段分离，把攒到的 prefill 合并成一次 execution。预估
> 2B batch4 TTFT 从 47 ms → 8.4 ms（2.63x → ~0.43x）。**代价未验证**：会牺牲并发下的
> TPOT，必须实测。

## 三、决策点在哪（代码位置）

上节两个瓶颈都落在"分块/调度决策"上，不在 kernel 实现质量上。三层决策点如下，
**「已读」= 逐行看过，不是猜的；「只 grep 过」= 知道位置但没读全**：

### 3.1 GEMM 分块尺寸（已读，§二.1 的直接来源）

```rust
// crates/backend/cuda/src/constants.rs:32
pub const QUANT_GEMM_TILE: [usize; 2] = [16, 64];
//                                       ↑ 行数（block 高度）

// crates/backend/cuda/src/resident/nvfp4_gemm/workspace.rs
kernels::packed(output.partition(crate::constants::QUANT_GEMM_TILE), …)  // :126，4-bit 路径
kernels::matmul(output.partition(crate::constants::QUANT_GEMM_TILE), …)  // :167，8-bit 路径
//   ↑ 两处是 QUANT_GEMM_TILE 在整个 crate 里的唯一使用点（grep 确认）

// crates/backend/cuda/src/resident/nvfp4_gemm.rs
fn packed<const K: i32>(out: &mut Tensor<f32, { [16, 64] }>, …) {
    let wp = weight.partition(shape![64, 128]);   // [N 方向, K 方向]
    let x = xp.load([pid.0, k]);                  // pid.0 = row block 序号
    let w = wp.load([pid.1, k]);                  // pid.1 = 列 tile；不依赖 pid.0
}
```

`w` 的加载只依赖列与 K，**不依赖 `pid.0`** → 每个 row block 重读一遍权重。
64 行 ÷ 16 = 4 个 row block = **权重读 4 次**。verify 路径只有 1 个 row block，读 1 次。

**同一份 kernel，只因分块参数不适合，效率掉一半** —— 所以这不是"kernel 写得差"。

### 3.2 prefill graph 宽度（已读；决定 prompt 要跑几次 replay）

| 位置 | 作用 |
|---|---|
| `crates/backend/cuda/src/loading/mod.rs` | 算出 `prefill_width`，两个门：`profile.arena_budget_bytes()` 与 recurrent 上限 |
| `crates/backend/cuda/src/resident/batch.rs:99-110` | `build_pair`：`prefill_width >= PREFILL_LANES` 决定走哪条捕获路径 |
| `crates/service/cli/src/support/serving.rs:38` | 把后端报告的 `prefill_width` 写进 `scheduler.prefill_chunk_tokens` |

27B 的 `prefill_width = 64` → 511 token 需要 8 次 replay。

### 3.3 execution 里装什么（**只 grep 过，未深读**；§二.2 的来源）

| 位置 | 作用 |
|---|---|
| `crates/engine/scheduler/src/policy/packing.rs` | 打包逻辑。`ExecutionRole::{Decode,Prefill,Forward,Mixed}`；`compatible()` 只校验 program 与 backend 一致，**不禁止 stage 混装** |
| `crates/engine/runtime/src/config.rs:22` | `DEFAULT_TOKEN_BUDGET = 64` |
| `crates/engine/runtime/src/config.rs:91` | `max_num_batched_tokens` |
| `crates/foundation/ir/src/scheduling.rs:12` | `DEFAULT_FAIR_QUANTUM_TOKENS = 8` |

**注意**：动 §二.2（stage 分离）需要先完整读这一层。上面只是位置索引，不足以据此下方案。

### 3.4 `QUANT_GEMM_TILE` 修法的阻塞点（三次实测）

| 尝试 | 结果 |
|---|---|
| 全局改成 `[64, 64]` | 编译过、27 个 host 测试过；short wall **+20.8%**（decode 只有 1 行，63/64 空转） |
| 从 `output.shape()[0]` 推导高度 | 运行通过、测试过；short wall **+394.6%**，且 TTFT 收益消失（decode 行数 4–12 → 高度降到 1–4，`mmaf_scaled` 退化 3–4 倍） |
| `Workspace` 加 `tile_rows` 字段 | `output partition shape mismatch` —— **一个 `Workspace` 服务多种宽度的图**（prompt 用 `prefill_width`，batched/restore 图用 `batch_width`，lane 图另有），而 kernel 期望的高度由**每张图捕获时各自张量的 partition** 决定；共享 workspace 上的可变字段无法区分（最后写入者赢） |

**根因**：prompt 与 decode 的最优行数相反（64 vs 1），但**共用同一个量化 GEMM 和同一个
workspace**，且 kernel 有 **16 行硬下限**。

## 四、已被实测否掉的方向（不要重试）

| 方向 | 否掉理由 |
|---|---|
| split-K（窄行投影） | kernel 快 1.8x，serving **慢** 1.8x |
| 128-lane 27B prompt graph | short TTFT 0.058→0.084，batch4 0.247→0.291 |
| decode slot pool 4→8 | 8 路 batch 4.10 s → **6.6 s** |
| host readback / scan | host 在等设备；真实 greedy scan 1.67 ms，不是 90 ms |
| 128-lane conv tile | cuTile 要求 tile 为 2 的幂；per-lane state 无法用超过自身行的 tile 切分 |
| per-lane `delta` launch | 用错误的总体缩放过 per-op 份额；实测只有 18% TTFT，且 short 回退 17% |
| chunked recurrent prefill 数值门 | 18% TTFT 收益不值得冒已记录的数值风险 |
| 提高 fair-quantum 8→128 | prefill execution 成本与 token 数无关，调度阈值改不动 |
| 带宽作为限制性 roofline | 拿 BF16 balance 去比 4-bit/8-bit 权重，分母用错 pipeline |
| row tile（全局 & 推导两种） | 见 §二.1；那 7% 收益来自全局常量对辅助图布局的副作用，不是消除重读 |

## 五、已落地

- **跨引擎 harness** —— `tools/bench/serve-{compare,workloads,cases}.py`，含标准矩阵与单测
- **Profile（可选开关）** —— `INFER_CUDA_EXECUTION_PROFILE`（含 host phase，覆盖 decode 与
  prefill）、`INFER_CUDA_PREFILL_PROFILE`、`INFER_CUDA_PROFILE_GRAPH_ONLY`；经
  `/native/v1/runtime` 暴露 `execution_profile`
- **两个加载期缺陷已修** —— prompt graph 宽度只受 arena 预算约束（2B 507-token TTFT
  0.0835→0.0542 s）；CUDA prefill chunk 取自后端解析宽度，不再用通用 64
- **按行数分派的量化 GEMM tile** —— 见 §二.1 的实测结论；`packed`（FP4）与 `matmul`（FP8）
  的 row/column tile 变成 capture 期泛型常量，新增 `nvfp4_packed_tile_sweep` 隔离基准
  （`--ignored`）。原前置补丁 `docs/patches/quant-gemm-row-tile-m.patch` 已合入并删除

## 六、测量纪律

**target prompt graph 与 MTP draft head 在同一个 profile 文件里都报 `graph = "prefill"`**，
两者不可区分。曾据此把 draft 的 1.94 ms 当成 target 的成本、把两个 program 的数字混算，
浪费了数轮。

规则：**用任何数字前先确认它来自哪个 program、哪张捕获图、哪个总体；优先同源重测，
不要用两个来源的数字拼比值。** 本文档历史上 9 次更正大多源于此。
