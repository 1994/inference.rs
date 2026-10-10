# CUDA 性能实验记录

**定位**：保留既有 A/B、失败实验、源码调查和已落地改动。本文包含历史假设与早期待办，当前任务状态和执行顺序统一见 [路线图](../plans/README.md)，设计与验收见对应详细方案；正式性能实验统一执行 [性能基线方案](../plans/performance/baseline.md)。引用 [CUDA Serving 历史记录](cuda-serving-baseline.md) 处仍记为 RM§x。

**历史探索目标**：第一阶段 8 格全矩阵 ≤1.10x；第二阶段 TPOT 类指标稳定领先 vLLM 20–30%。这些是目标，不能作为已取得或可直接承诺的收益。

当前拟议的 recurrent 回滚方案采用保留实际 F32 输入的 Record/Fold，尚未实现；资源与精度边界见 [状态重放方案](../plans/speculation/state-replay.md)、[Infernix 调研](infernix-performance.md)。下文的早期 BF16 记录容量、草稿压缩目标及探索性建议以这些方案的完整报价与验收为准。

**置信度标记**：【确证】= 代码或实测确认；【推断】= 有依据的推理；【待测】= 依赖尚
未做的实验。推算收益一律以绝对锚点给出，方便下一轮 bench 后用实测替换。

## 〇、三个共识前提

后续所有方案建立在这三条上：

1. **三个瓶颈同一条主线：「一遍权重服务多少 token」**。27B long TTFT 是 prompt 图权重
   读 4 遍（RM§二.1）；batch4 TTFT 是准入串行 + 每请求各一次全宽 replay；27B batch4
   TPOT 疑似每步 replay 数随并发上涨。修法都指向"权重一遍过、token 成批过"。
2. **调度预算按 replay 次数、不按 token 数**。后端实测 prefill 单次 replay 成本与
   chunk 内 token 数无关（固定宽度图，RM§四 与 `executor/execution.rs:112-141` 互证）
   → 切碎 prefill 是纯亏损；保护 ITL 的旋钮是"本步是否发 prefill"，不是"发多少
   token"。这条与经典 continuous batching 的成本假设相反，是调度侧一切改动的起点。
3. **测量纪律沿用 RM§六**，新增三个一等指标（见 §五）：每图带宽 %、每 token replay
   数、MTP 接受长度。

## 一、追平（≤1.10x）：三项修法

### ① 27B batch4 TPOT 1.897 —— 先测量，假设是 MTP 路径

**机制事实**【确证】：batch decode 走共享 slot 图、一次 replay 推 ≤4 条 lane
（`resident/batch.rs:66`，权重一遍过）；MTP 开启时 decode 走 `slot_speculate` =
propose + verify + catch_up **三次 replay**（`executor/execution.rs:481-550`）；单发
/非池化 decode 回退到每请求私有图串行（`execution.rs:314,348-380`）；
`CB_DECODE_SLOTS = 4`（`constants.rs:101`）。

**假设**【待测】：batch4 恰好顶满 slot 池，每步 replay 数 >1（MTP 三倍化或溢出串行），
每步权重遍数随并发上涨 —— 与"batch1 TPOT 1.14–1.27、batch4 反而 1.897"自洽。

**实验（不改代码）**：

1. `INFER_CUDA_EXECUTION_PROFILE` 跑 27B batch4 稳态段，数每 step replay 次数与 graph
   名。≠1 则根因坐实。
2. 同负载 `--num-speculative-tokens 0` vs 开启，对比 TPOT 与接受长度。

**修法（按成本排序，确认后执行）**：

- **自适应 MTP**：active lanes ≥ 阈值或接受长度 < 阈值时跳过 draft/verify，走纯 slot
  decode。vLLM/SGLang 高并发下同样关闭投机解码。【推断：改动在 executor/调度交界，
  中等】
- 若 verify 确认未跨 lane 成批：verify lane 化。
- 顺带削减每 replay 的同步 readback（`batch.rs:807`）。

**预期**【待测，条件性】：27B batch4 TPOT 1.897 → ~1.2（回归 batch1 水平）；2B
batch4 TPOT 1.383 → ~1.0。

**实测（两个实验都做了，假设被否）**：`INFER_CUDA_PREFILL_PROFILE` 数出的每步
replay 是 2×`slot_decode`（合批 draft propose，各 ~1.6 ms）+ 1×`slot_verify`
（target 12-lane 前向，~27 ms）；`slot_verify` 里 `linear` 19.9 ms、`delta` 4.2 ms、
`conv` 2.3 ms。同负载 `--num-speculative-tokens 0` 反而更慢：TPOT 24.3 ms vs 开启
20.2 ms。**所以 batch4 TPOT 的根因不是"投机三倍化"，关掉投机是净亏损**；瓶颈是
target 前向本身（杠杆 A 的带宽 + `delta` 延迟），MTP 已经在帮忙（每步 ~1.9 token/lane）。
自适应 MTP 作为"高并发让位"机制仍可留作长尾保护，但不能当作 TPOT 的修法。

### ② batch4 TTFT 2.45–2.63x —— 快修 A/B + 结构修 C

**机制链**【确证，来自调度层逐行调查】：

- 执行严格单飞（`runtime/src/engine/mod.rs:123`）；worker 在 step 飞行中不服务控制
  命令（`runner/worker.rs:254`）→ 准入被量化到 step 边界。
- 准入 = 2–3 轮异步工单（quote → Reserve → Prefix，
  `pipeline/admission.rs:361-413`、`pipeline/scheduling/resources.rs:33-74`）；冷态
  Reserve 现场捕私有 CUDA 图（`executor/state.rs:100-108`，且要求 `idle()`，
  `state.rs:45`）。
- mixed step = 多个 task 各自串行 replay（`execution.rs:112-141`），无跨 task 融合；
  prefill chunk 跑固定宽度图、成本与 token 数无关；MTP 时另有 `prime_draft`
  （`execution.rs:727-729`）。27ms ≈ 8.4ms prefill replay + draft/verify 开销 +
  readback/host 间隙【构成确证，精确分解待 profile】。
- CudaBackend 未实现 `completion_timing`（`foundation/spi/src/backend.rs:142-144`
  默认 None）→ 成本模型永不校准（`scheduler/src/cost.rs:272-274` fallback 1µs/token）
  → `gpu_budget_us` 是死代码。

**快修 A：准入去串行化**【推断：低风险】

- 压缩工单往返（quote+Reserve 合并；无 prefix 可命中的新请求跳过 Prefix 往返）。
- 冷捕图改为**加载期按容量预热池**（池复用路径已存在，`state.rs:70-98`）。
- worker 单飞模型不动，只降往返次数。
- 附带收益：RM 中 27B long TTFT 的 ~55ms 前后开销、2B hot_long TTFT 1.873 同源自降。

**快修 B：实现 CudaBackend 的 `completion_timing`**【确证是前置】

使能项：不产生直接毫秒收益，但它校准分角色 replay 成本、`gpu_budget_us` 才生效，
后续一切按时间预算的调度调参都依赖它。

**结构修 C：lane 化 prefill（跨序列一次 replay）**

修正 RM§二.2 的预期：调度层合并 execution **不够** —— 后端对非 decode task 仍逐个串行
replay，4×8.4=33.6ms 赢不了 vLLM 的 19.5ms。要赢需跨序列 lane 化 prefill 图（注意力
按 lane 屏蔽；slot decode 图已证明该机制可行）。配套：步级 token 预算 ≥ Σ pending
prefill token（现默认 64，`runtime/src/config.rs:22`，装不下 4×48）。

**预期**（锚点：2B 单次 48-token prefill 8.4ms；native batch4 TTFT ≈47ms vs vLLM
≈19.5ms）：

- 仅 A：2.63x → ~1.3–2.0x（省准入排队，设备时间仍 4 次串行 replay）【推断】。
- A+C：→ ~0.5–0.8x（4×48 token 一次 replay ≈ 8.4–12ms）【推断，依赖 C 成立】。
- 27B batch4 同机制，量级需一次 profile 补数据。

**TPOT 代价门槛**：阶段分离/合并须实测 batch4/hot_long 稳态 TPOT，回退 ≤5% 才合入
（RM§二.2 的遗留要求）。

**已落地（比 A/C 更前置的一步）：prompt chunk 原子化**。profile 显示真正的浪费不在
准入，而在调度把 prompt chunk 切碎：`packing.rs` 的 `candidate()` 把 prefill quantum
再截到 `fair_quantum_tokens`（8），`probe()` 还会递减重试，于是 4 个并发 52-token
prompt 被切成 11/21/31/41 的碎片，而**每个碎片仍然付一次固定宽度图的完整 replay**
（`INFER_CUDA_EXECUTION_PROFILE`：base 120 次 prefill chunk 里有 8 次纯碎片）。
改成 `candidate()` 对 prefill 直接给整块、本轮预算装不下就整轮推迟（不在 token 预算上
切片）；`probe()` 仍保留自 `quantum` 向下的递减搜索，所以设备预算（`gpu_budget_us`）
拒绝整块时，单 token 的 overrun 逃生口仍然有效、不会出现空 StepPlan；预算小于一个
chunk 时也回退旧的 fair-quantum 行为，两种退化路径都不会饿死 prompt。实测：

| 用例 | wall | TTFT | TPOT |
|---|---|---|---|
| 27B batch4 | 1.342 → **1.313** | 2.620 → **2.114**（中位 241→194 ms） | 2.148 → 2.086 |
| 2B batch4 | 1.655 → **1.628** | 2.506 → **2.023**（中位 49→39 ms） | 1.549 → 1.584 |

5 轮交错 A/B（15 个 wall 样本、60 个 TTFT/TPOT 样本），token 序列逐字节一致；profile 里
prefill chunk 全部是整块（51/52/63/64），碎片 replay 消失。单请求用例（short/long/
hot_long）行为**可证等价**：预算 64 = 一个 chunk，切与不切发出的 token 数相同，所以只有
并发 prompt 的场景变化。推迟的 prompt 由既有 aging（20 ms）兜底，不会饿死。

### ③ 27B long TTFT 5.86x —— 64 行特化 GEMM + 宽度分档连锁

**阻塞的绕法**【确证，重读 `workspace.rs` 后的设计】：不需要给 Workspace 加任何可变
字段。`record()` 已从 output 张量推出 `rows`（`resident/nvfp4_gemm/workspace.rs:
109-110`），buffers 本就按 `(rows, columns)` 做 key（`:17`）。tile 选择做成 record
时的纯函数：

```
rows ≤ 16 → 现有 [16,64] kernel（decode 路径不动）
rows > 16 → 新 [64,64] 特化 kernel（prompt 64 行 → 1 个 row block → 权重读 1 遍）
```

同时绕开 RM§3.4 两次失败：clamp 两档不会跌破 kernel 16 行硬下限（避免 +394.6% 的
mmaf 退化）；按 rows 分派没有共享可变状态（避免"最后写入者赢"）。实现 = 复制
`packed` kernel 改输出 tile（`nvfp4_gemm.rs:90-116`，约 20 行）+ `workspace.rs:126`
按 rows 选 partition 常量与 kernel。FP8 兄弟 kernel `matmul`（`fp8_gemm.rs:34-59`，经
`workspace.rs` 的 `record_fp8` 用同一常量）有相同 4x 重读结构，同 PR 镜像修。前置补丁
`docs/patches/quant-gemm-row-tile-m.patch` 已合入并删除。

**预期**【推断，锚点：prompt replay 42.88ms@505GB/s vs verify 24.71ms@947GB/s，同
权重同节点数】：单 replay 42.88 → ~24.7ms；27B long TTFT 5.86x → ~3.8x（8×24.7+55
≈ 253ms vs vLLM ≈68ms）。单独不到 ≤1.10x。

**实测（已落地）**：修法与预期机制不符。按行数分派落地后，隔离 kernel 基准
（`nvfp4_packed_tile_sweep`）显示"权重读 4 遍"确实存在（`[16,64]` → `[64,64]` 在
m=1..64 全部形状上快 1.3–1.5x），但服务侧收益小得多：27B long 的 prompt replay
46.9 → 43.1 ms（−8%），`linear` 桶 27.3 → 23.1 ms（−15%），TTFT 比值 5.91 → 5.45
（−8%）；short 1.30 → 1.22，batch4 2.53 → 2.39，wall 回退 0.3–2.8%，token 逐字节
一致。**不要把 tile 全局放宽到 `[64, 128]`**：隔离基准里它在每个行数都更快
（0.66–0.87x），服务侧却全局回退（short wall 1.14 → 1.30、batch4 TPOT 1.86 →
2.05）。隔离 GEMM 基准缺 launch 间隙与并发形状，不能替代服务矩阵。

**否掉的剩余假设**：prompt replay 的其余时间不在权重流量上 —— 同一 replay 里
`delta`（48 层线性注意力的循环状态）占 16.2 ms，`linear` 只降了 15%；12-lane 的
target verify 里 `linear` 19.9 ms、`delta` 4.2 ms、`conv` 2.3 ms。带宽只有峰值
~40–50%，属延迟遮蔽问题（杠杆 A），不是重读问题。

**连锁（③′）：prompt 图宽度分档 —— 必须重测 RM§四 被否的 128-lane 实验**。该实验是
在坏 GEMM 上做的（128 行 = 8 个 row block = 权重读 8 遍），结论已失效。③ 修好后带
宽 bound 的宽图边际成本骤降（更宽仍一遍权重）。目标形态：宽度分档的多张 prompt 图
（如 64/256，按 prompt 长度路由）。预期：27B long TTFT → ~1.5x（2×24.7+55 ≈ 105ms）；
2B long 3.26x → ~1.2x（507 token 从 4 次 replay 到 1 次 512 宽）。**27B long 要到
≤1.10x 还需 ②A 把 ~55ms 前后开销砍到 ~18ms 以下 —— 该格需要 ③+③′+②A 同时落地。**

**实测（已落地）**：不对。③′ 的收益**不**来自更宽的 bound，而来自"**按 chunk 长度路由到最窄够用的图**"。
一次 prompt chunk = 一次固定宽度 replay，成本与 chunk 内 token 数无关，所以
128 宽图对 511-token prompt 是省 4 次 replay，对 51-token prompt 是白跑 60% 的 lane。
两次实测：27B 全程 128 宽（`--max-num-batched-tokens 128`）long TTFT 5.72 → 4.98，
但 short 1.18 → 1.47、batch4 1.95 → 2.41（+24%，与 loading 注释里记的三格回退同源）。

落到 `ProgramWeights::{prefill_width, narrow_prefill_width}` + `DeviceProgram::prompt_narrow`
后：**捕获宽窄两张 prompt 图，按 chunk 长度选图**（≤ 窄宽走窄图，否则走宽图），
engine 的 chunk 仍取宽宽（`prefill_chunk_tokens = 128`）。27B 选到
`prefill_width=128, narrow=64`，2B 仍是单张 128 图（窄宽=宽宽，不额外捕获）。
实测（4 轮交错 A/B 池化，24 个 wall / 96 个 TTFT·TPOT 样本，token 逐字节一致）：

| 27B 用例 | wall | TTFT | TPOT |
|---|---|---|---|
| long | 1.914 → **1.686**（−12%） | 5.973 → **4.717**（−21%） | 1.383 → 1.321（−5%） |
| batch4 | 1.240 → 1.198（−3%） | 1.953 → 1.972 | 1.980 → **1.837**（−7%） |
| short | 1.253 → 1.202（−4%） | 1.210 → 1.200 | 1.260 → **1.201**（−5%） |
| hot_long | 0.992 → 0.992 | 0.368 → 0.366 | 1.249 → 1.244 |

2B 配置完全未变（单图），其 ±5% 摆动即噪声下限。显存：27B 两张图并存后 resident 22.9 GiB /
state_budget 4.8 GiB，与单张 64 图（23.0 / 4.7）持平，准入并发不变。
`MAX_PREFILL_LANES` 仍是 128；再宽需要同时放开 `loading/mod.rs` 的宽度白名单。

### 合成预期（全落地后，推算）

| 用例 | wall 现值 | wall 预期 | 依赖 |
|---|---:|---:|---|
| 2B short | 0.974 | ~0.97 | 不动 |
| 2B hot_long | 1.084 | ~1.0–1.05 | ②A |
| 2B long | 1.137 | ~1.0 | ③′ |
| 2B batch4 | 1.490 | ~1.0–1.2 | ②A+①（+②C 更低） |
| 27B hot_long | 0.951 | ~0.95 | 不动（① 不得误伤 batch1 MTP） |
| 27B short | 1.150 | ~1.05 | ③ |
| 27B long | 1.761 | ~1.3–1.4 | ③+③′；仍红，TPOT 1.272 是 decode 侧问题 |
| 27B batch4 | 1.201 | ~1.0–1.1 | ①+②+③ |

约 7/8 格进 ≤1.10x；27B long 残留为下一个主攻目标（需单独的 decode 侧根因轮）。
下一轮 bench（perf-r41）后用实测绝对值替换本表。

### 模型通用性（换模型是否要重改）

kernel 工作在"量化格式 × 张量形状"层面，无模型特定假设：K 是 const generic 按
columns 加载期自动编译；prefill_width/batch_width 由加载期 arena 预算推导
（`loading/mod.rs:252-279`）；③ 的分派依据 rows 同为形状驱动。

| 新模型 | 是否要动 |
|---|---|
| 同结构同 NVFP4 不同尺寸 | 零改动 |
| 同结构 BF16 | 不走此 kernel，无影响 |
| 同结构 FP8 | 镜像修 `matmul`（已含在 ③ 内） |
| 不同结构（MoE/hybrid 等） | kernel 仍通用；新 GEMM 路径（如 grouped GEMM）需单独检查同类问题 |

tile 常量绑定的是硬件（tensor core MMA 形状、带宽），换 GPU 架构才需重调。换模型
要做的只是常规复验（数值门 + bench），不是适配。

## 二、超越（<1.0x）：五个杠杆

追平后双方都是"每步一遍权重"，赢家 = 带宽利用率更高、每 token 步数更少的人。

**前置基线核查**【待测，一次配置检查】：bench 里 vLLM 是否跑同一 NVFP4 checkpoint。
若它实际跑 BF16，我们手握 4x 权重流量优势却没赢，余量比矩阵显示的更大。

- **杠杆 A：decode GEMM 带宽 53% → 75–80%**（最大、最通用）。verify 947 GB/s 是该卡
  峰值（~1.79TB/s）的 53%；Marlin 级 W4 kernel 同档卡可到 75–85%。方向：K 向
  cp.async.bulk/TMA 双缓冲、更宽 N tile 增加在途字节（现 weight tile [64,128]、每 k
  步仅 4KB，延迟遮蔽不足是 53% 的典型成因）、quantize 与 GEMM 的依赖链流水化。
  **不要走 split-K（RM§四 已否）**。预期 TPOT 全面 ×0.7 量级。【推断，工程量大但无
  未验证假设】**外部佐证**：同栈的 HF grout（cuTile Rust）在同档 5090 上 decode 达
  76% 带宽（171 tok/s × 8GB fp16），论文报告其 GEMM 达 cuBLAS 98% —— 带宽余量真实
  存在，cuTile DSL 不是天花板（arXiv:2606.15991）。**委托对照路**：quantize 本来就
  是独立 kernel（`workspace.rs:115-123` 与 `packed` 分开 record），换 cuBLASLt
  NVFP4（Blackwell 原生 block-scaled）零融合损失，Rust FFI 即可、不引入 C++；按形状
  赛跑进 `tuning.rs` 成对测量，差 <2% 用二进制、差得多留自研。regime 判断【推断】：
  decode（M≤16）带宽 bound 可追平；prefill（M=64，AI≈256 FLOP/B）在分水岭上、部分
  算力 bound，cuBLASLt 可能保有 10–20% kernel 级优势 —— 用系统级优势覆盖，不死磕。

  **实测（2026-10，RTX 5090 / CUDA 13.4，冷 L2 协议）**：`grout` 的做法是**线性层全部
  交给 cuBLAS**，自己只用 cuTile 写 attention / norm / rope / KV / argmax（`src/cublas.rs`，
  `gemm_ex` + TN + fp16 in/out；`GROUT_CUBLAS_COMPUTE16` 在 sm_120 上默认用 **fp16 累加**，
  并带 per-arch/per-shape 的 tuning record）。grout 的对比是同并发下单请求吞吐略胜 vLLM
  （B200 Qwen3-32B `request_gen_tps` 79.6–80.1 vs 78.8–79.2），**它不是 serving 吞吐引擎**，
  所以"像 vLLM"= 单请求快路径做到极致，与我们的 batch4/并发格不是同一件事。

  照抄 `杠杆 A` 的委托路做了一次成对测量（同权重布局、不重排、同一 L2 flush 协议）：

  | 形状 (m,n,k) | 我们出厂 tile | 我们最好 tile | cuBLASLt NVFP4 | 倍率 |
  |---|---|---|---|---|
  | 12, 17408, 5120 | 0.0630 ms `[16,64]` | 0.0427 `[64,128]` | **0.0369 ms** | 1.71x |
  | 64, 17408, 5120 | 0.0426 `[64,64]` | 0.0422 `[64,128]` | **0.0389 ms** | 1.10x |
  | 12, 5120, 17408 | 0.0648 `[16,64]` | 0.0524 `[16,128]` | **0.0389 ms** | 1.67x |
  | 64, 5120, 17408 | 0.0648 | ~0.053 | **0.0410 ms** | 1.58x |

  即：**窄行（decode/verify）形状赢 1.6–1.7x，prompt（M=64）只赢 1.1x**；TPOT 是最大受益
  项（verify replay 线性 19.9 → ~12 ms 量级），prefill 收益有限。热 L2 下 cuBLASLt 看着有
  3x，是 L2 假象，必须用冷 L2 协议比较。

  **落地阻塞点【确证，查 cuBLAS 文档 + 实测】**：`CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`
  的 scale 不是线性 `[n, k/16]`，而是 **128×64 分块 + swizzle** 布局（"a single tile of
  scaling factors is applied to a 128x64 block"，offset 公式
  `(sf_inner + sf_outer*sf_inner_dim)*128`，起始地址 16B 对齐，且**不支持转置**）。用线性
  scale 布局实测：k=64 全对、k=512 只对一半、n=5120 时只累加了部分 K 块 —— 与文档一致。
  所以委托实现 = ①加载期把权重 block-scale 重排进该 swizzle；②`quantize` kernel 直接按
  swizzle 写激活 scale（或再加一个重排 kernel）；③`TN` + `COMPUTE_32F` + `CUDA_R_32F`
  scale type；④按形状赛跑进 `tuning.rs`。这是有确定规格的工程量，不是未验证假设。
  FP8 那条更简单：`CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F` 正好是"每行 A 缩放 ×
  每行 B 缩放"，与 `record_fp8` 的 per-token × per-channel 一一对应，无需 swizzle。
- **杠杆 B：MTP 融合成一张图 + 动态 lane**。27B 是 4-bit 权重、decode 相对优势却没
  跑赢 2B —— 最可能是三次 replay + 同步 readback 吃掉了量化红利。propose/verify/
  catch_up 一次捕获、设备侧采样与接受判定；按接受率动态调 lane 数。vLLM 的 spec
  开销是其公认弱项，图捕获引擎做这件事结构性更便宜。与 ① 是同一投资的两面：① 决
  定"何时不做"，B 决定"做的时候多便宜"。【推断】
- **杠杆 C：FP8/FP4 KV cache**。hot_long TPOT 1.187 的税在 KV 带宽；FP8 KV 减半注意
  力读流量，目标 <0.9。需过数值门。【推断】
- **杠杆 D：固定开销 55ms → ~10ms**。②A 覆盖准入部分，剩余是 readback/采样/
  detokenize 链路。short 类 TTFT 稳定 <1.0，配合 ③′ 长 prompt 压到 ~1.0。
- **杠杆 E：离线 autotune 常量**。基建已存在 —— `crates/backend/cuda/src/tuning.rs`
  （成对图测量、tile 搜索空间、5e-4 数值匹配容差、胜者须超基线 2% 才采纳、结果持久
  化）；本杠杆是把搜索空间扩展到 `QUANT_GEMM_TILE`、lane 宽度、group size 等更多常
  量，按 GPU 型号在打包期运行。典型 5–15%。IR 层目前只有 validate/plan_lifetimes
  （`foundation/ir/src/dataflow.rs`）、无融合重写 pass；融合以手工 kernel 承担。若未
  来要补，正确形态是闭集 rewrite 规则（RMSNorm+residual、SiLU gate 等），跨后端共
  享 —— 不做通用 auto-fusion 编译器（op 空间是闭集，ROI 为负）。

远期（研究性质，先记账）：prefill/decode 双流共跑（prefill 算力 bound、decode 带宽
bound，SM 分区；现 worker 严格单流）；device-side graph launch 零 host 间隙 decode。

## 三、调度侧路线（生产并发）

前提见 §〇.2：预算单位是 replay 槽位。近/中/远三程：

**近程（直接改 batch4 两格）**

1. 双预算打包：prefill/decode 预算分离；prefill 只在 ITL 余量内发射（布尔决策，
   不切碎）。改 `packing.rs:310-321` 的 min 链与 `policy.rs:186-216` 的 role_bias。
2. 准入微合批窗口：到达后等 2–5ms 攒批，配合 ②A 把"串行准入"变"一批准入"。
3. 懒捕获全部改启动预热：slot 池现在首次连续 ≥2 个 decode 才懒捕获
   （`execution.rs:173-229`），生产上是周期性毛刺源；启动时按容量预捕全部图。
4. ① 自适应 MTP。

**中程（生产并发的命门）**

5. **并发 >4 的 slot 悬崖调查**【待测】：`max_num_seqs=16` vs `CB_DECODE_SLOTS=4`，
   第 5 个并发 decode 起溢出到每请求私有图串行。RM§四 已证明盲目加宽（4→8）更慢
   （8 路 batch 4.10s→6.6s）且根因未查。两条路：查清 8-lane 回退根因【推断：更宽
   图的状态读写/readback 成本】，或分层批（多组 4-lane，请求固定分组）。
6. 前缀亲和调度：按 prefix 树对排队请求聚类，把 hot_long 0.369 的前缀优势从"碰运
   气"变"人为制造"。
7. 长度感知排序：`max_tokens` 提示或预测器做 SRPT 式排序，降 TTFT p99。
8. 饱和降级：准入控制 + 背压 + 优先级类间抢占（restore 图已有 KV 换出底子）；租户
   公平与限速（`policy.rs` 已有 tenant tiebreak）。

**远程（结构性答案）**

9. **PD 分离**（prefill/decode 实例级拆分 + KV 传输）：治 TTFT/ITL 互相干扰的终极
   方案。本后端严格单飞 + 图捕获的架构反而让它更干净（每实例只跑自己的图集，无动
   态形状问题）。②C 是通往它的踏脚石。
10. 重叠调度扩展：`pipeline/scheduling/overlap.rs:23-71` 已有 prepare_next 预规划；
    下一步让准入在 step 飞行中完成（控制通道与飞行解耦），消掉最后的 step 边界量化。

## 四、执行顺序与验证

| 阶段 | 内容 | 验证门槛 |
|---|---|---|
| 0 | §五 观测指标落地；① 的两个测量实验 | 不改代码；产出 batch4 每步 replay 数 |
| 1 | ②A + ②B + ③（kernel 特化，小改动可并行）+ §六 预占阈值与预算瀑布（消费端先只接 KV/前缀池） | 27 host 测试 + 数值门；③ 单 replay ≤28ms |
| 2 | 全矩阵重测（perf-r41），实测替换 §一 推算表 | 已赢格子（2B short/decode、27B hot_long）零回退 |
| 3 | 杠杆 A + B | TPOT 格 ×0.7；MTP 接受长度不降 |
| 4 | ②C lane prefill、③′ 宽度分档（大改动，视阶段 2 结果排期；图资产经 §六 瀑布资助） | ② 的 TPOT ≤5% 回退门槛 |
| 5 | 调度中程项、杠杆 C、E | p99 与饱和负载 soak（过载面见 [serving-stability.md](../guides/serving-stability.md) §五.8） |
| 6 | §七 profile 自适应（auto 默认档） | 同一配置双负载矩阵复测，两侧 ≤5% 差距 |
| 7 | §八 不变量收口（不建 TP，只收口假设） | 8 条不变量逐项过 review；engine 无直接设备引用、动态决策只经 SignalSnapshot |
| 8 | §九 H200 接入与 FP8 三制度（与阶段 2–5 可并行，依赖 H200 机型到位） | mma-probe 证据落盘 + wgmma lowering 数值门 + H200 全矩阵基线 |

统一用 `tools/bench/serve-compare.py` 标准矩阵 + `INFER_CUDA_EXECUTION_PROFILE`
系列开关；任何比值遵守 RM§六（同源重测，不拼来源）。

## 五、观测基建（一切调参的前提）

进 metrics 的一等指标（现状：都没有）：

1. 每图带宽 %（kernel 工作的仪表盘）；
2. 每 token replay 数（① 与 slot 悬崖的直达指标）；
3. MTP 接受长度（① 与杠杆 B 的反馈信号）；
4. 每 step 队列深度与分角色等待时间、ITL 直方图（p50/p99）、KV/arena 占用率
   （调度调参依据）；预算瀑布与各池水位（§六）。

## 六、显存预算：预占阈值（空间换效率）

**共识**：单模型单卡是主要部署形态，显存不折算成能力就是浪费。机制已备（加载期捕获 +
`loading/budget.rs` 准入记账 + 运行期零分配），缺的是"余量 → 能力"的预算策略与一个
运维可调阈值。

**现状**【确证】：`device.rs:96-145` 的预算全是 `total_memory_bytes` 的纯函数 ——
headroom total/32、arena total/64（5090 上 512MiB/请求）、checkpoint total/16、
resident_states total/512MiB clamp [4,64]；消费端被常量封顶（`CB_DECODE_SLOTS=4`、
`CB_SLOT_TOKENS=4096`、`MAX_CAPACITY_TOKENS=32768`）。27B NVFP4 权重 ~14GB 之外，
~15GB 余量闲置：不变并发、不变上下文、不变图资产。

**参数设计**：

- `memory_utilization`（runtime.json / CLI，对齐 vLLM `--gpu-memory-utilization` 语
  义）：单实例可动用总显存比例，**全包口径**（权重 + 图 + 各池），serve 默认 0.90。
- per-model-instance 语义：多模型共卡（如 LLM + embedding）各自显式下调；加载期总
  claim 超额直接报错 —— fail at load，不 OOM at runtime。
- 除阈值外不加旋钮：瀑布优先级固定，避免 vLLM 式旋钮税（参数对齐的原则是能力平价，
  不是旋钮平价）。

**预算瀑布**（加载期顺序）：

1. claim = total × utilization；
2. − 权重（加载实测）；
3. − headroom 下限（`device.rs` 现有派生保留为 floor）；
4. − 图资产与 workspace（按实际捕获计；③′ 分档图、②C lane 图在此入账）；
5. 余量 → 状态预算：先 recurrent 池（目标并发 × 每序列固定字节），其余全部 → KV 页
   池；前缀保留**共享** KV 池（retention 策略，不独立设池）。

**常量 → 派生改造清单**：

- `MAX_CAPACITY_TOKENS=32768` → min(池 ÷ 目标并发, 显式 max_model_len)；
- `CB_SLOT_TOKENS=4096` → 随池派生；
- `CB_DECODE_SLOTS=4` 维持不动，预算只资助 slot **组**（分层批，§三.5），禁止加宽
  lane（RM§四 4→8 回退）；
- arena/checkpoint 除数进 tuning 搜索空间（杠杆 E），不再拍常量；
- `sequence_budget()`（`loading/budget.rs`）保持为准入执行器，改核对口径为 claim 后
  的池。

**27B hybrid 实账**（`examples/qwen3.8-27b/config.json`：64 层 = 16 全注意力 + 48
线性，kv_heads 4、head_dim 256）：

| 项 | 数值 |
|---|---|
| KV | 16 层 × 2 × 4 × 256 × 2B = **64 KiB/token**（FP8 减半） |
| recurrent 状态 | 48 层 × 48 vheads × 128×128 × 2B ≈ **72 MiB/序列**（与长度无关） |
| 256k 上下文 KV | bf16 = 16 GiB；FP8 = 8 GiB |
| 0.90 claim 余量（5090 32GB） | 28.8 − 权重 ~14 − 图/arena ~2.5 ≈ **12 GiB** |

结论一：**对齐 `--context-length 262144` 在 32GB 卡上强依赖杠杆 C（FP8 KV）** ——
bf16 的 16 GiB 放不进 12 GiB 余量。结论二：64 并发的 recurrent 池 = 4.6 GiB，这是
hybrid 模型的固有税，瀑布让它显性化（对应 SGLang `--mamba-full-memory-ratio` 的维
度，但由瀑布自动定，不做旋钮）。

**收益排序与门槛**：

1. KV/前缀池扩容 + 容量帽派生化 —— 现在可做，确定性收益（hot_long 前缀优势放大、
   256k 准入解锁）。指标：前缀命中率（`--enable-cache-report` 对应物）、可准入的
   并发 × 上下文。
2. 图资产扩容 —— 门禁：③ 落地 + ③′ 重测（`loading/mod.rs:246-251` 的实测陷阱：
   上次用空闲显存资助 128-lane 图，四格回退三格；当时是 4x 重读的坏 GEMM）。

**验证**：加载期打印瀑布 + `/native/v1/runtime` 暴露 + 池水位进 metrics（§五.4）；
bench 加三类：多 distinct prefix 的 hot_long 压力、256k 准入选通、并发 1→64 扫描；
现有矩阵零回退。

## 七、部署 profile：默认兼得（auto），可选钉死

**原则**："单机低延迟 vs serving 吞吐"不是用户的选择题。默认 `auto` 档下引擎按
负载信号在延迟/吞吐谱上自动滑动 —— 队列空时自动深投机、准入短路（延迟行为）；
队列涨时自动降 draft 深度、开攒批窗口、prefill 让路 ITL（吞吐行为）。
`latency` / `throughput` 两档只是把曲线端点钉死（SLO 场景用），**不是另一套代
码路径**。

**诚实的边界**：饱和点上真兼得违反物理 —— 高并发时投机 verify 与在跑请求抢算
力，必须让位（① 的自适应 MTP 就是这个让位机制）。"兼得"的定义是**每个负载点
的行为自动正确**，不是同时满足两个极点。

**信号**（全部依赖 §五 观测基建，阶段 0 前置）：队列深度、ITL 余量、MTP 接受长
度、每 token replay 数。

**执行器**（全是运行时策略，无 kernel 改动）：

1. MTP 深度/lane 动态化（① 的推广：阈值 → 连续曲线）；
2. 准入微合批窗口 0–5ms 随队列深度伸缩（§三.2 的参数化）；
3. prefill 发射布尔门（§〇.2 共识的直接应用）；
4. 深投机图/设备侧闭环的启用与挂起。

**迟滞**：所有阈值双侧化 + 最小驻留时间，防止 MTP 深度在边界抖动。

**图资产与显存**：加载期尽量捕获两模式图资产的并集；装不下时瀑布内降级顺序为
slot decode 图（必需）→ prompt 宽度分档（③′）→ 深投机变体 → 设备侧闭环；KV 池
地板优先于投机图。27B/32GB 的 ~12GiB 余量（§六 实账）大概率装得下并集；装不下
的硬件组合靠秒级冷启动做快速重载切换。

**前置依赖**：①（自适应 MTP 机制前身）、②B（`completion_timing` 成本校准）、
§六 瀑布（资产入账）、`CB_DECODE_SLOTS` 加载期派生化（§六 清单）。

**验证**：同一二进制同一 `auto` 配置连跑两类负载 —— batch1 延迟矩阵 + 并发
goodput 矩阵；两侧与各自钉死档的差距 ≤5%，否则自适应曲线不合格。profile 三档
的切换不重编译、不重捕图（并集已捕获时）。

## 八、并行化路线与防重构不变量

**定位**：数据中心主战场。吞吐的第一答案是 DP 不是 TP —— 模型单卡装得下时，
多副本 + 路由零通信开销，永远优先；TP 解决容量（装不下）或延迟（分摊权重字节）；
EP 等 MoE 落地后再谈。

| 阶段 | 内容 | 前提与工作主体 |
|---|---|---|
| 0 | DP + 路由层 | 现可做；实例互不知情，路由按前缀指纹粘会话 |
| 1 | TP=2/4 节点内（NVLink 机型） | rank 运行时壳、图内 all-reduce、加载期分片、hybrid mixer 状态分片、§七 信号全局化 |
| 2 | PP（可选） | 点对点通信最简单，但有流水气泡；吞吐场景与 TP 组合 |
| 3 | EP | 前置 = MoE 本身（grouped GEMM 单卡先行），最远 |

**kv_heads=4 → 本模型高效 TP 上限为 4**（attention 按头分片，TP>4 需复制 KV），
加载期强校验。5090 无 NVLink：TP 的开发/CI/验收需 NVLink 机型，硬件预算先行。

### 防重构不变量（现在做便宜，以后做是重构）

1. **逻辑后端**：engine 只经 `BackendProvider` 与"逻辑后端"对话，任何代码禁止
   假设 1 进程 = 1 设备；TP 时同一 trait 由 rank-group 实现（submit/poll/
   reserve 向各 rank 广播）。排查点：runtime/pipeline 与 executor 的直接设备引
   用。
2. **CollectiveOps 抽象**：图构造中的集合通信走 trait —— 单机 = no-op 直通，
   TP = NCCL（C API/bindgen，同 cuBLASLt 路径）。**单机与 TP 共用同一张图构造
   代码**，collective 节点只是实现不同。这是"不重构"的核心保险。
3. **决策点唯一**：一切动态决策（§七 的 MTP 深度、准入窗口、prefill 门）只读
   `SignalSnapshot`；单机 = 本地采样，TP = rank 归约/rank0 广播。执行器禁止读
   rank 本地状态，否则 TP 下决策发散 = collective 死锁。
4. **分片是加载期视图**：权重命名钩子（`weight_prefixes`/`anchor_slot`）之上加
   shard view（column/row 切 = 形状变换）；provider 对 TP 无感，分片计划由后端
   按 tp_degree 声明进 ExecutionProgram。
5. **KV/状态按 rank 参数化**：heads_per_rank 进 KV manager 参数而非从 config
   直取；mixer 状态分片规则进 IR 的 StateKind 声明（linear attention 按 head、
   conv 按 channel）。
6. **确定性瀑布**：§六 预算瀑布必须是（设备 identity, 模型, 配置）的纯函数 —
   TP 各 rank 独立算出同一结果、图资产集逐字节一致，否则 capture 顺序发散即死
   锁。
7. **配置与观测预留**：runtime.json 预留 parallelism 度数（dp/tp/pp）；metrics
   标签预留 rank 维度；`memory_utilization` 保持 per-instance 语义（DP = N 个
   互不知情的实例）。
8. **DP 前缀亲和钩子**：实例暴露前缀指纹/命中率 API（复用 §三.6 信号）；DP 不
   需要实例间共享任何状态。

**已知冲突点（提前记账）**：

- §七 auto profile 在 TP 下每个动态决策 = 一次全局同步，延迟模式下 TP>1 的投机
  加深收益要打同步折扣。
- 杠杆 B 的图内接受判定天然 rank 一致（logits all-gather 后同 seed 确定性），
  不冲突。
- 单 rank 故障 = 全组 hang（NCCL 超时）：[serving-stability.md](../guides/serving-stability.md)
  需补分布式故障章。

**现在不做**：不建 NCCL、不建 rank shell、不写分片实现 —— 只收口假设。

## 九、硬件与精度矩阵：H200/FP8 数据中心线

**定位**：双硬件线 —— 5090/NVFP4（消费单机）与 H200/FP8（数据中心）。H200 是
sm_90 Hopper，**无 FP4 Tensor Core：NVFP4 在 H200 上没有硬件路径，FP8 即正确精
度**。provider 的 `PrecisionPolicy` 按设备能力选路是现成机制（"支持原生 FP4 与
支持某个高效 Tensor Core tile 是两件事"），不需要新架构。

**现状**【确证】：FP8 GEMM kernel 已存在（`nvfp4_gemm.rs:14-55` 的 `matmul`，
与 `packed` 共用常量），③ 的 4x 重读修复同 PR 镜像到它；Hopper 转换路径存在但
**未实机验收**（cuda-performance.md §能力与精度）；tuning 表按设备 identity 索
引，新硬件零代码接入、首轮自动测量。

**工作清单**：

1. **H200 硬件接入**：`cuda-mma-probe` 能力验收、证据落 `benchmarks/capabilities/`、
   `baselines/tuning.json` 加 H200 基线。CI/bench 机型预算先行（软件就绪度无法
   在无硬件时验证）。
2. **lowering 重验**：wgmma（sm_90a）与 tcgen05（sm_120）是两条 lowering 路径，
   数值门 + `inspect-cutile-cache.py` SASS 检查全量重跑；TMA 在 Hopper 是主场，
   杠杆 A 的双缓冲方向直接受益；PDL sm_90 原生支持（`mlp/pdl.rs` 无需改动）。
2a. **chunked recurrent prefill（本轮实测：奖品很大，但现在是错的）**。27B `long` 用例
   （4×511 token）的 prefill 是 **~900 ms 设备时间、不是主机开销**：`INFER_CUDA_EXECUTION_PROFILE`
   显示 `prefill_chunk` 每块 63 ms（mtp0 也 63 ms，所以与投机/readback 无关），
   `INFER_CUDA_PREFILL_PROFILE` 按算子拆开是 **delta 345.8 ms（38%）+ linear 336.0 ms（37%）**
   + embedding 93.5 + attention 51.9。delta 是逐 lane 读写整块状态：48 层 × 128 lane ×
   3.146 MB × 2 = 38.7 GB/块，38.7 GB ÷ 1.79 TB/s = **21.6 ms/块**，与实测 21.6 ms 完全吻合
   —— 它已经跑在 100% HBM 带宽上，唯一出路是少搬数据（把状态留在寄存器里跨 lane 复用）。

   仓库里正好有这条路（`recurrent_prefill`，状态在寄存器 tile 里跨 lane 迭代），但它对
   **激活量化的模型被关掉**（`input_scales`/`fp8_inputs` 非空即返回空 workspace）。本轮把门
   打开实测：**long TTFT 4.565 → 3.433（−25%）、wall 1.621 → 1.460（−10%）、batch4 TTFT
   1.971 → 1.659（−16%）**，但 **21/21 条 token 序列全部不同**。

   于是用仓库自带的全模型数值门复现：
   `INFER_TEST_MODEL=/home/r/models/Qwen3.8-27B-NVFP4 INFER_TEST_TOKENS=<json> INFER_TEST_PREFILL_WIDTH=128
   cargo test --release -p infer-backend-cuda --features cuda checkpoint_chunk_recurrence_matches_legacy_logits -- --ignored`
   → **`hidden relative_l2=0.0436, max_abs=0.4986`（门限 1e-4 / 1e-2）**。原注释写的是
   "FP32 rounding slightly"，实测是 **4.4% 的语义级偏差**，所以这是正确性门、不是精度取舍。
   二分：把卷积强制回逐 lane（`conv_prefill::record` 返回 false）后偏差**逐位相同**
   （0.04364158），说明卷积路径在该模型上根本没被选中，问题在 **chunked Delta 本身**；
   而合成测试 `chunk_delta_matches_independent_recurrence` 在同一几何（kh:vh=1:3、dim 128、
   lanes 128、count<lanes）下 2e-5 通过 —— 所以缺陷在 capture 喂给 kernel 的东西，不在
   表达式树（两侧逐算子结构一致）。**下一步**：对比 `capture_state.rs:74`（逐 lane：扁平
   1-D view，`[head]` 索引）与 `recurrent_prefill/workspace.rs:111`（chunked：显式
   `view([lanes, 2*KH+VH, D])`，`[lane, head, 0]` 索引）两处输入 view 的布局假设。

   **第 6 轮把范围收窄到"capture 的输入绑定"**。(a) cuTile JIT cache 不背锅：`l2_key`
   是对 tileiras 字节码做 SHA-256，kernel 体一改 key 就变，我这轮的 kernel 编辑都是真生效的。
   (b) 逐个排除：masked lane 写零（去掉后偏差**逐位相同**）、chunked 卷积（把 delta 换回逐 lane、
   卷积保持 chunked → **偏差恰好 0.00000000**，即卷积精确等价）、metadata 布局（chunk 的
   `[position, count, state_offset, 0]` 与逐 lane 的 `state_pos = pos + state_offset` 语义一致）、
   共享 scratch（改成每个 delta 节点独占 scratch，偏差仍逐位相同）、prefill 宽度（32/64/128
   三档偏差完全一样）。(c) **新增测试证明两个 kernel 各自都是对的**：
   `chunk_delta_matches_independent_recurrence` 补上模型几何 (kh16/vh48/dim128/lanes128) 后通过；
   新增 `per_lane_delta_matches_independent_recurrence_at_model_geometry`（此前**只有 chunked
   被测过**，shipping 的逐 lane 路径从没在这个几何上对过参考），worst state 6.9e-8、
   worst out 8.7e-9。既然两边在相同输入下都精确，那 5% 只能来自 **capture 喂进去的输入不同**
   —— 下一步是 dump chunked capture 的 `qkv/beta/alpha/state` 绑定与逐 lane 路径逐值对比。

   **同轮落地的小改动**：MTP 草稿的 priming 过去按固定 `PREFILL_LANES`(32) 切块，
   511-token prompt 变成 16 次 replay/请求（profile 里 64 条 23-node 记录 × 2.1 ms ≈ 134 ms）。
   草稿的 prompt 图其实能捕到 128 宽，现在按 `spec.program.prefill_width()` 切块：
   **long TTFT −1.7%（4 轮 long-only 池化，0.9833）、wall −0.6%，token 4 轮逐字节一致**；
   常驻显存与池化宽度不变（仍 width 4、0 个 disabled step）。收益远小于 134 ms 的原因是
   priming 与其它请求的 prefill replay 在 GPU 上重叠，只有约 0.3 ms/次落在关键路径上。

   **第 7 轮：把"4.4% 漂移"这个数字本身证伪了，并给出真实的分层误差。**
   `checkpoint_chunk_recurrence_matches_legacy_logits` 里两次 `prefill_batch` 之后直接比较
   host 向量；在比较前插一次 `reclaim_barrier()`（device synchronize）后，**单 token 用例
   全部 state 与 readout 都是 `relative_l2 = 0.00000000`** —— 原来的 0.0436/0.0529 是把
   放大后的端到端数字当成了逐层误差。加上同步后逐 state 打印，真实情况是：

   | 位置 | 误差 |
   |---|---|
   | 第一个 delta state（TensorId(25)） | **9.5e-6** |
   | 后续 delta states | 2.4e-3 … 1.7e-3 |
   | 下游 conv states | 2.1e-2 … 6.5e-2 |
   | 单 token、同几何 | **恰好 0.0**（每个 state 与 readout） |

   即：**单 lane 逐位一致，多 lane chunk 才差**，是 lane 循环里的 f32 重结合，经 64 层 +
   激活量化放大成不同的贪心续写（端到端 21/21 序列改变、long TTFT −25%）。这一轮同时确认
   chunked 卷积路径在该模型上**从未被选中**（`weights.constants.contains_key` 命中），
   所以那些"conv state 漂移"是 delta 漂移的下游后果，不是卷积的问题。

   新增的测试资产（都是这轮或上轮加的、可复用的）：
   `per_lane_delta_matches_independent_recurrence_at_model_geometry`（补上 shipping 路径
   从未被测过的模型几何参考）、`chunked_recurrence_matches_per_lane_at_model_geometry`
   （参数化几何的集成 fixture，~2 s 复现）、`chunk_delta_transpose_permutes_head_and_lane`
   （scratch→输出的置换）、以及 model_check 里的 device 同步与 state 一致性诊断。

   **结论**：门继续关着是对的，但理由从"语义级 4.4% 偏差"改成"lane 循环的 f32 重结合"。
   修法是让 chunked kernel 在 lane 循环内逐位复现逐 lane 的 store/load 序列（布局/metadata/
   scratch/transpose 都已对独立参考验证过），而不是找布局 bug。收益仍是最差格 −25% TTFT。

   **第 8 轮：先给出可用的开关与实测，再谈 e2e。** 定位到差异是**纯 kernel lowering**：
   新测试 `chunk_and_per_lane_delta_agree_exactly` 在**逐字节相同输入**上直接跑两个 kernel
   （无 capture、无 program），state `max_abs = 9.3e-9`、output `7.5e-9`（目标 0），
   0.67 s 一轮 —— 这就是修 lane 循环的快门。把 q/k/v/beta 改成与逐 lane 完全相同的扁平
   `[D]`/`[1]` 分区+索引后差异不变，所以不是载入形状，而是循环体本身的收缩/调度。

   在此前提下把 chunked 路径做成**显式 opt-in**（`INFER_CUDA_CHUNKED_RECURRENT=1`，
   默认仍是逐 lane、逐位不变），27B 两轮实测（ratio native/vLLM）：

   | 用例 | wall | TTFT | TPOT |
   |---|---|---|---|
   | long | 1.619 → **1.455**（−10.1%） | 4.513 → **3.411**（−24.4%） | 1.271 → 1.222 |
   | short | 1.144 → 1.240（+8.4%） | 1.274 → **1.086**（−14.8%） | 1.134 → 1.253（+10.5%） |
   | batch4 | 1.593 → 1.582 | 2.312 → **2.079**（−10.1%） | 1.681 → 1.651 |
   | hot_long | 0.950 → 1.009（+6.2%） | 0.369 → **0.330**（−10.6%） | 1.186 → 1.284（+8.3%） |

   即：**TTFT 全面下降 10–24%**（预填充收益），但 short/hot_long 的 TPOT/wall 反而退 ~8–10%。
   后者经 3 轮复测确认不是噪声（short wall 1.084 / TPOT 1.105、hot_long 1.062 / 1.083）。

   **一处纠正**：先前把 TPOT 回退归因于"verify 图也切到了 chunked"是错的。`Workspace::new`
   只在 `batch.rs:364` 的 `build32` 里被调用，而 `build()` 仅在 `width >= PREFILL_LANES` 时
   分派到 `build32`（`batch.rs:265`），所以 chunked 路径**本来就只作用于预填充图**；batch/slot/
   verify 图走 `Workspace::default()`（逐 lane）。因此 short/hot_long 的 wall/TPOT 变化目前
   **没有解释**，是一个待查项，而不是已知机制。

   默认保持关闭的理由因此收窄为两条：token 会变（0/21 与默认一致，属数值取舍）、以及短序列
   的 wall/TPOT 回退尚无机制解释。long 用例则是全面变好（TTFT −24%、wall −10%、TPOT −4%）。

   **第 9 轮：把"修 lane 循环"这条路彻底关掉。** 新增 `chunk_and_per_lane_delta_gap_map`
   按几何逐点测两个 kernel 的差：

   | lanes×count | state gap | output gap |
   |---|---|---|
   | **1×1** | **2.98e-8** | **7.45e-9** |
   | 2×1 / 2×2 | 2.98e-8 / 5.96e-8 | 7.45e-9 |
   | 128×1 / 128×51 | 2.98e-8 / 9.31e-9 | 7.45e-9 |

   **单 lane、单 token 就已经差**，所以不是循环carry、不是 masked lane、不是 chunk 长度。
   逐元素统计（`INFER_GAP_INDEX=1`）：1×1 时 **6144 个 output 里 4634 个、786432 个 state 里
   59412 个**在末位不同 —— 大面积末位差，即同一个表达式树在 cuTile 里被收缩/调度成不同指令。
   已验证与载入形状无关（把 q/k/v/beta 换成与逐 lane 完全相同的扁平 `[D]`/`[1]` 分区+索引，
   数字一模一样）。**结论：这不是 kernel 源码能修的 bug，需要编译器级开关（例如禁止 FP
   contraction），或让两条路复用同一个 kernel。** chunked 递推因此作为显式数值选择保留
   （`INFER_CUDA_CHUNKED_RECURRENT=1`），默认仍是逐 lane。

   收益已实测并记录：long TTFT −24%、wall −10%、TPOT −4%，各用例 TTFT 全面 −10~24%；
   short/hot_long 的 wall/TPOT 回退 8–10%（机制仍未解释）。

   **第 11 轮：委托到底值不值，先量再接线。** `resident/prefill_gemm/bench.rs` 把
   "cast + cuBLAS" 与现在的 prompt GEMM 放在同一张图里对打（2B 的真实形状，30 次重放，
   含必需的一次热身）：

   | 形状 (m×n×k) | 我们的 kernel | cast + cuBLAS | 比值 |
   |---|---|---|---|
   | 128×2048×2048 | 18.1 µs | 5.2 + 15.5 = 20.7 µs | **0.88×（我们更快）** |
   | 128×12288×2048 | 95.9 µs | 5.2 + 41.5 = 46.7 µs | **2.05×** |
   | 128×2048×8192 | 55.5 µs | 6.0 + 33.8 = 39.8 µs | **1.40×** |

   结论是**按形状分档**：宽 N（MLP 的 gate/up）委托赢 1.4–2×，方阵（attention 投影）打平，
   差的那一项恰好被"先把 f32 激活收窄成 bf16"这一步吃掉 —— 该步在 tile=1024 时 5.2 µs，
   tile=8192 时 8.1 µs，是纯粹的带宽/占用问题。因此接线规则应当**按权重规模分档**
   （≈ ≥8M 元素才委托），或者更稳的做法是在热身阶段对每个形状真跑一次赛跑再缓存胜者。

   注意 cuBLAS 要求两个操作数同类型，所以激活必须先转 bf16；同一层里 q/k/v 共享一份
   normalized hidden、MLP 的 gate/up 也共享，所以**每层转一次**可把 5.2 µs 摊到 2–3 个投影上，
   这是接线时的正确形态。

   **第 12 轮：接线完成并端到端实测。** 委托按第 11 轮的分档规则接进
   `prefill_projection::record`（`Support::record_dense`，权重 ≥8M 元素才委托），
   workspace 照 `nvfp4` 的模式挂在 `BatchBuilder`/`DeviceProgram` 上，激活按元素数惰性分配
   bf16 scratch。开关 `INFER_CUBLAS_PROJECTIONS=1`，默认关闭。

   接线中发现一个**必须遵守的约束**：cuBLAS 首次遇到某个 configuration 会做 host 侧工作
   （选 workspace/算法），捕获流会拒绝，所以 `DeviceProgram::new` 里先用 `warm_delegated`
   把每个 (width, n, k) 在设备流上跑一次再捕获图。否则请求直接失败：
   `operation not permitted when stream is capturing`。

   2B 两轮交错实测（ratio native/vLLM）：

   | 用例 | wall | TTFT | TPOT |
   |---|---|---|---|
   | short | 0.972 → 0.975 | 1.015 → **0.914**（−10.0%） | 0.973 → 0.983 |
   | long | 1.137 → 1.111 | 3.190 → **2.925**（−8.3%） | 0.983 → 0.979 |
   | batch4 | 1.430 → 1.428 | 2.151 → **1.921**（−10.7%） | 1.354 → 1.374 |
   | hot_long | 1.093 → 1.090 | 1.882 → **1.814**（−3.6%） | 1.031 → 1.032 |

   **TTFT 全面下降 3.6–10.7%**，wall/TPOT 基本不动（委托只作用于 MLP 的宽投影，attention 的
   方阵按测量被排除）。代价是数值：**21 条序列中 16 条与默认逐位相同，5 条不同**（cuBLAS 的
   累加顺序不同）。因此与 chunked 递推一样保持 opt-in，默认路径不变。

   **需要用户裁决的一点**：目前"性能改动必须 token 逐位一致"是仓库自己设的验收线（见
   `docs`），但两个已验证的加速（chunked 递推 −24% long TTFT、委托 −4~11% TTFT）都过不了
   这条线。若改为"logit 级相对误差 + 抽样一致性"这类数值门禁，它们就能成为默认路径。

   **第 13 轮：一个不改数值的加速（27B long TTFT −4.3%）。** 先给两个模型做了分桶剖析
   （2B：linear 53%、attention 32%；解码每 token 要读 4 GB bf16 权重，实测 3.1 ms/token
   对带宽下限 2.2 ms，vLLM 也在同一水平，没有余量）。真正可下手的是**提示图宽度**：

   仓库里已有的记录（`loading/mod.rs`）本来就写着"128-lane 提示图与两个 64-lane chunk
   **数值完全一致**，并把八次 replay 变成四次（long TTFT −13%）"。因为 key/value 一律过 KV
   cache、prompt GEMM 逐元素只在 K 上累加，所以**加宽 chunk 是数值中性的**。于是把
   `MAX_PREFILL_LANES=128` 之上加了一级 `WIDE_PREFILL_LANES=256`：

   - 27B（量化 recurrent）：`prefill_width=256, narrow=64`（arena 361/503 MiB），
     **long TTFT 0.3061 → 0.2929（−4.3%）、wall −1.0%**，两轮 **3/3 序列逐位相同**。
   - 2B（dense）：256 实测**无收益**（long TTFT 0.999、short 1.007），因此 dense 模型的
     阶梯仍止于 128，只有量化 recurrent 模型才拿这一级——这也解释了为什么原先的注释里
     只有 64→128 有收益。

   宽度选择逻辑抽成了 `select_prompt_width`，并写明"arena 是真正的约束，这个 cap 只是阶梯
   上限"。这轮没有触碰任何数值路径，默认矩阵因此净赚。

2b. **DFlash2 草稿模型（已调研 + 已下载，未实现）**：`z-lab/Qwen3.8-27B-DFlash2`
   （1.924B / 81 个 BF16 张量 / 3.849 GB，已下载到 `/home/r/models/Qwen3.8-27B-DFlash2`
   并校验张量可读）。结构：`fc.weight [5120,25600]` 把 **5 个目标层
   （`target_layer_ids [5,19,33,47,61]`）的 hidden 拼接**投回 5120；5 层 Qwen3 骨干
   （32 heads / 8 kv heads、head_dim 128、滑动窗口 2048、**块内非因果**）；每层两组
   "双抽头动态卷积" `attention_conv`/`mlp_conv`（`base_kernel [2,2,5120]` +
   `kernel_projection [1280,5120]`，即 conv_kernel_size 2 × conv_group_size 16）；
   `candidate_selector`（`hidden_projection [256,5120]` + 两个 `[248320,256]` 码本）在选择
   器里从 top-16 候选中追踪一条连贯路径。它**没有自己的 embedding / lm_head**，复用目标
   模型的。`block_size 8` 意味着**每条序列 8 条 verify lane**。

   **前置阻塞已定位（实测）**：我们现在的 verify 宽度上界不是算力而是**显存**。MTP 深度
   扫描（27B，2 轮，ratio native/vLLM-mtp2）显示 depth 2 最优（short wall 1.140 /
   TPOT 1.133），depth 3 起 batch4 崩塌（wall 1.110 → 2.679、TPOT 1.775 → 3.715），
   而 short 几乎不动——这是**池化被静默关闭**：`INFER_CUDA_EXECUTION_PROFILE` 记录
   `pool_creation_failed`，depth 3 时 width 4 报 `CUDA driver error: out of memory`，
   width 3/2 报 `Capacity: resident F32 state exceeds available device memory minus 1 GiB
   headroom`，于是 205 个 step 全部 `pool_disabled`，四条序列退回串行。27B 常驻已 22.9 GiB
   / 预算 27.7 GiB，恢复池化所需的 checkpoints 是按 lane 逐份的 F32 状态快照。

   **显存账（实测 + 解析）**：27B 是 64 层 = 48 层 `linear_attention` + 16 层
   `full_attention`，Delta 状态每层 48 个 value head × 128 × 128 × **F32** = 3.15 MB，
   48 层合计 **151 MB/序列**。池化回滚快照（`slot_verify.rs:51 slot_checkpoints`）
   只快照 `Conv | LinearAttention`，数量 = `slots × (verify-1)`：
   verify 3 → 4×2×151 MB = **1.21 GB**，verify 4 → **1.81 GB**，
   **verify 8（DFlash2）→ 4×7×151 MB = 4.23 GB**。而 27B 常驻后只剩
   27.7 − 22.9 ≈ **4.8 GiB**，还要容纳 4 份 slot 状态、arena、图和（DFlash2 的）
   3.85 GB 草稿权重。**8 lane 的 verify 仅回滚快照一项就超过全部余量** —— 这就是
   depth 3 起池化被关掉、以及 DFlash2 在本机不可行的真正原因；KV 不是主项
   （16 层 full attention，capacity 128 时约 16 MB，且有 `kv_scales` 时根本不在此分配）。

   **本轮实测（含一个已修的真 bug）**：一次失败的池化尝试**不会把内存还回来** —— 失败前
   `free 2476 MiB / pool 6 MiB`，失败后 `free 44 MiB / pool 2438 MiB`。内存进了 device
   memory pool，而 `cuMemGetInfo` 把 reserve 未用的块算作已用，于是 width 3/2 的重试在
   "free 只剩 44 MiB"上做准入判断，必然失败，池化被永久关闭。同时 `reclaim_cached_for_pool`
   只认 `Capacity`，而 driver OOM 是 `Backend`，所以 OOM 之后一次回收都没做。

   落地：回收条件扩到 `Capacity | Backend`；新增 `CudaDevice::pool_reclaimable_bytes()`
   （pool reserved − used）供诊断；两处拒绝信息带上 MiB；池化被放弃时 warn 出最后一次失败
   原因。**试过但撤回的两条路**：(i) 把 `available_memory_bytes = free + pool 可回收` 用于
   准入 —— 语义正确，但实测变成"准入通过、真分配 driver OOM"，depth 3 时好时坏，故撤回
   以免把干净的 Capacity 拒绝变成硬失败；(ii) `set_release_threshold(0)` —— 驱动是惰性
   trim，`cuMemGetInfo` 仍然显示 ~0 空闲，无效。

   **结论（本机硬约束）**：verify 4 的 4-slot 池需要约 2.9 GB 而当时只有 2.5 GB，回滚快照
   就是主项（`slots × (verify−1) × 151 MB`）；即使勉强降到 width 2，4 条序列只有 2 条能
   入池，实测 depth 3 的 batch4 仍输给 depth 2（TPOT 3.64 vs 1.80，wall 2.685 vs 1.093）。
   **DFlash2 的 verify 8 需要 4×7×151 MB = 4.23 GB 快照，在这台 31.4 GB 卡上不可行** ——
   要它成立，必须先让每个 lane 的状态快照不再是 151 MB F32（接受后重建 / 降精度 / 缩小
   Delta 状态本身），而不是继续调池化参数。基准配置 mtp2 完全不受影响：token 逐字节一致，
   short/long/hot_long 比值变化 <1%。

   因此 DFlash2 的顺序是：**(1) 降 verify 的 per-lane 状态成本**（接受后重建设为默认）；
   (2) 独立 checkpoint 加载（草稿是另一个 package）+ 草稿量化（3.85 GB → ~1 GB）；
   (3) DFlash2 IR：fc 融合、5 层双抽头卷积、块内非因果注意力、目标层 hidden 抽头、
   草稿侧 KV cache；(4) 选择器（码本打分 + 路径遍历）；(5) 块草稿 + 8-lane verify 集成与
   实测。**(1) 之前，(3)-(5) 做完也无法在本机跑起来。**

3. **FP8 精度制度补齐**：128×128 block-scaled（DeepSeek 系与 Flash-Next MTP
   expert 的事实标准）、per-tensor（PLE）、per-channel 三制度进 loader + kernel；
   **FP8 KV（杠杆 C 在本线提前）** —— 141GB 卡 + FP8 KV 是长上下文数据中心标
   配。

   **进展：block-128×128 已落地并实测**（`Qwen3-VL-2B-Instruct-FP8`，vLLM 原生
   `fp8` scheme：`weight_block_size [128,128]`、`weight_scale_inv`、动态 per-token
   量化）。loader 侧 `normalize_fp8_scheme` 把 vLLM scheme 翻译成内部形状，
   `WeightEncoding::Fp8Block` 绑定 `weight_scale_inv`，并把每个通道块的 scale 沿
   行展开，kernel 因此与 per-channel 路径同形；kernel 侧
   `fp8_gemm::quantize_block`（每 128 列一个 scale）+ `matmul_block`（每个 K step
   的部分积先乘 `xs[行,k块] * ws[通道块,k块]` 再累加——缩放操作数会二次过 FP8 舍
   入）。数值门 `fp8_block_mma_matches_independent_block_quantization` 对独立构造
   的 per-128-block 参考（f64、真值 scale）在 (12,65,512)/(64,129,1024)/(12,256,5120)
   三形状通过。2B FP8 矩阵（native/vLLM，2 轮）：short wall 1.239 / TTFT **0.749**
   / TPOT 1.299；long 1.496 / 3.085 / 1.312；batch4 1.666 / 1.251 / **1.750**；
   hot_long 1.457 / 2.093 / 1.393。**历史记录：token 与 vLLM 仅 3/21 逐字节一致，存在权重处理差异**：vLLM 在
   sm_120 上选 DeepGemmFp8BlockScaledMMKernel 且 `is_deep_gemm_e8m0_used()` 为真，
   `requant_weight_ue8m0_inplace` 会**用 checkpoint 的 fp32 scale 反量化、再用
   `per_block_cast_to_fp8(..., use_ue8m0=True)` 重新量化并把新 fp8 权重与 2 的幂
   scale 原地写回**（`fp8_utils.py:881-945`）。在本 checkpoint 上抽 12 个投影实测：
   权重相对 Frobenius 变化均值 **2.67%**、最大 2.73%，单权重变化中位数 **2.19%**
   —— 即 vLLM 评估的是"同一份权重的另一种量化"，多一次 e4m3 舍入 + 2 的幂
   scale 约束；native 的加载路径保留 checkpoint 的权重与 scale。
   **2026 年 10 月 9 日审查补充**：保留权重不能证明整模型算术误差为零，抽样权重变化也不能证明所有 token 差异均由再量化导致。本轮未取得原始报告复核历史 dispatch；仍需固定上下文的独立 logits/state 对照与任务质量验收，不能仅据该差异排除实现问题。具体边界见 [对比方法审查](../reviews/vllm-benchmark-methodology-2026-10-09.md)。
4. **委托赛跑扩到 FP8**：cuBLASLt FP8 在 Hopper 极成熟，杠杆 A 的成对测量在
   H200 上预期更多形状判给 cuBLASLt —— 这不丢脸，是赛跑机制按设计工作。
5. **§六 瀑布按 141GB 重算**：27B FP8 权重 ~27GB，KV/并发余量与 5090 完全不同
   量级；瀑布是纯函数，重算即得，但验收矩阵要补 H200 行。

**竞争判断**【诚实】：H200+FP8 是 vLLM 最强主场（FlashInfer/FP8 路径全部战
验），不要指望复刻 5090 线的领先故事。本线目标是**入场券** —— 不在 H200 上就
不在数据中心。差异化指望：MTP 融合图（杠杆 B）、§七 auto profile、单二进制运
维形态，而不是 kernel 裸速。

**与 Flash-Next 的衔接**：该模型官方姿势 TP=8×B200；FP8 checkpoint（推断
~135GB+）在 H200 上单卡 141GB 是极限、TP=2 稳妥 ——  MoE 前置清单见 §八 阶段 3
与上一轮评估（grouped GEMM → 混合精度 loader → Qwen Sparse Attention → PLE →
expert offload）。

## 十、2026-10-09 B3 基线后的实测归因

本节记录首批服务基线（`2b-mtp0-serving-v1`、`27b-mtp0-serving-v1`、`27b-mtp2-serving-v1`）
签发时的实测归因，用于替换此前的推算锚点。比率方向统一为 **native / vLLM**。

### 服务层结论【确证】

| profile | short | long | batch4 | hot_long | long TTFT | batch4 TTFT |
|---|---:|---:|---:|---:|---:|---:|
| 2b-mtp0 | 1.03–1.06 | 1.24–1.26 | 1.56–1.58 | 1.17–1.22 | 3.4–3.5 | 1.8–2.3 |
| 27b-mtp0 | 1.07–1.12 | 1.28–1.29 | 1.35–1.41 | 1.09–1.13 | 3.9–4.1 | 1.7–2.0 |
| 27b-mtp2 | 1.26–1.31 | 1.70–1.73 | 3.50–3.56 | 1.05–1.13 | 4.2–4.5 | 7.3–8.1 |

两个独立配对单位结论一致；单位间最差格漂移 12.0–18.6%，接近阈值的格目前不可分辨。

启动时间不属于以上任何数字：vLLM 13.8–39.6 s，native 3.3–5.1 s。vLLM 运行时的整机负载与
风扇主要由这段启动（CUDA graph capture、inductor 编译、按 0.88 分配 KV）造成。

### 硬件归因【确证】

同一次测量内（active window，仅工作负载运行期间）：

| profile | native GPU 利用率 | vLLM GPU 利用率 | native 进程 CPU | vLLM 进程 CPU |
|---|---:|---:|---:|---:|
| 2b-mtp0 | 20.0–20.5% | 10.3–20.0% | 91–92% | 42–44% |
| 27b-mtp0 | 62.3–63.6% | 56.4–65.9% | 98–99% | 26% |
| 27b-mtp2 | 54.8–55.0% | 34.9–41.6% | 98% | 30–38% |

**native 的 GPU 利用率不低于 vLLM**，因此"GPU 没吃满"不是差距来源；差距是每步的效率与
主机侧成本。native 同时占满一个 CPU 核，而 vLLM 只用 26–44%。

### 逐算子设备时间（27B、mtp 0）【确证】

`INFER_CUDA_PREFILL_PROFILE` + `INFER_CUDA_PROFILE_GRAPH_ONLY=1`（仅图边界事件，避免逐节点
事件自身的开销），同一 profile 的完整矩阵：

| 图 | 每次 replay | 设备时间占比 | 主要算子 |
|---|---:|---|---|
| `slot_decode` | 18.69 ms | 5514 ms | linear 17.3 / delta 1.0 / norm 0.8 ms |
| `prefill` | 142.95 ms | 2573 ms | delta 58.3(41%) / linear 41.3(29%) / attention 37.3(26%) |
| `prefill_last` | 51.21 ms | 1434 ms | linear 24.5(45%) / delta 19.3(36%) / attention 5.4 ms |

`INFER_CUDA_EXECUTION_PROFILE` 同一轮给出的主机侧单次往返：`prefill_step` 16.09 ms × 761 次，
`prefill_chunk` 107.95 ms × 38 次。该数值是**包含等待设备的同步往返**，因此不是叠加项——
它也说明解码路径上主机回读**没有**成为关键路径。

并发确实被合并：181 次调用带 4 条 decode lane（batch4 真批处理），其余单条来自单请求用例。

### 与内存 roofline 的对比【确证】

27B NVFP4 权重 21.81 GiB（23.42 GB）；RTX 5090 峰值带宽 1.79 TB/s。

- 读一遍权重的下限：**13.07 ms/token**。
- native `slot_decode` 实测 18.69 ms → **1.43× 下限**。
- 服务层 long 用例：native ≈20.8 ms/token，vLLM ≈17.3 ms/token（1.32× 下限）。

结论：解码已经接近带宽墙，两者相差约 1.2–1.3×，**不是数量级差异**；真正的倍差在
prompt/prefill 与并发准入上（long TTFT 3.9–4.5x、batch4 1.35–3.56x）。

### 优先顺序【推断】

1. **Delta 的 prefill 占用率**（`recurrent_prefill::delta`，见 §一 与本节 prefill 表）：
   每个 CTA 为每个 value head 串行处理 LANES 个 token，每次迭代读写整个 D×D=64 KB 状态，
   网格只有 48 个 CTA（170 SM）。把 value 维切成 S 块是**数值精确**的（列之间互不依赖，
   归约只发生在 key 维），网格变成 48×S，这是当前 prefill 41% 占比的直接杠杆。
2. **每 token 的 replay 数**（§〇 结论 1）：prefill 单次 replay 成本与 chunk 内 token 数无关，
   因此"权重一遍过、token 成批过"仍然成立。
3. 解码侧：把 greedy 采样搬到设备端、只回传 token，可去掉每步 993 KB 的整词表 D2H 与
   一次 `to_vec` 主机拷贝；但它不是当前关键路径，收益应低于上面两项，排在后面。

复现命令（必须在受保护范围内）：

```sh
bash tools/bench/safe-run.sh env \
    INFER_CUDA_PREFILL_PROFILE=/tmp/ops.jsonl INFER_CUDA_PROFILE_GRAPH_ONLY=1 \
    INFER_CUDA_EXECUTION_PROFILE=/tmp/steps.jsonl \
    python3 tools/bench/serve-compare.py --engine native \
    --model /home/r/models/Qwen3.8-27B-NVFP4 --executable target/release/infer \
    --inputs artifacts/workloads/Qwen3.8-27B-NVFP4-inputs.json --output-dir artifacts/b3 \
    --mtp 0 --tokens 64 --gpu-memory-utilization 0.88 --max-model-len 65536 --max-num-seqs 16 \
    --run-id <新 ID> --profile-id 27b-mtp0
```

### 十.1 值维分块的 A/B 与真正的瓶颈【确证】

按 §十 的推断 1，实现了 per-lane Delta 的 value 维分块：状态列互不依赖（归约只在 key 维），
把 `state`/`out` 按 `[1, D, D/S]`、`[1, 1, D/S]` 分区即可，网格从每 value head 一块变成 S 块。
切换用 `INFER_CUDA_RECURRENT_VALUE_SPLIT`，默认 1（保持既有行为）。

**数值**（`per_lane_delta_matches_independent_recurrence_at_model_geometry`，kh=16、vh=48、D=128
的 f64 独立参考）：split 1 最差 state 6.90e-8 / out 8.67e-9；split 4 为 6.71e-8 / 1.04e-8；
split 4 相对 split 1 的差为 state 2.98e-8 / out 7.45e-9，与既有 chunked-vs-per-lane 的
9.31e-9 / 7.45e-9 同量级，属于 cuTile lowering 的末位差，不是数值口径变化。

**性能：两个核都没有收益。** 27B、mtp 0，同一 profile 的完整矩阵（图边界事件）：

| 内核 | split | `prefill` 每次 replay | `prefill_last` | `slot_decode` |
|---|---:|---:|---:|---:|
| per-lane | 1 | 143.60 ms | 51.51 ms | 18.71 ms |
| per-lane | 4 | 147.36 ms | 52.74 ms | 18.76 ms |
| chunked | 1 | 143.98 ms | 51.46 ms | 18.70 ms |
| chunked | 4 | 147.64 ms | 52.86 ms | 18.68 ms |

网格从 48 块变到 192 块（170 SM）却不变快，说明这个核**不是**占用率受限。逐节点画像还显示
27B 的 prefill 图每层只有 **一个** delta 节点（1154 节点 / 64 层 ≈ 18 个/层，delta 占 1 个），
即实际走的是 chunked 核，因此第一次只改 per-lane 核时测到的"无收益"同时意味着改错了核；
补上 chunked 核后仍是 +2.5%，两条路径一起否定了占用率假设。

**真正的瓶颈是 prompt 图的节点数和每节点开销。** `INFER_CUDA_PREFILL_PROFILE` 的逐节点输出：
32-lane 的 prefill 图有 **1154 个节点**，单次 replay 119.3 ms，即 **每节点 103 µs**；其中 delta 占
61.28 ms。按 64 层换算，per-lane 的 prompt token 每层约 **70 µs**，而 vLLM 的 511-token prompt
TTFT 约 150 ms、即每层每 token 约 5 µs——**差在每层每 token 的固定开销，约 15×**，不在带宽。

**下一次要测的是 replay 次数，而不是核内并行度。** `prefill` 图在整轮矩阵里只 replay 18 次
（`prefill_last` 28 次），即一个 prompt 基本由 1–2 次 replay 处理完；`long` 的 TTFT 248 ms 对应
511 token，约 0.49 ms/token，而 vLLM 同负载约 63 ms、即 0.12 ms/token。所以下一步先量清楚
「每个 prompt 长度用了多少次 replay、每次 replay 覆盖多少 token、以及单次 replay 里各节点占比」，
再决定是合并窄算子、加宽 tile，还是减少 replay 次数；在这些数据之前不再改核。

§十 推断 1 作为优化方向被这次 A/B 否定。分块代码保留：split=1 时是原有路径，数值已验证
（chunked 在 4 组几何上 split 4 相对单块最差 2.98e-8 state / 1.49e-8 out，per-lane 为
2.98e-8 / 7.45e-9，都在 f64 参考的 2e-5 之内），后续若要换 lowering 或换 tile 宽度可直接复用。

### 十.2 prompt 路径的逐算子定位【确证】

按 §十.1 的结论，把每个 case 单独成 workload 跑一遍（这样 replay 能归属到 case），27B、mtp 0、
release 制品，图边界事件。权重一遍的物理下限是 **13.07 ms/replay**（21.81 GiB / 1.79 TB/s）。

| case | 图 | 每次 replay | replay 次数 | 相对下限 |
|---|---|---:|---:|---:|
| short | `prefill_last` | 36.4 ms | 4 | 2.8× |
| long | `prefill` | 112.8 ms | 4 | 8.6× |
| long | `prefill_last` | 119.8 ms | 4 | 9.2× |
| batch4 | `prefill_last` | 36.1 ms | 16 | 2.8× |
| batch4 | `slot_decode` | 18.8 ms | 281 | **1.44×** |
| hot_long | `prefill` | 153.3 ms | 14 | 11.7× |

**解码已经贴住带宽墙（1.44×），prompt 路径差 2.8–11.7×**，这与服务层"只有 decode 的格子在 1.1×
附近、含 prefill 的格子 1.3–4.5×"完全一致。

`long`（511 token，8 次 replay、9236 个节点、951.4 ms 设备时间，每节点 103 µs）的算子构成：

| 算子 | 设备时间 | 占比 | 节点数 | 每节点 |
|---|---:|---:|---:|---:|
| **delta** | 462.3 ms | **48.6%** | 384 | **1204 µs** |
| **linear** | 335.6 ms | **35.3%** | 3972 | 84.5 µs |
| attention | 71.1 ms | 7.5% | 128 | 556 µs |
| gated_norm | 24.0 ms | 2.5% | 384 | 62.4 µs |
| norm | 16.0 ms | 1.7% | 1288 | 12.4 µs |
| conv | 11.0 ms | 1.2% | 384 | 28.7 µs |
| 其余（add/multiply/silu/rope） | 26.5 ms | 2.8% | 2432 | ≤17 µs |

两个可归因的靶子：

1. **delta：48.6%，每节点 1.2 ms。** chunked 核每块循环 LANES 条 lane，状态已在寄存器里跨 lane 复用，
   每 lane 的全局流量只有 q/k/v（各 D 个 f32）；按 128 lane 折算每 lane 约 9.4 µs，而每 lane 的状态
   流量下限只有 0.07 µs。§十.1 已证明加块无效，所以下一步要么降低每 lane 的依赖链延迟（例如把
   `q/k` 的加载与上一层重叠、把 reduce 换成 warp 级实现），要么换算法；在拿到核内计数前不动它。
2. **linear：35.3%，每节点 84.5 µs**，而 M=128、K=N=5120 的单次投影权重 52 MB、下限约 29 µs，
   即偏离 2.9×。这是按形状选 tile 的问题，仓库已有 `prefill_gemm/bench_check.rs` 可直接量。

另外，整图**每节点 103 µs** 这个量级本身说明：prompt 路径的成本主要由"节点数 × 每节点固定开销"
决定，1154 节点/次 replay 的规模下，任何节点级浪费都会被放大 1000 次；这也是为什么先把每节点的
占比量清楚比继续调并行度更重要。

## 十一、2026-10-10 Stage A/B：并发准入计价与同步消除

§十 的归因把 batch4 3.50–3.56x 指向"并发准入"，本节记录当天据此落地的两批改动与实测。
比率方向统一为 **native / vLLM**；vLLM 侧在**同机同驱动**（615.78.08）上重测
（`stage-b2-mtp2-vllm-g1`），因为 10-09 签发基线是 615.71.09，硬件身份已不匹配。

### 11.1 症结：池化序列被按"私有验证图"计价【确证】

`CB_DECODE_SLOTS=4` 的槽位池本该让 4 条并发序列共享一张验证图，但准入计价对所有序列
一视同仁地收了"私有 verify 图 + 私有 wide prompt 图"的账。结果是：**第 4 条请求进不了
准入，落回串行私有路径**，而它又与 3 条池化序列抢同一块显存，于是"每步只跑 1 条序列
（blocked=3）"——batch4 的 7.3–8.1x TTFT 就来自这里，不是内核慢。

### 11.2 Stage A：按实际用途计价

| 位置 | 改动 |
|---|---|
| `executor/state.rs` | 新增 `private_verification_for`：只有当池已满（`pooled >= pool.width()`）或形状不适合池化时才按私有图计价 |
| `loading/budget.rs` | `sequence_budget(capacity, readout, private_verification)`：池化序列不再为 verify 回滚快照/批工作区付账 |
| `loading/mod.rs`·`resident/program.rs` | 新增 `sequence_pooled`/`new_pooled`/`graph_policy`/`forfeit_verification`：池化程序不捕获私有 verify 图；显存紧张时也不捕获 wide prompt 图 |
| `resident/capture.rs` | `record_program(graph, skip_logits)`：prefill 图把 logits 读出留给它的 `_last` 孪生图 |
| `executor/execution.rs` | `warm_slot_pool`：装载期就捕获槽位池，第一条并发请求即可批处理（代价是启动 +≈2.7 s） |
| `constants.rs` | `CB_SLOT_TOKENS` 4096 → 2048：池是装载期捕获的，它的 fp8 KV 与并发序列争物理显存；4096 行时池 + 4 条序列超卡 |

### 11.3 Stage B1：把"浪费的同步"换成流序

单流执行器里，host 只在真正要读数据时才该同步。原先 catch_up 的"零拷贝 readback"、
commit 的部分接受 restore 都是纯同步浪费：

| 位置 | 改动 |
|---|---|
| `device.rs` | `copy_h2d_pinned`：从 pinned 暂存区入队的异步 H2D，不在这里同步 |
| `resident/batch.rs` | `stage_lanes`（校验+metadata staging 抽出）、`replay_detached`（state-only 图 `async_on`，不保留 future）、`barrier()` |
| `resident/slot_batch.rs` | 每 lane 的 pinned 上传环（`EXTERNAL_UPLOAD_RING = MAX_VERIFICATION_WIDTH + 1`，无同步链内不得覆写）+ `SlotPool` 的 `Drop` barrier |
| `resident/slot_batch/draft.rs` | `upload_external`/`upload_lanes`/`run_external_detached`：catch_up 走无同步链 |
| `resident/slot_verify.rs` | commit 的 restore 改 `async_on`；`barrier()` 供池teardown 排空 |
| `executor/drafting.rs` | catch_up 改用 `run_external_detached` |

### 11.4 实测（2026-10-10 23:14，`mine-mtp2-native-g1`）

| case | ttft | tpot | wall |
|---|---:|---:|---:|
| short | 1.85 | 1.58 | 1.60 |
| long | 6.10 | 1.44 | 1.92 |
| batch4 | 2.64 | 2.07 | 2.10 |
| hot_long | 0.42 | 1.28 | 1.05 |

- **Stage A 命中目标**：batch4 TTFT 668 → 236 ms（7.3–8.1x → 2.64x），batch4 wall
  3.50x → 2.04–2.10x；4 条序列同时驻留。
- **Stage B1 收益很小**：TPOT 只降 1–3%（batch4 18.96 → 18.34 ms），落在单位间漂移内。
  说明 10-09 记的那些 host 同步原本就不在关键路径上，§十.2 的"主机回读没有成为关键路径"
  在这里第二次被证实。
- 三次复跑（21:33 / 21:46 / 22:49 / 23:14）互为复现，单格漂移 ≤4%。

### 11.5 未决项一：native 绝对 TPOT 比 10-09 基线慢 4–25%【待归因】

同一 profile、同一命令行、同一模型，10-09 签发基线（rev `2bda883`）与今天：

| case | tpot 10-09 | tpot 今天 | Δ | vLLM 10-09 | vLLM 今天 |
|---|---:|---:|---:|---:|---:|
| short | 10.85 | 13.5 | +25% | 8.66 | 8.59 |
| long | 12.32 | 13.0 | +6% | 9.07 | 9.10 |
| batch4 | 16.66 | 19.0 | +14% | 9.38 | 9.19 |
| hot_long | 11.42 | 11.9 | +4% | 7.85 | 9.26 |

排除项：SM 时钟（2833 vs 2841 MHz）、功耗（205 vs 208 W）、进程 CPU（97.8% vs 97.4%，
同为一核占满）、`config_readback` 的调度参数逐字相同；`2bda883..HEAD` 之间**没有任何**
改动 CUDA 解码路径的提交（只有测试搬迁、文档、`fdbcb89` 的 workspace 依赖声明——Cargo.lock
无 diff、一次 refuted 实验的 Delta value-split 且默认 split=1）。

因此嫌疑集中在**环境**：驱动 615.71.09 → 615.78.08，或 cuTile 运行期 JIT 的产物变化——
vLLM 侧绝对数字不动支持这一点（它的 kernel 是预编译的，我们是 JIT/lowering 出来的）。
在解释清楚之前，**跨驱动的绝对数字不可比**，只有同驱动的成对比率能用；签发新基线前必须
先复现/推翻这条。

### 11.6 池化程序放弃 wide prompt 图的代价，以及为什么换不回来【确证，已否】

`graph_policy` 为了 4 条池化序列能同时驻留，让池化程序只保留 narrow prompt 图
（`wide_prompt = private_verification || narrow_prefill_width < PREFILL_LANES`）。
单请求也走池化程序，于是 511-token 的 `long` 被切成更多次 replay：long TTFT
317.6 → 396 ms（+25%），比值 4.2–4.5 → 6.10。

**把 wide 图还给池化程序的两个变体都实测否掉了**（同 profile、同 vLLM 参照，
`exp-wideprompt-native-g1`、`exp-firstwide-native-g1`）：

| 变体 | long TTFT | batch4 TTFT | batch4 wall |
|---|---:|---:|---:|
| narrow only（现状） | 398 ms | 243 ms | 1.48 s |
| 4 条池化都给 wide | **311 ms** | 671 ms | 2.32 s |
| 只有第一条池化驻留给 wide | **311 ms** | 671 ms | 2.31 s |

- 给全部池化程序 wide：long 快 22%，但 batch4 退回串行准入（TTFT 243 → 671 ms，
  比值 2.64 → 7.27），因为 wide arena 是每序列最大的分配，4 份加池就超卡。
- **只给第一条驻留：结果一样坏。** 冷启动 server 上只发 batch4（无历史 shell）测得
  热态 group wall 1.52 → 2.15 s，第 4 条 lane 的 TTFT 1.26 s，defer `Preparation` 15 次：
  一份 wide arena 已经足以让第 4 条排不进来。shell 复用池按 `(capacity, readout)` 取，
  还会把前序用例留下的 wide shell 再发给并发请求，加重这一点。
- 结论：**wide prompt 图与"该请求是否独自在跑"绑定，而这件事在准入时刻不可知**
  （`reserve` 只拿到 capacity/readout，池是空的还是即将来 3 条看不出区别）。
  在 prefill 能与并发共享同一份 arena（或按批次大小惰性升级程序）之前，
  池化序列只能走 narrow 图；这条记为已知代价，不再尝试调 `graph_policy`。

**第三次变体：池化程序只捕一张 prompt 图，但捕 wide 那张**（`exp-wideonly-native-g1`）。
前两次失败都是"一个程序捕两张图"，于是把约束改写成"一个程序一张图"再测：

| case | 指标 | narrow（现状） | wide-only | 比值 narrow → wide-only |
|---|---|---:|---:|---|
| long | TTFT | 394 ms | **314 ms** | 6.04 → **4.82** |
| short | TTFT | 56 ms | 99 ms | 1.81 → **3.22** |
| batch4 | TTFT | 240 ms | 286 ms | 2.60 → **3.10** |
| batch4 | wall | 1454 ms | 1439 ms | 2.06 → 2.03 |
| hot_long | wall | 802 ms | 804 ms | 1.00 → 1.01 |

**batch4 这次没崩**（1.44 s，与 narrow 相同），所以前两次的瓶颈确实是"每个池化程序捕几张图"
而不是图有多宽 —— 4×1 张能同时驻留，4×2（或 1×2 + 3×1）不能。但 wide-only 把 `short`
的 51-token prompt 也推上 256-lane 图：TTFT 56 → 99 ms（**+78%**）。三个 TTFT 格子的比值之和
从 1.81+6.04+2.60=10.45 变成 3.22+4.82+3.10=11.14，**净亏**，因此回滚。

**这次量出了 prompt 图的两个机制【确证】**（后面要动 prefill 的人必须知道）：

1. **prompt 图单次 replay 的成本随 lane 数近似线性**：128 → 256 lane 让 51 token 的 prefill
   从 56 ms 涨到 99 ms（≈1.8x）。所以"更宽的图"只在 prompt 长到值得用它换 replay 次数时才划算
   —— 511 token 是 2×256 便宜于 4×128（394 → 314 ms），51 token 反过来。
   这正是 `prefers_narrow_prompt` 按 chunk 长度选图的依据。
2. **池化程序的显存瓶颈是"捕了几张 prompt 图"，不是"图有多宽"**：1 张（无论 128 还是 256）
   都能让 4 条并发驻留，2 张就不行。

因此"既要 short 快又要 long 快"的正路**不是加第二张图**，而是让**一张图同时服务两个区间**
（例如让 256-lane 图对掩码行不再付满代价），或让 prefill 与并发共享同一份 arena。
在那之前 `graph_policy` 维持 narrow-only。

### 11.7 decode 侧的两条负结果：tile 已经是优解，逐节点画像不能再往下挖

`slot_verify`（4 lane、MTP2、12 行）是 decode 的最大单项：图边界事件中位
**26.68 ms**，逐节点一次 replay 的构成（node_count 1155，与 total 一致）：

| 算子 | 每 replay | 节点数 | 每节点 |
|---|---:|---:|---:|
| **linear** | 19.76 ms | 497 | 36.8 µs（中位） |
| delta | 3.89 ms | 48 | 81 µs |
| conv | 2.34 ms | 48 | 49 µs |
| attention | 1.38 ms | 16 | 86 µs |
| norm+add+multiply+rope+silu+split+sigmoid+gated_norm | ~2.7 ms | 529 | ≤7 µs |
| lm_head（组 64 的第 2 个 linear） | 0.78 ms | 1 | 783 µs |

**tile 扫描【确证，已否】**：给 `quant_gemm_tile` 加了一个 capture 期的 env 覆盖
（`INFER_CUDA_QUANT_TILE=rows,cols`，只覆盖 decode 宽度），用同一套 `slot_verify`
图时间扫了列 tile：

| 列 tile | slot_verify 中位 |
|---|---:|
| **64（现状）** | **24.74 ms** |
| 32 | 25.05 ms |
| 128 | 25.06 ms |
| 256 | 26.68 ms |

`[16,64]` 已经是优解；加宽（更少 CTA）和收窄（更多 CTA 但激活重读翻倍）都更差。
覆盖代码已回滚——decode GEMM 的问题不在形状选择。

**逐节点画像的 9 个离群点不是形状病理【确证】**：64 层里 9 层各有一个 linear 节点
200–320 µs，而同样的形状在其它层只有 20–53 µs。但它们

- 出现在**不同的投影位置**（组内下标 11 / 6 / 19 / 14 都有），
- 间隔**非常规律**（约每 96 个节点一次，即每 5–6 层），
- 且 `segments` 之和恰好等于 `total_ms`（没有未计入的间隙）。

形状相同、位置随机、间隔规律 —— 这是**采集期的干扰被摊到某一节点**，不是可修的核。
结论：per-node 画像的**总量**可信，**单个离群节点**不可信，不要照着它改 kernel；
要判断某个形状慢不慢，用隔离 bench + 服务矩阵两把尺子。

### 11.8 下一步（按证据排序）

1. **B2b cuBLASLt NVFP4 委托**（§二 杠杆 A，规格已确证）：这是 decode 唯一还没试过的
   大杠杆。11.7 已经把"换 tile/换形状"这条排除掉，剩下的就是**同一个形状换更好的核**：
   隔离测量窄行形状 cuBLASLt 赢 1.6–1.7x，而我们的 GEMM 在服务里只有 ~50% 带宽。
   两个必须带着的教训：① `[64,128]` 隔离赢 1.47x、服务端反而慢 1.7%
   （`slot_verify` 26.69 → 27.43 ms），所以验收只认服务矩阵；② decode 单个 GEMM 只有
   ~40 个 CTA，填不满 170 SM（`PROMPT_GEMM_TILE_ROWS` 注释），任何减少 CTA 的方向先输。
2. **B4 设备端 greedy argmax**：draft 路径 host 侧 5.5 ms/tick 对设备 2.8 ms/tick，
   差额主要是 8×993 KB 的 logits D2H 与 host 串行 argmax。门控条件
   `temperature == 0 && presence == 0 && repetition == 1.0`，tie-break 必须与 host
   `greedy()` 逐位一致（首个最大值、total order、非有限值报错）。
3. **B3 Row 派发跨 lane 批量化**：`delta` 3.9 + `conv` 2.3 + norm/rope/add ≈ 2.7 ms 的
   12 路串行延迟链。

**测量前先确认二进制**：同一 worktree 里有另一条工作线在构建，`make local-build` 的产物
`target/release/infer` 被 CPU-only 构建覆盖过（24.8 MB → 8.3 MB，随后服务报
`no supported GPU backend available`）。跑基准前 `stat -c%s target/release/infer`
应 >20 MB，否则先重新 `make local-build`。

复现命令（native 侧，vLLM 侧复用 `stage-b2-mtp2-vllm-g1`）：

```sh
bash tools/bench/safe-run.sh python3 tools/bench/serve-compare.py --engine native \
  --model /home/r/models/Qwen3.8-27B-NVFP4 --executable target/release/infer \
  --inputs artifacts/workloads/Qwen3.8-27B-NVFP4-inputs.json --output-dir artifacts/stage-b-20261010 \
  --mtp 2 --tokens 64 --gpu-memory-utilization 0.88 --max-model-len 32768 --max-num-seqs 16 \
  --run-id <新 ID> --profile-id 27b-mtp2 --checklist benchmarks/profiles/27b-mtp2.json
python3 tools/bench/compare-results.py \
  artifacts/stage-b-20261010/Qwen3.8-27B-NVFP4-mtp2-vllm-stage-b2-mtp2-vllm-g1.json \
  artifacts/stage-b-20261010/Qwen3.8-27B-NVFP4-mtp2-native-<新 ID>.json
```

## 十二、prefill 逐算子浪费归因（2026-10-10）

触发这个问题的是服务层的一个规律：**prompt 越长，比值越差**（short 1.81 → long 6.04），
而 native 的 GPU 利用率**高于** vLLM（53.5% vs 41.6%）。利用率更高却更慢，说明差距不是
"GPU 没吃饱"，而是**每个有用 token 上烧掉的周期更多**。本节把 prefill 的每次 replay
拆到算子级，并用"有用 FLOPs / 有用字节"两个下界去卡每个算子。

### 12.1 复现口径

`INFER_CUDA_PREFILL_PROFILE`（逐节点，不加 `GRAPH_ONLY`），单条 `long`（511 token）请求，
release 制品。profile 每条记录带 `tokens` 与 `position`，因此可以直接读出**每个 prompt
graph replay 实际吃了多少 token**：

| graph | tokens | position | node_count | total_ms |
|---|---:|---:|---:|---:|
| prefill | 64 | 256 | 1154 | 45.4 |
| prefill | 64 | 320 | 1154 | 45.8 |
| prefill | 64 | 384 | 1154 | 46.4 |
| prefill_last | 63 | 448 | 1155 | 47.5 |

→ **511 token 被切成 8 次 replay，每次 64 token**（8 × 46 ms ≈ 368 ms，与服务层
long TTFT 394 ms 一致；vLLM 同负载 65 ms）。

### 12.2 每次 replay（64 token，46.1 ms）的构成与下界

权重一遍 = 15.79 GB（48 层线性注意力 249 MB + 16 层全注意力 239 MB，FP4 按 0.5 B/参数、
FP8 按 1 B/参数）→ 带宽下界 **8.82 ms**；GEMM 有用 FLOPs = 2·64·24.3 G = **3.11 TFLOP**。

| 算子 | ms | 占比 | 节点 | µs/节点 | µs/token | 有用下界 | 结论 |
|---|---:|---:|---:|---:|---:|---|---|
| **linear** | 22.46 | 48.7% | 496 | 45.3 | 351 | 8.82 ms（带宽） | 2.5×；实测 138 TFLOP/s = FP8 峰值 ~16%、704 GB/s = 峰值带宽 ~39%，**先撞发射/算力，不是带宽** |
| **delta** | 16.00 | 34.7% | 48 | **333** | **250** | ~0.1 ms（FLOPs）/ 0.17 ms（字节） | **~150×**，实测 ~0.5 TFLOP/s = fp32 峰值 ~0.5% |
| **attention** | 3.25 | 7.1% | 16 | 203 | 51 | ~0.02 ms | **~160×** |
| norm | 1.17 | 2.5% | 161 | 7.3 | 18 | ~0.2 ms | 5×，且 161 个独立节点 |
| gated_norm | 0.94 | 2.0% | 48 | 19.5 | 15 | ~0.06 ms | 15× |
| conv | 0.67 | 1.5% | 48 | 14.0 | 10 | ~0.05 ms | 13× |
| add / multiply / silu / rope / split / sigmoid / embedding | 1.70 | 3.7% | 321 | 4–97 | 17 | ~0.5 ms | 3–5× |

**每个有用 token 的 prefill 设备时间是 720 µs**，其中 GEMM 351 µs、**非 GEMM 369 µs（51%）**。
vLLM 整条 prefill 的每 token 预算只有 ~128 µs —— 也就是说**光是我们的 delta 一项
（250 µs/token）就超过 vLLM 做完整个 token 的预算**。这才是"native 计算量更大"的实处。

### 12.3 无效计算在哪：delta 与 attention 是同一个病

`recurrent_prefill::delta`（`crates/backend/cuda/src/resident/recurrent_prefill.rs:16`）是
**逐 token 串行扫描**：网格 = VH × SW = 48 × 1 = **48 个 CTA（170 SM）**，每个 CTA 把 chunk
的 LANES 个 token 一个个循环过去，每个 token 做两次"跨线程归约 key 维 D=128"：

- 算法本身每 token 每 head 需要 3×D² ≈ 49 K FLOP（状态衰减、rank-1 更新、读出），
  ×48 head ×48 层 = 113 MFLOP/token。**这部分 FLOP 是不可避免的**；
- 但每个 token 要付两次跨线程归约（`reduce_sum(..., 0i32)`）的同步/共享内存代价，
  64 个 token 就是 128 次归约/CTA，而只有 48 个 CTA 去遮这段延迟。

§十.1 已经把"值维分块（SW=4 → 192 CTA）"实测否掉了：**加 CTA 没用**，因为限制不是占用率，
而是每个 token 的归约依赖链。**跨 chunk 宽度的对照也支持这一点**：同一次 profile 里
decode 的 delta 节点是 53.7 µs / 12 lane（4.5 µs/token/层，走 per-lane 核），prefill 是
333 µs / 64 lane（5.2 µs/token/层，走 chunked 核）—— **两者每 token 成本几乎相同**，
说明 chunked 核并没有把逐 token 的开销摊掉，只是把状态留在了寄存器里。

所以正路是**把逐 token 的串行扫描换成 chunk 级矩阵乘**（WY / chunked linear attention：
intra-chunk 用 (C×C)×D 的张量核矩阵乘，状态更新用 D×(C)×(C×D) 一次算完），把"每 token
两次归约"变成"每 chunk 两次归约"。

`attention`（3.25 ms / 16 节点 / 203 µs）是同一类：C=64、KV≤320 的 prefill attention
有用 FLOPs 只有 ~8 GFLOP/replay，本该是几个张量核矩阵乘 + softmax，实测 ~160× 于下界。

### 12.4 另外两条

1. **chunk 宽度与图宽度脱节**：`select_prompt_width` 对 quantized-recurrent 模型给出的
   `narrow_prefill_width = 64`、`prefill_width = 256`；而 Stage A 之后池化程序
   `graph_policy` 只捕 narrow 那张 → 池化序列的 prompt 图 **只有 64 lane**，
   于是 chunk = 64。私有程序捕 256 → chunk = 256，511 token 只要 2 次 replay。
   这就是 §11.6 里 long TTFT 317 → 396 ms 的**全部原因**：replay 次数 2 → 8。
   但把 256 lane 图给池化程序又有 §11.6 的代价（51-token prompt 从 56 → 99 ms，
   因为掩码 lane 也要付钱）。**修完 12.3 之后这条才值得再谈**：非 GEMM 的每 token
   成本塌下去以后，chunk 宽度的收益才会体现为权重流量（8 遍 → 2 遍，126 GB → 32 GB）。
2. **linear 是"发射受限"而非带宽受限**：138 TFLOP/s（FP8 峰值 ~16%）而字节只用到
   704 GB/s（39%）。所以 tile 形状（§11.7 已扫完，`[16,64]` 是优解）不是出路，
   出路是同形状换更高效的核 —— §二 杠杆 A 的 cuBLASLt 委托。

### 12.5 结论与动作顺序

| 序 | 动作 | 依据 | 预期 |
|---|---|---|---|
| 1 | **delta 改 chunk 级矩阵乘**（WY 形式） | 34.7% 的 prefill 设备时间，150× 于下界，且 §十.1 已排除占用率 | replay 46 → ~31 ms；long TTFT 394 → ~280 ms；decode 也减 ~2.6 ms/replay |
| 2 | **prefill attention 同批处理** | 7.1%，160× 于下界 | replay 再 −3 ms |
| 3 | 窄算子融合（norm+add 等 537 节点/4.5 ms） | 9.7%，3–15× 于下界 | replay −2 ms 量级 |
| 4 | GEMM 委托 cuBLASLt（§二 杠杆 A） | linear 48.7%，138 TFLOP/s = 16% 峰值 | 隔离 1.6–1.7×，须服务矩阵验收 |
| 5 | 再谈 chunk 宽度/单图双区间（§11.6） | 依赖 1 落地 | 权重流量 4× |

抓取命令（单条 long、逐节点画像）：

```sh
pkill -f '^target/release/infer'
INFER_CUDA_PREFILL_PROFILE=/tmp/gp.jsonl target/release/infer /home/r/models/Qwen3.8-27B-NVFP4 \
  --listen 127.0.0.1:38091 --num-speculative-tokens 2 --gpu-memory-utilization 0.88 \
  --max-model-len 32768 --max-num-seqs 16 &
python3 /tmp/repro-case.py http://127.0.0.1:38091 long   # 先跑一遍预热，再删 profile 重跑
```

### 12.6 已否掉的低成本尝试：delta 的代数重排【确证，已回滚】

§12.3 的机制假设是"每 token 两次跨线程归约"。最便宜的验证是不动并行度、只重排代数：
输出可以写成

```
sum_d (decay*old + k⊗diff)[d,s] * q[d]
  = decay * sum_d old[d,s]*q[d] + diff[s] * (k·q)
```

这样两次大归约都只读 `old`，互相独立，链从"归约→更新→归约"变成"归约(并行)→更新"，
并且不再需要为了求和而物化 `updated` 整块 tile。实现后跑现成的配对基准
（`resident_recurrent_prefill_benchmark_tests::recurrent_chunk_performance_gate`，
`do_bench_paired` + 清 L2，chunked 核 vs 逐 lane 核）：

| 几何 | 原式 new_ms | 重排后 new_ms | 变化 |
|---|---:|---:|---:|
| (2, 4, 32, 32) | 0.0431 | 0.0451 | **+4.5%** |
| (16, 48, 128, 32) | 0.0720 | 0.0789 | **+9.6%** |
| (16, 48, 128, 128) | 0.2529 | 0.2766 | **+9.4%** |

**慢了 4.5–9.6%**：cuTile 对原式的 elementwise+归约已经融得很好，重排反而多了一次
`k·q` 归约和若干 broadcast。数值上重排是安全的（chunked-vs-per-lane 输出差
7.45e-9 → 1.12e-8，仍比断言 1e-7 小一个量级），但**没有收益，已回滚**。

结论：delta 的每 token 算术已经是最优写法，成本来自**结构**（逐 token 串行扫描 +
每 token 归约 + 48 CTA 的网格），所以只有算法级改写（WY / chunked 矩阵乘）或换并行
分解能救它；**不要再去调这个核的算式**。这也解释了为什么 §十.1 的值维分块同样无效。

顺带修好了这个基准本身：它自 `0ba23c5` 加入 `SW` 泛型后就没再传第 5 个泛型，
在 HEAD 上是**编译不过**的（`not enough generic arguments to instantiate const parameter SW`），
所以"chunked 核对逐 lane 核"的护栏一直没在跑。现在它能在 release 下直接量
（`cargo test --release -p infer-backend-cuda --features cuda -- --ignored recurrent_chunk_performance_gate --nocapture`，
约 2 秒），是后续改 delta 的快速 A/B 工具。

### 12.7 未解释但最便宜的线索：delta 在图内比隔离慢 2.6×【待查】

同一颗核、同一几何（KH=16、VH=48、D=128、SW=D），两个口径对不上：

| 口径 | LANES | 核时间 | 每 token | 来源 |
|---|---:|---:|---:|---|
| 隔离配对基准（清 L2） | 64 | 0.1322 ms | **2.07 µs** | `recurrent_chunk_performance_gate` (16,48,128,64) |
| 隔离配对基准（清 L2） | 128 | 0.2527 ms | 1.97 µs | 同上 (16,48,128,128) |
| 服务图内逐节点事件 | 64 | 333 µs | **5.2 µs** | prefill 每次 replay |

同一宽度（64）下差 **2.5×**。图内那 46 ms 的记账是自洽的（各节点之和 = 图总时间），仪器本身只解释约 7%
（同一 slot_verify：GRAPH_ONLY 26.68 ms vs 逐节点 28.5 ms），所以这不是测量口径问题。

如果这条差距能收掉，**不需要改算法**：replay 46 ms 里 delta 占 16 ms，压到隔离水平即
~6.5 ms，prefill 直接少 ~10 ms/replay（22%），long TTFT 394 → ~310 ms。

顺带：`recurrent_chunk_performance_gate` 现在也量 64 lane（模型 prompt 图实际捕的宽度），
省得再插值；它在 (16,48,128,64) 上是 chunked 0.132 ms vs 逐 lane 0.334 ms。

下一步的判别实验很便宜（改 `resident_recurrent_prefill_benchmark_tests.rs`，约 2 秒一轮）：
在候选图里插一个与服务器同量级的大 GEMM（例如 [64, 17408, 5120] 的 FP4/FP8 投影），
让 delta 在它之后跑。若 2.6× 复现，就是邻接 kernel 造成的 L2/带宽干扰或图内调度缺口，
方向转成"保护 delta 的工作集"或"把 delta 与邻居合并"；若不复现，则差距来自服务侧的
别的因素（例如同一 stream 上的依赖链、图节点间的发射间隙），要另找口径。

### 12.8 结论

- **prefill 的差距不在"GPU 没吃饱"**：利用率比 vLLM 高，每 token 设备时间却是 vLLM 的
  5.6×（720 µs vs 128 µs），其中非 GEMM 占 51%。
- **两个真靶子**：`delta`（34.7%，150× 于下界）与 `linear`（48.7%，138 TFLOP/s = FP8 峰值 16%）。
  `attention` 7.1% 同类但量级小一半。
- **已排除的低成本路线**（不要再试）：值维分块（§十.1）、tile 形状（§11.7、§12.7 相关）、
  delta 代数重排（§12.6）。
- **剩下的路只有三条**：① delta 的算法级改写（WY / chunked 矩阵乘）；② GEMM 委托
  （FP8 那条不需要 scale swizzle，覆盖 15.79 GB 里的 7.22 GB）；③ 先查 §12.7 的 2.6× 缺口。
