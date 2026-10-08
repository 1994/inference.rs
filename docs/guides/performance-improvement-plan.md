# Inference 性能改进计划

**定位**：本文档是 `performance-roadmap.md` 的执行侧续篇。roadmap 回答"差在哪、什么不要
重试"（现状、根因、§四 已否方向）；本文档回答"做什么、预期多少、怎么验证"。引用
roadmap 处记为 RM§x。

**总目标**：第一阶段 8 格全矩阵 ≤1.10x；第二阶段 TPOT 类指标稳定领先 vLLM 20–30%。

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
| 5 | 调度中程项、杠杆 C、E | p99 与饱和负载 soak（过载面见 [serving-stability.md](serving-stability.md) §五.8） |
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
- 单 rank 故障 = 全组 hang（NCCL 超时）：[serving-stability.md](serving-stability.md)
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

   即：**TTFT 全面下降 10–24%**（预填充收益），但 short/hot_long 的 TPOT/wall 反而退 ~8–10%
   —— 因为 verify 图也用同一个 workspace，同样切到了 chunked。所以默认保持关闭是对的；
   要用它应当只对**预填充图**开启（把开关放到 BatchBuilder 的 prompt 构建上，而不是整个
   workspace）。这条留作下一步，另外 token 会变（0/21 与默认一致），属于数值取舍。

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
   hot_long 1.457 / 2.093 / 1.393。**token 与 vLLM 仅 3/21 逐字节一致 —— 已裁决，不是我们的 bug**：vLLM 在
   sm_120 上选 DeepGemmFp8BlockScaledMMKernel 且 `is_deep_gemm_e8m0_used()` 为真，
   `requant_weight_ue8m0_inplace` 会**用 checkpoint 的 fp32 scale 反量化、再用
   `per_block_cast_to_fp8(..., use_ue8m0=True)` 重新量化并把新 fp8 权重与 2 的幂
   scale 原地写回**（`fp8_utils.py:881-945`）。在本 checkpoint 上抽 12 个投影实测：
   权重相对 Frobenius 变化均值 **2.67%**、最大 2.73%，单权重变化中位数 **2.19%**
   —— 即 vLLM 评估的是"同一份权重的另一种量化"，多一次 e4m3 舍入 + 2 的幂
   scale 约束；我们按 checkpoint 原样计算，误差 0。差异来源已定，无需 PyTorch 参考。
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
