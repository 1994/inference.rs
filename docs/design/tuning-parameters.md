# 参数来源审计：通用逻辑 / 上下文推导 / 配置

本文审计 CUDA 后端与 MTP 路径上的硬编码值，按三种性质分类：**通用逻辑**（格式或算法事实，保持常量即可）、**上下文推导**（值应由形状、lane 数、格式或预算算出来）、**数据驱动/配置**（值取决于设备与模型，必须来自测量表或配置，不能写死在代码里）。

判定标准：一个值如果在换 GPU、换模型、换精度或换并发规模后应当改变，它就不属于「通用逻辑」。

## 1. 通用逻辑（保持常量）

| 位置 | 值 | 为什么通用 |
|---|---|---|
| `constants.rs` | `F32_BYTES = size_of::<f32>()` | 元素宽度，由类型系统给出 |
| `constants.rs` | `MIB` / `GIB` | 单位换算 |
| `constants.rs` | `AUX_KERNEL_TILE = 256` | 辅助算子的固定 tile |
| `constants.rs` | `PREFILL_LANES = 32` | prefill 批量图的 lane 数（GEMM M tile 与之绑定） |
| `constants.rs` | `NVFP4_GROUP_SIZE = 16` | NVFP4 格式定义（1 个 E4M3 scale 覆盖 16 个 FP4） |
| `constants.rs` | `METADATA_FIELDS = 4` | 每 lane metadata 字段数 |
| `tuning.rs` | `MIN_CONFIRM_SPEEDUP = 1.02`、`MATCH_TOLERANCE = 5e-4`、`WARMUP_MS`/`REP_MS`/`MIN_REPS`/`MAX_REPS` | 自动调优的统计口径，与硬件无关 |
| `strategy.rs` | `SUPPORTED_TILE_ROWS`、`MIN/MAX_TILE_COLUMNS` | kernel 支持域的边界 |
| `linear_batch.rs` | `convert_tile` 的 F32 累加、无 activation 量化 | 算法选择（精度契约） |

## 2. 上下文推导（本轮已修）

| 位置 | 原值 | 现在 | 依据 |
|---|---|---|---|
| `batch_projection.rs` | `VERIFY_OUTPUT_TILE = [4, 8]` 的行数 `4` | `FUSED_VERIFY_LANES + 1` | verify 批量有 3 个 lane，kernel 额外写 1 行 padding；lane 数变化时行数必须跟着变 |
| `batch_projection.rs` | nvfp4 泛型 `BP = 128` | `VERIFY_TILE_DEPTH / 2` | 一个字节装两个 FP4 nibble |
| `batch_projection.rs` | nvfp4 泛型 `BS = 16` | `NVFP4_GROUP_SIZE` | scale 分组必须等于格式定义的 group size |
| `constants.rs` | `MAX_MTP_DEPTH = 8` | `MAX_VERIFICATION_WIDTH - 1` | verify 批量宽度 = 提交 token + 候选数，深度上界由批量宽度决定；两个常量不能再漂移 |
| `executor/execution.rs` | draft 位置里的字面量 `index - 1` | `index - MTP_KV_OFFSET` | draft 的 KV 比 target 落后一个位置（MTP head 的 `kv_offset`），这是模型结构的属性，不是散落的魔法数 |

这些改动保持取值不变（4、128、16、8、1），只把来源变成可推导；GPU 63 项数值回归与 greedy 等价复测通过。

## 3. 数据驱动 / 配置（未做，按优先级）

### 3.1 tile 选择的 GPU/模型硬编码 —— 已改为数据驱动

**现状**：tile 选择不再依赖 GPU 名与模型形状的代码分支，改为四级，全程零配置：

1. **仓库内置基线**（数据）：`crates/backend/cuda/baselines/tuning.json` 保存已实测设备的
   `{"device": "<名称|架构|显存MiB>", "entries": [{"dtype","rows","columns","tile_rows","tile_columns"}]}`。
   只有运行设备的 identity 与条目记录的完全一致时才命中；换卡不会误用别处的 tile。
2. **机器本地测量表**（数据）：`~/.cache/infer-cuda/tuning.json`（或 `INFER_CUDA_TUNING_TABLE`），
   记录本机自动测量过的 shape，优先级高于内置基线；同样按设备 identity 隔离。
3. **自动测量**（默认开启）：前两级都没覆盖的几何，用 `tuning::tune()` 在加载时现场搜索
   （`[1,4,8,16] × [128,256,512]`、2% 确认门槛、逐候选数值校验），结果按 identity 落盘供下次启动复用；
   进程内按 (dtype, rows, columns) 记忆，同一几何只测一遍。`--no-autotune` 可关闭。
4. **兜底**：测量关闭或失败时使用 `strategy::default_tiling()`——一个与 GPU 名、架构和模型形状都无关的
   保守 tile（`DEFAULT_TILE_ROWS/COLUMNS`）；它同时是自动测量的基线，两者不会漂移。测量失败会记入
   `TuningReport` 并由 CLI 打印原因，不会让模型加载失败（可选优化不应阻断加载）。

键只有设备 identity 与 `(dtype, rows, columns)`，**没有模型维度**：任何模型只要投影几何相同就直接
复用已有条目，模型自己的权重（含 MTP 的 fusion 投影，`FusionWeights.tiling`）也走同一条
`select_tiling`；只有没测过的 `(dtype, 形状)` 才测一次。新增模型因此同样零配置。

**实测**（RTX 5090 / Qwen3.8-27B-NVFP4）：autotune 28 s 内测出 11 个几何并落盘，输出与基线
**逐 token 一致**（72 token）；之后加载 7.1 s（不再重测）且输出仍一致。autotune 独立复现出了原来
硬编码的那些 tile（`fp8-channel:6144x5120 → 16×512`、`bf16:48x5120 → 1×512`），同时补上了原表没有
覆盖的形状。这些结果已随 `baselines/tuning.json` 内置，所以该设备 + 该模型组合启动即命中基线、
不再测量；换卡或换模型才触发第 3 级。

改造时第三级曾保留一张 5090 静态表（下文存档）。它同时绑定设备名与模型形状：换卡返回保守值、
换模型静默回落，因此已整体删除。设备专属结果现在只有两个来源——测量数据（内置或本地），或现场测量。

### 3.1b 原始硬编码（存档，已删除）

`strategy.rs:102-126`：

```rust
if *target != CudaTarget::BlackwellSm120 || gpu != "NVIDIA GeForce RTX 5090" {
    return LinearTiling::new(DEFAULT_TILE_ROWS, DEFAULT_TILE_COLUMNS);   // (16, 256)
}
let (rows, columns) = match key {
    "bf16:48x5120" => RTX5090_TILE_NARROW,
    "nvfp4:17408x5120" | "nvfp4:5120x17408" | "fp8-channel:6144x5120" | ... => RTX5090_TILE_WIDE,
    _ => (DEFAULT_TILE_ROWS, DEFAULT_TILE_COLUMNS),
};
```

`17408x5120`、`48x5120`、`6144x5120` 是 **Qwen3.8-27B 的具体投影形状**。换模型（哪怕同为 Blackwell）就静默掉到 `(16, 256)`——而 decode 是带宽问题，tile 选错直接就是那位数的带宽损失。**这不该是代码分支，而应是一张按 (arch, dtype, rows, columns) 索引的测量表**：

1. 把 `benchmarks/baselines/rtx5090-resident-projections.json`（已存在）或独立的 tuning 文件作为数据源，启动时加载；
2. 查不到的形状走**已有的** `tuning::tune()`（搜索 `[1,4,8,16] × [128,256,512]`、要求 2% 确认增益），把结果写回缓存供下次启动复用；
3. `DEFAULT_TILE_ROWS/COLUMNS` 只作最后兜底，且应可配置。

### 3.2 硬件事实与显存策略 —— 已改为驱动查询 + 推导

**不需要第三方库、也不需要维护硬件表**：`DeviceProfile`（`crates/backend/cuda/src/device.rs`）在
`LoadedModel::open` 时从 CUDA driver API 查一次事实值，策略值全部由它推导：

| 字段 | 来源 |
|---|---|
| `name` / `architecture` / `multiprocessors` | `cuDeviceGetName` / `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_*` / `..._MULTIPROCESSOR_COUNT` |
| `total_memory_bytes` | `cuMemGetInfo_v2` |
| `memory_clock_khz` / `memory_bus_width_bits` | `..._MEMORY_CLOCK_RATE` / `..._GLOBAL_MEMORY_BUS_WIDTH` |
| `l2_cache_bytes` | `..._L2_CACHE_SIZE` |

推导值：

- `memory_bandwidth_bytes_per_second()` = 2 × 时钟 × 位宽/8。**实测 1792.1 GB/s，与项目文档里的 roofline 1792 GB/s 一致** —— 这条自动给出 decode 的带宽分母，不用人手填。
- `device_headroom_bytes()` = 显存/32，夹在 [256 MiB, 1 GiB]。在 32 GB 卡上得到 0.98 GiB（原常量正是 1 GiB，行为保持不变），小卡自动变小、大卡有上界。
- `identity()` = `名称|架构|显存MiB`，用作**机器本地调优表的键**：表自己在哪块卡上测的就记哪块，换卡自动失效，无需人工维护。autotune 落盘时写入的就是这个 identity。

GPU 实测：`cuda-provider-check` 输出 profile 且验收仍通过（`391`、token `[18,24,16,248046]`）。

**关于开源库**：`nvml-wrapper`（Apache-2.0/MIT）是 Rust 侧的 NVML 绑定，能多给功耗/温度/ECC/进程显存——
但**本项决策不需要它**：预算与 roofline 需要的事实值 driver API 已经全有，而「哪块 tile 最快」本质是测量问题，
没有任何库能给答案（Triton、TensorRT-LLM、CUTLASS 都是自动调优 + 按设备缓存，和这里的做法一致）。
若以后要做功耗/温度/ECC 的运行时遥测，再加 `nvml-wrapper` 是合理选择；当前 `tools/bench/hardware-monitor.py`
（nvidia-smi）已经覆盖了监控需求。

### 3.2b 原始硬编码（改造前，存档）

`DEVICE_MEMORY_HEADROOM = 1 GiB`、`GRAPH_ALLOCATOR_HEADROOM = 256 MiB`、`VERIFICATION_CHECKPOINT_BUDGET = 2 GiB`、`ACTIVATION_ARENA_BUDGET`/`PROJECTION_WORKSPACE_BUDGET = 512 MiB`、`DEFAULT_DEVICE_MEMORY_MIB = 512`、`CUDA_MAXIMUM_STATES = 64`。

这些是**策略**而非事实：在 32 GB 卡上它们决定并发上限（实测每序列 2.75 GB、MTP 打开后 3.44 GB），在 80 GB 卡上应有不同的比例。建议按设备显存推导（比例 + 绝对下限）或暴露为配置；`CUDA_MAXIMUM_STATES` 更应由 `state_budget / 单状态预算` 推导，而不是拍一个 64。

### 3.3 `verification_width` 的隐式策略

`loading/mod.rs` 在 `mtp_depth > 0` 时自动把 `verification_width` 设为 `FUSED_VERIFY_LANES`。这是一个会**同时改变显存占用**（每序列 +25%）的策略决定，却藏在 loader 里，调用方无法覆盖。建议至少做成显式配置项（默认值保持现在行为）。

### 3.4 推测能力边界应上报引擎，而不是后端静默回退

`executor/execution.rs::speculate()` 里 `sampling.temperature != 0.0 → 不推测`。这是一个**能力声明**（非贪心需要残差采样，当前只支持贪心），但引擎完全不知道：它永远把 decode 标成可推测，后端每步静默回退。建议提升为 `DeviceCapabilities` 的字段（如 `speculation: { greedy_only: true }`），让调度与统计能看到真实路径。

### 3.5 用「写死的限制」代替通用逻辑：fusion × 批量

`resident/batch.rs:75-81` 直接 `return Err(...)` 拒绝任何带 fusion 的程序建批量图，`capture32` 把 `fusion: &mut None` 写死。后果是 draft 永远单步、预热 O(prompt)（实测 650 µs/token，TTFT +0.55 s）。正确形态是让批量捕获支持 per-lane external hidden，而不是在批量入口拒绝一类程序。

### 3.6 其他

- `DEFAULT_PREFILL_CHUNK_TOKENS = 32` 与 `PREFILL_LANES = 32` 是两个独立的 32：应引用同一个常量，或明确区分「几何」与「策略」。
- `model_smoke` 诊断示例里的 `kv_offset = 1` 与生产路径的 `MTP_KV_OFFSET` 现在是同一语义的两份表达，长期应共用来源。

## 4. 结论

三批改动全部落地并验证：

1. **该推导却写死的 5 处** → 推导式（verify tile 行数/packed 深度/scale 深度、`MAX_MTP_DEPTH`、draft KV 偏移）。
2. **设备相关值与配置** → `DeviceProfile` 从 CUDA driver 查一次事实值，派生出带宽、显存 headroom、graph headroom、arena/workspace 预算、checkpoint 预算、resident state 数；**不存在需要人工维护的硬件表**。tile 选择是「仓库内置基线 + 机器本地测量表（都按设备 identity 隔离）+ 默认自动测量 + 与硬件无关的兜底」，默认零配置，`--no-autotune` 才关闭测量。
3. **能力边界与策略上移到接口**：`--verification-width` 显式可配（0 = 按 draft 深度推导）；
   `DeviceCapabilities.speculation { draft_depth, greedy_only }` 上报引擎，`dispatch` 只在
   `draft_depth > 0` 时才把 sampling 交给后端——不再每步静默回退。

GPU 验证：`cuda-resident-check` 63/63 PASS；早期 `--autotune` 端到端跑通，表以设备 identity 落盘 11 条，
输出与基线逐 token 一致；这些结果已内置为 `baselines/tuning.json`，默认路径不再需要任何开关。

仍然保留的写死项是 **3.5**（fusion 程序不能建批量图），它是能力缺失而不是参数问题，需要改 capture 路径。
