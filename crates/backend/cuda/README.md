# NVIDIA CUDA 后端

GPU kernel 使用 NVIDIA cuTile Rust 0.4.0。host 与 device 程序都用 Rust 编写，由 CUDA Driver 与 Tile IR 负责执行和编译。

## 已验证范围

- dense BF16/F32 矩阵向量乘与显式 tile 选择。
- FP8 channel-scaled 与 NVFP4 packed-weight 的参考投影。
- RTX 5090 上全部搜索候选的逐元素数值检查，包括不整除的行 tile。
- 配套 CUDA event 基线：预热、冷 L2、原始样本与 A/B 比较。

量化参考 kernel 使用 F32 activation。动态 activation 量化、block-scaled Tensor Core 执行、完整模型执行与 MTP 推测执行**尚未实现**，CLI 也没有可用的 CUDA 模型执行器。kernel 验证不代表模型质量。

## 视觉塔（多模态）

Qwen3.5-VL 的图像路径完全由 cuTile kernel 组成，权重与几何都来自 provider 的 `ModalityEncoder` 声明：

- kernel：patch embed（GEMM + bias）、位置加权 gather（host 四抽头）、`LayerNorm`（补齐行 tile + 掩码）、QKV/MLP/merger 投影、轴向 `RoPE`、按图 packed 非因果 attention（在线 softmax）、`tanh` 与精确 `erf` GELU。
- cuTile 约束：tile 维度必须是 2 的幂，所以 head 采用 `[2, 64]` 半头布局（36 实数 + 28 补零），`intermediate 4304` 补齐到 4352，行归约用补齐行 tile 加掩码；补零对点积、GEMM 与 GELU 都是中性的。
- 精度：投影保留 BF16 高部和残差两项，attention 的 QK/PV 保留两个一阶残差乘积，以 F32 累加。仅把每个操作数舍入为 BF16 会被完整塔放大，不能用单层或单 key block 的误差代替完整塔验证。
- `RoPE` 在独立 kernel 中对 Q/K 各处理一次；attention 使用完整 padded head，避免逐 query block 重算旋转及按半头重复算分。按帧掩码防止 `(t,h,w)` 网格跨帧 attention。
- Qwen3-VL-2B 的 5 个官方 F32 golden 用例覆盖 16/24/32/64/256 patches（含非方阵和双帧）；完整塔与 merger 均通过原有 1% 门槛。此前 27B tower 3.8e-3 的记录只覆盖 ≤32 patch 用例，不能代表真实长图像。
- 图像预处理（`infer-models::image`）与官方 processor 对齐到两个 8-bit 量化级。
- 示例：
  - `cuda-vision-check`：逐级对齐 golden（需视觉 golden 文件）；JSON `passed` 与退出码一致，拒绝非有限值和形状不符。`INFER_VISION_TRACE_BLOCKS=1` 可输出同输入单层误差与累计误差，需生成时传 `--trace-blocks`。
  - `qwen3vl_vision_reference.py`：默认 CPU/F32，支持 `--dtype bfloat16 --device cuda` 做独立精度诊断，F32 CUDA 参考明确禁用 TF32。
  - `cuda-loaded-vision-check`：`LoadedModel` 绑定视觉塔并对 golden 图像编码。
  - `cuda-multimodal-generate`：图像 → 视觉嵌入 → resident 逐 token prefill（媒体位置嵌入覆盖）→ 贪心生成。

2026-10-07 重跑已有的官方 **CPU/F32** golden（原有 1% 门槛未放宽）：

| grid | patches | tower 最大相对误差 | merger 最大相对误差 |
|---|---:|---:|---:|
| `(1,4,4)` | 16 | 6.50e-5 | 3.19e-5 |
| `(1,4,6)` | 24 | 8.06e-5 | 2.45e-5 |
| `(2,4,4)` | 32 | 2.28e-4 | 4.83e-5 |
| `(1,8,8)` | 64 | 2.24e-4 | 1.94e-4 |
| `(1,16,16)` | 256 | 8.87e-4 | 1.34e-4 |

27B 的 72 维头（补到 128）也重建并跑过同样 5 个 F32/禁用 TF32 用例：tower 最大相对误差 7.65e-5、merger 最大 8.74e-5，全部通过。

2026-10-07 RTX 5090 / release 的同输入计时如下（单位 µs，16 heads，60 次 CUDA event 样本的中位数，warm CUDA Graph 只包含 RoPE + attention；不包含分配、JIT、QKV/输出投影）。参考为 PyTorch 自动选择的 SDPA 后端，关闭 TF32；BF16 列另含输入转换，精度契约不同。

| tokens | head dim | 补偿 32×32 基线 | 最终配置 | SDPA F32 | SDPA BF16 |
|---:|---:|---:|---:|---:|---:|
| 128 | 64 | 11.84 | 11.84 | 28.88 | 24.32 |
| 512 | 64 | 46.62 | 38.50 | 75.89 | 37.70 |
| 2048 | 64 | 508.14 | 483.66 | 713.12 | 160.54 |
| 128 | 72 | 26.85 | 26.85 | 35.10 | 23.58 |
| 512 | 72 | 145.97 | 121.10 | 95.36 | 44.85 |
| 2048 | 72 | 1731.22 | 1632.08 | 1186.38 | 221.95 |

比较了 32×32、64×64、流水 64×64、流水 128×64，共 24 组，全部满足独立 F32 SDPA 最大相对误差 <1e-4。最终仅在 tokens≥512 时选择 64×64，512-token 提速约 1.21×、2048-token 约 1.05–1.06×；流水加载没有稳定收益，未启用。72 维头补齐到 128，长序列仍慢于 F32 SDPA；长序列也未追平低精度 BF16 SDPA。这里的基线已包含补偿精度修复，不能用于声称相对旧低精度内核的净提速。

原始样本与比较位于 `artifacts/vision/attention-comparison.json`。复现时先运行 `tools/vision/attention_benchmark.py --out /absolute/path/attention.safetensors`，然后将 `INFER_ATTENTION_BENCH` 设为该**绝对路径**，通过 safe-run 运行 `cargo test --release -p infer-backend-cuda --features cuda attention_benchmark_against_sdpa -- --ignored --nocapture`。Python 脚本也必须通过 safe-run 启动。

GPU 回归 `vision::attention_check` 使用独立 F32 oracle（阈值 1e-4），覆盖 32/33/64/65/100 tokens、Online/Exact、帧内隔离，以及会被单次 BF16 转换丢失的投影残差。运行：

```sh
bash tools/bench/safe-run.sh --memory-gib 8 env CUDA_TOOLKIT_PATH=/opt/cuda \
  BINDGEN_EXTRA_CLANG_ARGS='-isystem /usr/lib/gcc/x86_64-pc-linux-gnu/16/include' \
  cargo test --locked -p infer-backend-cuda --features cuda --lib -- --ignored vision::attention_check
```

`--memory-gib` 只能降低默认 32 GiB 上限；启动仍要求上限之外至少留 16 GiB，禁用 swap。
2B 端到端参考必须同时清空 config 与已构造视觉模块的 `deepstack_visual_indexes`，并断言输出没有 DeepStack feature；仅改 config 不会关闭已构造模块。该共享路径对照不代表完整发布版模型的支持。

2026-10-07 最终端到端验收通过：本仓库 RGB8 预处理 → tower → merger → MRoPE prefill → greedy decode，95 个 prompt token、77 个视觉 token，生成 **16/16 tokens（含 EOS）完全一致**；视觉最大相对误差 **3.232e-04**，首步 logits 最大绝对误差 **0.002251**。参考请求最多生成 32 tokens，在第 16 个 token 自然结束；双方均关闭 DeepStack。完整输出保存于 `artifacts/vision/parity-native-long.json`。

文本侧已接入三轴 MRoPE：导入旧 `rope_scaling` 的 sections/interleaved 配置且保留顶层 theta，按合并后的图像网格放置 `(t,h,w)` 坐标，decode 使用网格压缩后的 rotary delta，KV 下标仍逐 token 顺序递增。普通文本三轴取同一位置。CPU 回归覆盖频率轴布局、多图/文本位置和非法网格；GPU 回归覆盖三轴元数据更新。

RGB8 预处理已匹配官方 CPU 路径的 16 位定点 bicubic 权重，缩小时扩大抗混叠支撑范围，并采用 ties-to-even 几何舍入。放大/缩小固定像素样本、随机实图 golden 和完整生成样本均按归一化误差 ≤1e-6 验收。`cuda-parity-2b` 默认使用本仓库预处理后的像素；`INFER_PARITY_OFFICIAL_PIXELS=1` 仅用于隔离内核诊断。贪心验收要求完整参考序列匹配且视觉误差 ≤1%，不再只看首 token。

复现完整生成：先通过 safe-run 运行 `tools/vision/qwen3vl_2b_parity_reference.py --package /path/to/2b --dtype float32 --max-new-tokens 32 --out /path/to/reference.safetensors`，再通过 safe-run 运行 `cargo run -p infer-backend-cuda --features cuda --example cuda-parity-2b -- /path/to/2b /path/to/reference.safetensors`。本例在 EOS 处结束，共 16 tokens。

不支持的能力一律返回明确错误，不会回退到纯文本执行；契约由这些测试固定：

- `prompt::tests::missing_encoder_is_unsupported_not_silent`（模型未声明编码器）
- `prompt::tests::placeholder_mismatch_is_an_explicit_error`（占位符与媒体数量不符）
- `prompt::tests::placements_follow_the_expanded_spans`（编码宽度/数量与 span 不符）
- `vision::session::tests::placeholders_without_encodings_are_rejected`（占位符没有编码，或反过来）
- `vision::session::tests::misaligned_placements_are_rejected`（位置错配、重复、越界、空编码）
- `image::tests::smart_resize_matches_the_reference_grid`（退化尺寸与非法规格）

视频/音频编码器、Engine/Scheduler 级多模态调度、官方 LLM 端到端对照仍是缺口。

## 自动 tile 选择

tile 尺寸是运行设备的属性，所以加载时自动决定，不需要操作者配置：

1. 先用运行设备的 identity（`名称|架构|显存MiB`）查仓库内置基线 `baselines/tuning.json`，再查机器本地
   测量缓存 `~/.cache/infer-cuda/tuning.json`（路径可用 `INFER_CUDA_TUNING_TABLE` 覆盖）。
2. 两者都没覆盖的几何在加载时现场测量（`[1,4,8,16] × [128,256,512]`、2% 确认门槛、逐候选数值校验），
   结果按设备 identity 落盘，下次启动直接复用；测量内容打印到 stderr。
3. `--no-autotune` 或测量失败时使用与硬件无关的保守 tile。测量失败只回退并在报告里注明原因，
   不会让模型加载失败。

换一张卡或换一个模型只影响“要不要测”，不影响“会不会用错别处的 tile”：任何条目都只在设备 identity
完全一致时命中。

条目按 `(dtype, rows, columns)` 索引，**从不按模型名索引**：任何模型只要投影几何相同就直接复用，
模型自己的权重（包括 MTP 的 fusion 投影）也走同一条 `select_tiling` 路径。因此新增模型同样零配置——
只有出现没测过的 `(dtype, 形状)` 时才测一次，其余沿用已有数据。

## 构建与验证

需要 Linux、Rust、带 `tileiras` 的 CUDA Toolkit、CUDA driver、libclang 及其标准 C 头文件。本地已使用 CUDA 13.4 与 RTX 5090 验证。

```sh
CUDA_TOOLKIT_PATH=/opt/cuda cargo +stable run -p infer-backend-cuda \
  --features cuda --example cuda-kernels

CUDA_TOOLKIT_PATH=/opt/cuda cargo +stable run -p infer-backend-cuda \
  --features cuda --example cuda-baseline -- 17408 5120
```

本机 bindgen 还需要 `BINDGEN_EXTRA_CLANG_ARGS='-isystem /usr/lib/gcc/x86_64-pc-linux-gnu/16/include'`；该路径与主机相关，不是可移植的构建要求。

不上传权重、只查看模型投影尺寸：

```sh
target/debug/examples/cuda-baseline --catalog /path/to/Qwen3.8-27B-NVFP4
```

采集流程与结果范围见 [CUDA 性能指南](../../../docs/guides/cuda-performance.md)。`make check-cuda` 运行严格 Clippy、测试、Rustdoc 与真实 GPU kernel 检查；托管 CI 的 `make check-rust` 只检查可移植的 CUDA strategy API，不启用依赖 Toolkit 的 feature。GPU 验收必须在 CUDA 主机上执行。

## 职责边界

| 职责 | 归属 |
|---|---|
| 校验存储格式、scale 与逻辑 shape | `infer-models::QuantizedPackage` |
| 把模型适配为投影 workload | `infer-models::ProjectionCatalog` |
| 枚举合法 launch 配置 | `strategy::LinearStrategy` |
| 分配并启动 typed device tensor | `device::CudaDevice` |
| 数学计算 | `kernels` |
| 对比配置并记录样本 | `benchmark` |

模型名称不会进入 kernel 来选择 tile。prefill、decode 与 MTP 需要各自独立的策略，decode GEMV 的测量结果不能用来选择 prefill GEMM tile。后续执行器必须遵守 `BackendProvider` 与 `KvCacheManager` 的所有权、checkpoint 与回滚契约。
