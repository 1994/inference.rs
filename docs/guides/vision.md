# 图像与视频模型及验证

视觉路径由模型 provider 的模态声明、图像/视频预处理、CUDA 视觉塔和文本位置编码组成。新增模型时由 provider 声明权重、几何、模态与融合位置，通用 attention API 不包含模型名或硬件型号。

## 执行范围

CUDA 提供 patch embed、LayerNorm、QKV/MLP/merger 投影、轴向 RoPE、按图/帧隔离的非因果 attention、tanh/erf GELU。视觉投影与 attention 使用补偿精度；head padding 必须配合 mask，并覆盖跨 key block 的累计误差。

文本侧支持三轴 MRoPE：图像与视频位置来自合并后的 `(t,h,w)` 网格，decode 使用 rotary delta，KV 下标逐 token 递增；普通文本三轴使用相同位置。RGB8 预处理对齐定点 bicubic、抗混叠与几何舍入。

图像与视频共用一套 patch 布局：`ImageProcessor` 把静帧在每个 temporal tap 上重复，`VideoProcessor` 按参考 video processor 的 `fps` / `min_frames` / `max_frames` 抽帧、按整段像素预算缩放，并把采样数补齐到 `temporal_patch_size` 的整数倍。帧缩放按可用核数并行，输出顺序与抽帧顺序一致。

视频 prompt 复现参考 processor 的格式：每个 temporal group 前放 `<t seconds>`，并各自带 `vision_start`/`vision_end` 与该 group 的时序 patch；每个 group 作为独立 media item 编码与放置。这样 video 的 token 序列与官方 processor 逐 token 一致（45 token 的示例 prompt 与参考完全相同），而不是把整段视频展开成一串连续占位符。

多模态 prompt 以融合分块执行：每个 media 行在设备流上拷入 lane 的外部 embedding 槽，embedding kernel 按每个 token 的 override 标志选择词表或外部向量，每个 lane 的三轴位置写入 prefill 的 M-RoPE 表，因此一个 1000 token 的图像 prompt 只花 `ceil(tokens / prefill_width)` 次图重放，而不是每 token 一次 decode。没有 prompt 图（`prefill_width < 32`）时退回逐 token 路径，对齐合同不变。

RTX 5090 上 Qwen3-VL-2B 的实测（prefill 阶段，含同一视觉编码）：172 token 双图 prompt 融合分块 60 ms，逐 token 回退 543 ms；45 token 视频（2 个 temporal group）9.7 ms 对 128 ms。数字只说明两条路径的相对差，不作为发布性能门槛。

当前完整图像与视频生成由 `cuda-multimodal-generate` / `cuda-multimodal-media` / `cuda-parity-2b` 示例驱动，尚未接入 Engine/Scheduler 的多模态服务调度。音频与完整 DeepStack 路径不在此验收范围：Qwen3-VL 没有音频模态，Qwen Omni 也未被 provider 导入；带 `deepstack_visual_indexes` 的权重仍按共享路径验收（双方关闭 DeepStack）。缺少编码器、占位符数量或位置不匹配时必须报错，不回退成纯文本执行。

## 独立参考与回归

| 入口 | 用途 |
|---|---|
| `tools/vision/qwen3vl_vision_reference.py` | 官方模型逐层视觉 golden；默认 CPU/F32，GPU F32 禁用 TF32 |
| `cuda-vision-check` | tower / merger 数值、形状与有限值校验 |
| `cuda-loaded-vision-check` | 加载真实权重后的视觉编码 |
| `vision::attention_check` | 独立 F32 oracle、尾块、帧隔离与投影残差 |
| `tools/vision/qwen3vl_2b_parity_reference.py` / `cuda-parity-2b` | 本仓库预处理到完整生成序列的对照 |
| `tools/vision/qwen3vl_video_golden.py` | 官方 video processor 的抽帧、缩放、时间 patch 与归一化 golden |
| `cuda-multimodal-media` | 单 prompt 双图与多帧视频：正/反顺序的 logits 位移与融合分块 prefill |

```sh
bash tools/bench/safe-run.sh --memory-gib 8 \
  cargo test --locked -p infer-backend-cuda --features cuda --lib \
  vision::attention_check -- --ignored
bash tools/bench/safe-run.sh cargo run --locked --release \
  -p infer-backend-cuda --features cuda --example cuda-multimodal-media -- \
  /path/to/model 6
```

端到端参考先通过 `safe-run.sh` 运行 Python 工具，传入 `--package /path/to/model --dtype float32 --max-new-tokens 32 --out /path/to/reference.safetensors`，再运行：

```sh
bash tools/bench/safe-run.sh cargo run --locked --release \
  -p infer-backend-cuda --features cuda --example cuda-parity-2b -- \
  /path/to/model /path/to/reference.safetensors
```

2B 对照要求双方关闭 DeepStack：参考端同时清空 config 和已构造视觉模块的 `deepstack_visual_indexes`，并断言无 DeepStack features。贪心验证要求完整参考序列匹配、视觉误差 ≤1%；`INFER_PARITY_OFFICIAL_PIXELS=1` 只用于隔离内核诊断，正式验收使用本仓库预处理。

已记录的共享路径样本为 95 个 prompt tokens、16/16 生成 tokens（含 EOS）匹配，视觉相对误差 3.232e-4。这不等于完整发布版模型或所有输入通过。复测应包含非方形、多帧、长图像、MRoPE decode 和 EOS。

单层数值、完整视觉塔和端到端生成分别验收。Attention 性能使用 [Candle 门禁](quality-gates.md#通用-attention-门禁)，不把自动选择的 SDPA 后端当作固定实现基线。
