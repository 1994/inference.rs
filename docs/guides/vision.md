# 图像模型与验证

图像路径由模型 provider 的模态声明、图像预处理、CUDA 视觉塔和文本位置编码组成。新增模型时由 provider 声明权重、几何、模态与融合位置，通用 attention API 不包含模型名或硬件型号。

## 执行范围

CUDA 提供 patch embed、LayerNorm、QKV/MLP/merger 投影、轴向 RoPE、按图/帧隔离的非因果 attention、tanh/erf GELU。视觉投影与 attention 使用补偿精度；head padding 必须配合 mask，并覆盖跨 key block 的累计误差。

文本侧支持三轴 MRoPE：图像位置来自合并后的 `(t,h,w)` 网格，decode 使用 rotary delta，KV 下标逐 token 递增；普通文本三轴使用相同位置。RGB8 预处理对齐定点 bicubic、抗混叠与几何舍入。

当前完整图像生成由 `cuda-multimodal-generate` / `cuda-parity-2b` 示例驱动，尚未接入 Engine/Scheduler 的多模态服务调度。视频、音频和完整 DeepStack 路径不在此验收范围。缺少编码器、占位符数量或位置不匹配时必须报错，不回退成纯文本执行。

## 独立参考与回归

| 入口 | 用途 |
|---|---|
| `tools/vision/qwen3vl_vision_reference.py` | 官方模型逐层视觉 golden；默认 CPU/F32，GPU F32 禁用 TF32 |
| `cuda-vision-check` | tower / merger 数值、形状与有限值校验 |
| `cuda-loaded-vision-check` | 加载真实权重后的视觉编码 |
| `vision::attention_check` | 独立 F32 oracle、尾块、帧隔离与投影残差 |
| `tools/vision/qwen3vl_2b_parity_reference.py` / `cuda-parity-2b` | 本仓库预处理到完整生成序列的对照 |

```sh
bash tools/bench/safe-run.sh --memory-gib 8 \
  cargo test --locked -p infer-backend-cuda --features cuda --lib \
  vision::attention_check -- --ignored
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
