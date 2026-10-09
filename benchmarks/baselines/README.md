# RTX 5090 性能基线

本目录保存 cuTile Rust 投影算子的可复现基线。所有数字只适用于记录的合成形状与同步提交流程，不是模型吞吐，也不是自动回归阈值。

新的正式测试、服务基线与 vLLM 对齐统一按 [性能基线方案](../../docs/plans/performance/baseline.md) 执行。本目录的投影记录保持自己的历史条件，不能补齐缺失的服务矩阵；更早的 debug 样本仅保留为历史证据，不参与正式 release 对比。

## 文件

| 文件 | 内容 |
|---|---|
| `rtx5090-release-bf16-17408x5120.json` | release 搜索样本 |
| `rtx5090-release-confirmation.json` | release 独立复测样本 |
| `rtx5090-release-environment.txt`、`rtx5090-release-source-sha256.txt` | release 构建环境与源码指纹 |
| `rtx5090-confirmation-environment.txt`、`rtx5090-confirmation-source-sha256.txt` | 复测环境与源码指纹 |
| `rtx5090-bf16-17408x5120.json` | 更早的 debug 搜索样本（15 组配对比较） |
| `rtx5090-bf16-wide-17408x5120.json` | 包含 8192 列 tile 的宽归约搜索，结果回退，保留为负面证据 |

release 复测中，`17408 × 5120` BF16 投影的 `4 × 128` 配置为 0.187120 ms，`16 × 256` 为 0.130672 ms（1.432×）。每个候选在测量前都通过了 CPU 数值参考。

## 复现

```sh
bash tools/bench/cuda-baseline.sh 17408 5120 artifacts/cuda-confirmation 16 256
```

脚本拒绝覆盖已有输出目录，并保存二进制与源码哈希、工具链、构建日志、GPU 前后快照与原始样本。构建前提与 GPU 遥测位置见 [CUDA 性能指南](../../docs/guides/cuda-performance.md)。

## 测量边界

- 计时使用 CUDA events，包含输出分配、清零、同步提交以及可能的主机提交间隙，不是孤立的 kernel 时间。
- debug 与 release、冷缓存与热缓存的结果必须分别保存，不能混合比较。
- 合成 BF16 矩阵不代表 FP8/NVFP4 模型吞吐，也不包含 activation 量化、prefill、模型 tokens/s、TTFT 或 MTP 接受率。
- 测试机为共享桌面 GPU、未锁定时钟；这些记录用于开发对比，不构成性能保证。
- 没有候选被设为运行时默认值；选择执行默认配置前仍需完整模型验收。
