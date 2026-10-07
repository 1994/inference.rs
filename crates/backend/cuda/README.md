# NVIDIA CUDA 后端

`infer-backend-cuda` 使用 NVIDIA cuTile Rust 0.4.0 实现设备 kernel、驻留执行图、权重加载和 `BackendProvider`。模型拓扑由 provider 提供，后端按设备能力、dtype 和形状选择具体实现，不按模型名分派。

## 构建与运行

需要 Linux、Rust、CUDA Toolkit 开发头文件、libclang 及标准 C 头文件；执行还需要 NVIDIA driver 与支持目标设备的 `tileiras`。打包 CI 使用 CUDA 13.2 开发头文件，本地设备验证使用 CUDA 13.4。构建通过不代表目标硬件与 JIT 工具链已验收。

```sh
export CUDA_TOOLKIT_PATH=/path/to/cuda
cargo build --locked --release -p infer-cli --features cuda
make check-cuda
```

bindgen 找不到 C 标准头时，将实际主机 include 路径传给 `BINDGEN_EXTRA_CLANG_ARGS`，例如 `-isystem /path/to/gcc/include`；该路径不能照搬其他机器。默认 CLI 不启用 CUDA feature，普通 hosted Rust 门禁不需要 Toolkit。

## 执行边界

- BF16/FP8/NVFP4 投影、设备驻留 prefill/decode、状态/图复用与受限连续解码批处理。
- 可选贪心 MTP，由能力声明、draft 深度、容量与采样参数共同约束；输出仍需与关闭 MTP 的 target 路径对齐。
- 通用 dense attention 位于 `attention`，契约来自 `infer-kernel-api`；Candle 只存在于隔离 benchmark。
- 图像视觉塔与 MRoPE 通过专用示例验证，使用范围和复现方法见 [图像指南](../../../docs/guides/vision.md)。

当前同步 provider、设备采样、跨设备和完整服务性能的限制见 [Backend](../../../docs/architecture/backends.md)。kernel 正确性不等于完整模型质量或生产吞吐。

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


## 验收入口

| 入口 | 用途 |
|---|---|
| `make check-cuda` | 严格 Clippy、测试、Rustdoc 与 GPU kernel 检查 |
| `make check-attention` | 独立 F64 oracle 与 Candle 数值/性能门禁 |
| `cuda-resident-check` / `cuda-provider-check` | 驻留执行图、加载与 provider 回归 |
| `cuda-model-smoke` | 完整模型诊断与 MTP 对照 |
| `cuda-mma-probe` / `cuda-mlp-check --pdl` | lowering 与可选 PDL 能力探针 |

示例先通过 `cargo build --locked --release -p infer-backend-cuda --features cuda --example <name>` 构建，再通过 `tools/bench/safe-run.sh` 执行。数值与性能测量规范见 [CUDA 性能指南](../../../docs/guides/cuda-performance.md)。

## 职责

模型包负责权重格式、scale、shape 和执行图；`LinearStrategy` 提供合法候选；设备层管理 CUDA 生命周期与 typed buffer；kernel 负责计算，executor 负责提交、完成、状态和回滚。所有权与 checkpoint 遵守 `BackendProvider` / `KvCacheManager` 契约，prefill GEMM 与 decode GEMV 独立调优。
