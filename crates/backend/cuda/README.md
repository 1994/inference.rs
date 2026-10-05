# NVIDIA CUDA 后端

GPU kernel 使用 NVIDIA cuTile Rust 0.4.0。host 与 device 程序都用 Rust 编写，由 CUDA Driver 与 Tile IR 负责执行和编译。

## 已验证范围

- dense BF16/F32 矩阵向量乘与显式 tile 选择。
- FP8 channel-scaled 与 NVFP4 packed-weight 的参考投影。
- RTX 5090 上全部搜索候选的逐元素数值检查，包括不整除的行 tile。
- 配套 CUDA event 基线：预热、冷 L2、原始样本与 A/B 比较。

量化参考 kernel 使用 F32 activation。动态 activation 量化、block-scaled Tensor Core 执行、完整模型执行与 MTP 推测执行**尚未实现**，CLI 也没有可用的 CUDA 模型执行器。kernel 验证不代表模型质量。

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
