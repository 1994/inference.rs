# NVIDIA：H200 生产、RTX 5090 测试

## 硬件与精度边界

生产目标是 Hopper H200（sm_90），本地测试是 Blackwell RTX 5090（sm_120）。cuTile Rust 同时支持两者，但 Hopper 至少需要 Tile IR 13.3；原生 FP4 类型与 block-scaled MMA 只有 Blackwell 支持，不能按架构编号大小推断全部功能。参考 [cuTile 兼容矩阵](https://nvlabs.github.io/cutile-rs/main/reference/compatibility.html)。

当前设备层真实查询 SM。`CudaTarget::supports_compute` 只回答"这块卡能原生计算哪些 dtype"；**用哪种精度由模型 provider 的 `PrecisionPolicy` 声明**（NVFP4 checkpoint 声明 `Preferred([nvfp4, bf16])`）：5090 命中原生 FP4 保留 packed，Hopper 回落到 BF16 展开。`CudaTarget::require_native_nvfp4` 仍是 native FP4 API 的能力门禁。直接调用 native FP4 API 在 Hopper 上仍返回 Unsupported。**转换实现已接入模型 runner，但没有 H200 实机端到端验证，不能声称生产验收完成。**

2026-10-06 已完成 [sm_120 scaled MMA 实测](../validation/cuda-sm120-mma.md)：cuTile 0.4.0 / CUDA 13.4.92 的 release 大 tile 生成 `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`，小 tile 即使数值正确也可能降为标量指令。`require_native_nvfp4` 只校验本实现的 FP4 类型路径，不能充当所有 tile 的 Tensor Core 保证。

| 项目 | H200 | RTX 5090 |
|---|---|---|
| 主要候选精度 | FP8 / BF16 Tensor Core | NVFP4 / FP8 / BF16，按阶段比较 |
| 当前 NVFP4 包 | 已接入加载时 NVFP4→BF16 转换；FP8 channel 权重保留 | 可使用原生 FP4 类型 |
| 验证范围 | 需要生产卡真实测量 | 本地 kernel correctness 与性能开发 |
| 调优缓存 | sm_90 独立记录 | sm_120 独立记录 |

把 NVFP4 反量化到 BF16 不会恢复原始权重精度，再量化到 FP8 还会额外增加误差。有原始模型时，生产 FP8 导出应优先从原始权重校准。加载器根据硬件策略在内存中转换，不覆盖模型文件。每个转换矩阵的 BF16 staging 限额为 1 GiB，检查 shape、block scale、global scale 和有限值；packed 数据与 scale 的读取仍受既有预算限制。NVFP4 数据部分从每值 0.5 字节展开为 2 字节，显存规划不能沿用 5090 packed 权重大小。

## 优化次序

[MLP PDL 验证](../validation/cuda-pdl.md)已覆盖一条可选 producer/consumer 边及三种权重存储。PDL 默认关闭，收益需单独测量，不能代替整步 graph 或消除主机提交开销。VMM `set_access` 是映射访问权限控制，与 L2 access-policy window 是不同机制。

| 优先级 | 改动 | 当前问题 / 验收 |
|---|---|---|
| P0 | 按硬件、精度与执行阶段分派 | `target.rs` 已隔离 SM 与 NVFP4 存储策略；生产服务执行分派待接入 |
| P0 | 预分配 workspace、异步依赖、CUDA Graph | `device.rs` 的 GEMV 每次分配/清零并同步，当前基线包含这些成本 |
| P0 | prefill 与 MTP verify 使用 Tensor Core GEMM | 32-token prefill 已使用 BF16 MMA 并有 SASS/短验证；decode 和 MTP verify 仍为 GEMV |
| P1 | H200 TMA、异步流水与合适的 tile/warp 配置 | 检查编译产物与 Nsight 指标，而不只看源码 hint |
| P1 | 融合 residual/RMSNorm、gate/up/SwiGLU、QK norm/RoPE | 同时评估访存、寄存器、spill 与数值误差 |
| P1 | hybrid 模型的卷积与 Delta 状态更新 | prefill 分块与逐 token decode 分开，避免每步搬回主机 |
| P1 | paged KV、长上下文 attention、批处理 | 记录上下文长度、KV 精度、batch 与尾延迟 |
| P1 | MTP draft/verify graph 与动态深度 | 净 tokens/s、接受率、额外耗时、拒绝后的状态回滚 |
| P2 | 多卡与通信计算重叠 | 先确认 H200 卡数、NVLink 拓扑与实际负载 |

Hopper 的 TMA 可以减少搬运指令和寄存器压力，并支持计算与数据搬运重叠；cluster 配置同样受 occupancy 约束，应实测而非统一调大。参考 [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)。

## 设计隔离

已有边界：`ProjectionCatalog` 负责模型元数据，`CudaTarget` 负责设备架构识别，`LinearStrategy` 负责候选配置，`CudaDevice` 负责 CUDA 生命周期，kernel 实现计算。

下一阶段分别引入精度计划、prefill/decode/MTP 执行计划与实测调优记录，避免把 H200/5090 的分支放进模型循环或数学 kernel。调优键至少包含实际 GPU/SM、编译器与 kernel 版本、dtype、布局、shape、batch、执行阶段与上下文范围。

5090 的结果只能证明该设备上的行为。H200 需要单独的数值验证、release 基线、显存峰值与服务指标。`make check-cuda` 包含当前的 Blackwell FP4 验收，会在 H200 上明确失败，不能把它标成 H200 全套验收通过。
