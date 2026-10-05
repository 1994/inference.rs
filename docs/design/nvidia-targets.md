# NVIDIA：H200 生产、RTX 5090 测试

## 硬件与精度边界

生产目标是 Hopper H200（sm_90），本地测试是 Blackwell RTX 5090（sm_120）。cuTile Rust 同时支持两者，但 Hopper 至少需要 Tile IR 13.3；原生 FP4 类型与 block-scaled MMA 只有 Blackwell 支持，不能按架构编号大小推断全部功能。参考 [cuTile 兼容矩阵](https://nvlabs.github.io/cutile-rs/main/reference/compatibility.html)。

当前设备层真实查询 SM，FP4 launch 先经过 `CudaTarget` 校验，H200 会在启动前返回明确的 Unsupported。**H200 的转换/软件解包路径尚未实现，不能声称模型已经可以在 H200 上运行。**

| 项目 | H200 | RTX 5090 |
|---|---|---|
| 主要候选精度 | FP8 / BF16 Tensor Core | NVFP4 / FP8 / BF16，按阶段比较 |
| 当前 NVFP4 包 | 需要软件解包、转换为 FP8/BF16，或重新导出 | 可使用原生 FP4 类型 |
| 验证范围 | 需要生产卡真实测量 | 本地 kernel correctness 与性能开发 |
| 调优缓存 | sm_90 独立记录 | sm_120 独立记录 |

把 NVFP4 反量化到 BF16 不会恢复原始权重精度，再量化到 FP8 还会额外增加误差。有原始模型时，生产 FP8 导出应优先从原始权重校准。加载转换的内存占用、精度和推理耗时需要一起比较，不自动转换权重或覆盖模型文件。

## 优化次序

| 优先级 | 改动 | 当前问题 / 验收 |
|---|---|---|
| P0 | 按硬件、精度与执行阶段分派 | `target.rs` 已隔离 SM，完整执行分派待实现 |
| P0 | 预分配 workspace、异步依赖、CUDA Graph | `device.rs` 的 GEMV 每次分配/清零并同步，当前基线包含这些成本 |
| P0 | prefill 与 MTP verify 使用 Tensor Core GEMM | `kernels.rs` 目前是乘法加归约的参考实现，尚未使用 MMA |
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
