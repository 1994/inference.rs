# Backend 能力与扩展

正式设备架构为 NVIDIA CUDA / Metal，CPU Host/Reference 位于 `testing/cpu`，仅在 `test-backends` feature 下启用。平台目标与实现缺口见[状态表](../design/status.md)。

## 能力与目录

`DeviceCapabilities.backend` 使用带 payload 的 `DeviceBackend`：`Cuda(NvidiaCapabilities)` 或 `Metal(MetalCapabilities)`。没有平行的 `nvidia: Option` 身份；`CapabilityRequirements.backend` 同样携带专属要求，跨 backend 匹配拒绝。

| 分组 | 内容 |
|---|---|
| Common | device、dtype、memory 与公共诊断 |
| CUDA | compute capability/SM、Tensor Core、warp、graph、TMA/clusters、pinned/IPC/NVLink/GPUDirect |
| Metal | SIMD、storage 与 Apple GPU 专属要求 |
| Backend API / SPI | compile、allocation、submit/poll、资源确认、recipe 与生命周期 |

能力定义放在 IR hardware 对应子模块，执行器、loader 与 device FFI 放在 `crates/backend/<name>`；选择 adapter 放在 CLI 的 backend 子模块。[目录约定](../../crates/backend/README.md)定义新增 backend 的入口。

## 选择与隔离

发布版 `--backend auto|cuda|metal`；auto 按 CUDA → Metal 选择已实现、可用的设备，无可用设备返回 Unsupported。当前 CUDA 未实现，显式选择也返回 Unsupported。测试 CPU 不进入 auto 和 supported catalog。

ModelIr/DataflowGraph 描述共有语义，KernelRegistration/ExecutionProgram 绑定 backend。编译和执行均校验能力、精度、kernel 和 program target；换 backend 重新编译，checkpoint 不直接跨 backend 恢复。

Runtime 通过静态 BackendProvider 协调资源与 tickets。具体设备 buffers/handles 留在 backend，公共 trace/probe/stats 留在 IR；Metal 不依赖 CPU executor。

## Metal 执行契约

Metal 在加载期上传 F32/BF16/F16 权重、编译 MSL pipelines，预绑定节点并按 lifetime 分配 scratch。计算/状态为 F32；每个 compute flight 独占共享 scratch，当前深度为 1。票据绑定 executor owner，完成后才读取 shared output。

shared storage 用于 CPU/device 可见数据，private storage 用于设备私有数据；共享地址仍遵守同步与 reader 生命周期。raw pointer 与系统 selector 限制在 device FFI 边界。[Metal 同步规则](https://developer.apple.com/documentation/metal/resource-synchronization)

模型算子和 retention 见[模型执行](../guides/model-execution.md)，KV/COW/prefix 与物理 checkpoint 见[KV Manager](kv-manager.md)。GPU profile 记录 command-buffer 时间，OpTrace 记录 CPU encoding，不能当作逐 kernel GPU counters。

## 新增 backend

1. 在对应 group 定义能力/要求与 loader，公共结构不增加独立厂商字段。
2. 实现编译、权重上传、状态报价/recipe、资源确认、submit/poll 与 fence/reader 契约。
3. 注册匹配目标的 kernels，并通过独立数值、状态 invariance、取消/排空与跨 owner 拒绝测试。
4. 接入 CLI catalog、硬件门禁、profile 和原始性能报告；未经执行的能力保持明确 Unsupported。
