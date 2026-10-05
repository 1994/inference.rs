# NVIDIA CUDA 迁移位置

首要设备 RTX 5090，首要模型 Qwen3.8-27B。用户已要求 CUDA 部分在迁移至测试环境后实现；此目录当前不包含执行器 crate，CLI 明确返回 Unsupported。

专属类型位于 infer_ir::hardware::cuda：NvidiaArchitecture、NvidiaCapabilities、CudaRequirements。DeviceBackend::Cuda 绑定这些属性，Metal/CPU 无法携带 CUDA 属性或满足 CUDA 要求。

后续执行器在本组接入真实 device query、HBM buffer、stream/event、submit/poll 和 checkpoint。复用 infer-state 的物理 BlockLease {owner, index, generation} / BlockPool、PrefixCache，以及 BackendProvider 的 state_page_growth、kv_cache、prefix attachment、recompute preemption 合约；CUDA kernel 读取 u32 page table。设备字节与 stream lifetime 由 CUDA 自己管理，不能复制 Metal handle 或把 Host 结果作为 CUDA 验收。

物理页与 prefix 的入口已集中到 `infer_state::kv::KvCacheManager<P>`：prepare_append 返回 KvCopy，manager 负责引用、COW、淘汰、rollback 与容量证据，backend 负责 device buffers/复制。CUDA 接入时使用自己的 KvPrefix snapshot，并把 shared-tail copy pins 保留到真实 event 完成；多 stream 要逐 flight 管理 pins。模型加载使用 `infer_models::WeightTarget`，异步 H2D 必须在自己的 staging budget 与 fence 合约内消费 loader 借用切片。当前 Metal 的共享 tile prefill/F32 compute 不代表 CUDA precision 或性能已经实现。

BF16/量化、CUDA Graph、批量 attention 和 CUPTI 在真实设备上验证后再声明支持；当前 Metal F32 数值结果只用作公共语义对照。
