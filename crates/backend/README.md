# Backend 分组

支持架构仅 NVIDIA CUDA 与 Metal。CPU 对照位于 testing，默认发布版不构建或注册。设备专属代码按执行接口管理：

```text
crates/backend/
  api/          # 设备描述与 CPU/GPU 交接协议
  kernel-api/   # kernel 注册与选择
  cuda/         # NVIDIA 首要生产目标；原生实现留待 5090 迁移
  metal/        # macOS GPU 验证执行器、MSL kernels、受限 FFI
crates/testing/cpu/
  host/         # 仅测试：权重驱动 CPU 正确性对照
  reference/    # 仅测试：旧 fixture 对照
crates/foundation/ir/src/hardware/
  mod.rs        # 后端判别、共有能力、共有兼容性接口
  cuda.rs       # NVIDIA architecture/capabilities/CUDA requirements
  metal.rs      # Metal capabilities/requirements
crates/foundation/ir/src/testing.rs  # feature gate 下的测试 CPU 能力
crates/service/cli/src/backend/
  mod.rs        # 选择、共同 provider 转发与 ticket 匹配
  cuda.rs
  metal.rs
  testing.rs    # feature gate 下的 CPU adapter
```

crate 名称保留 infer-backend-host、infer-backend-reference、infer-backend-metal，外部 Rust import 不因目录重组改变。workspace 路径和 kernel source map 已同步。

执行器实现 BackendProvider 的状态生命周期、submit/poll、身份、成本计时和 checkpoint；模型/调度通过统一合约调用。KernelProvider 只注册本后端可执行 kernel。设备缓冲、票据及 unsafe FFI 留在本组；共享 KvCacheManager 统一管理页 ledger、prefix acquire/publish/evict、COW/append/rollback、copy pins 与容量证据，内部复用 BlockPool / PrefixCache，设备复制由 backend 执行。

添加 backend 时，在本目录新增独立 crate，在 hardware 子模块定义它的专属能力/要求，再注册 BackendKind / DeviceBackend / BackendRequirements 与 CLI 选择 adapter。公共结构不添加厂商 Option 字段。现阶段为编译期封闭枚举，动态加载 ABI 尚未实现；新增 variant 必须通过编译器穷尽匹配检查，不能悄悄退到 CPU。

CUDA 接入边界见 [CUDA 迁移约定](cuda/README.md)，KV 合约见 [KV Cache Manager](../../docs/architecture/kv-manager.md)，公共加载器与 Metal forward 见 [模型执行](../../docs/guides/model-execution.md)。

默认生产依赖树不含 infer-backend-host / infer-backend-reference，默认 workspace members 排除 CPU testing crates。运行测试使用 cargo test --workspace --features infer-cli/test-backends；测试 CLI 仅通过 --backend test-cpu 显式进入。正式 supported / auto_priority 始终为 CUDA/Metal，auto 无可用 GPU 时返回错误。
