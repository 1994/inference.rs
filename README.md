# ml-llm

Rust 推理引擎：由 CPU 管理请求、调度、KV 页所有权和异步交接，由独立 GPU backend 执行模型。Linux 是首要生产平台，NVIDIA CUDA 是首要设备 backend；macOS 使用 Metal 做本地验证。CPU 执行器仅用于正确性对照，不进入默认生产依赖。

## 代码入口

```text
crates/
  foundation/   core · ir · spi             基础类型、IR 与扩展合约
  backend/      api · kernel-api · metal    设备执行；cuda 保存迁移约定
  model/        package · compiler          权重包、文本资产与执行图编译
  engine/       state · scheduler · workloads · runtime
  diagnostics/  observe · quality           观测、诊断与质量验证
  service/      frontdoor · agent · cli      服务入口、诊断 Agent 与命令行
  testing/cpu/  host · reference             显式启用的 CPU 对照
```

[代码布局与调用链](docs/architecture/layout.md)说明模块职责、依赖方向和改动归属。[文档导航](docs/README.md)汇总运行、架构、设计与验证资料；总体设计与模块契约见[技术方案](docs/design/technical-plan.md)。

## 运行

```sh
cargo build --locked --release -p infer-cli --no-default-features
target/release/infer --backend metal run --package examples/qwen-hybrid-tiny \
  --requests examples/requests.json --config examples/runtime.json
```

macOS 需要可用的 Metal 设备。服务、模型加载、文本与测试对照命令见[开发指南](docs/guides/development.md)。

## 验证

```sh
make check-rust     # 布局、严格 lint、测试、文档、独立 CPU golden 与分配检查
make check-tools    # Python 与 CI 配置
make check-msrv     # Rust 1.90
make check-linux    # Linux placement 原生测试或跨平台编译检查
make check-metal    # 真实 GPU、服务、KV 压力与短负载验证
make check-security
```

完整入口为 `make check`，规则见[质量门禁](docs/guides/quality-gates.md)。[工具导航](tools/README.md)按 `tools/check`、`tools/validation`、`tools/fixtures`、`tools/bench` 分组；运行产物位于 `artifacts`，模型样例位于 `examples`。

## 当前边界

微型混合模型、Metal paged KV 和 CPU 调度协议已有本机验证；完整 Qwen3.8-27B、RTX 5090 CUDA、全引擎零分配、多 GPU flight 与远程 PD 尚未验收。CPU 目标负载的 P99 与吞吐仍未达标，具体结果和剩余项见[CPU 验证](docs/validation/cpu.md)及[设计状态](docs/design/status.md)。
