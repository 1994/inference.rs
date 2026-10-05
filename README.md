# inference.rs

[![Quality](https://github.com/1994/inference.rs/actions/workflows/ci.yml/badge.svg)](https://github.com/1994/inference.rs/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.90%2B-orange?logo=rust)](rust-toolchain.toml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

用 Rust 编写的模型推理引擎：直接加载 Hugging Face Safetensors 模型，在一个二进制里完成推理、服务与诊断。目标是单节点 SLO 吞吐对照 vLLM、SGLang 与 TensorRT-LLM。

## 特点

- **纯 Rust，无 Python 运行时**：直接读取 HF 模型包，不需要 Python、PyTorch 或现场编译，部署就是一个二进制。
- **不止于文本生成**：同一个引擎原生支持 Generate、Embed、Rerank 和 Decision（分类 / 打分 / 决策），不必拼装多个服务。
- **为并发服务而建**：paged KV、prefix cache、copy-on-write、多租户调度、SLO 感知准入与重算抢占，配合有界队列、背压、取消和在途资源排空。
- **可确定复现**：quiescent checkpoint 与 journal replay 能从请求、状态、成本三层重建一次执行，线上问题可以在本地回放。
- **自带诊断 Agent**：通过 JSON-RPC 暴露调度决策、状态所有权与源码关联等 typed 证据，并能在隔离环境运行 baseline / candidate 配置实验。
- **观测与正确性内建**：Prometheus、Chrome Trace、OTLP JSON 开箱可用；每项能力都有独立于实现的 golden / 参考验证和零分配门禁。

> **项目处于开发阶段。** 当前可在 Apple GPU 上运行仓库自带的微型混合模型；CUDA 算子已验证、完整执行器尚未接入，Qwen3.8-27B 全模型与生产性能目标仍待验收。详见[实现状态](docs/design/status.md)。

## 快速开始

需要 Rust 1.90+（仓库固定 1.99.0）和 C 编译器。

```sh
git clone https://github.com/1994/inference.rs.git
cd inference.rs

# 用仓库自带权重在本机验证数值
cargo build --locked -p infer-cli --features test-backends
target/debug/infer --backend test-cpu verify \
  --package examples/qwen-hybrid-tiny \
  --golden examples/qwen-hybrid-tiny/golden.json --atol 0.000002 --rtol 0.00002

# macOS 上运行样例请求
cargo build --locked --release -p infer-cli
target/release/infer --backend metal run \
  --package examples/qwen-hybrid-tiny \
  --requests examples/requests.json --config examples/runtime.json
```

`infer serve` 启动服务后，除原生接口外还提供 `/v1/models`、`/v1/chat/completions` 和 `/v1/completions`。完整参数见 `infer --help`。

## 文档

- [文档导航](docs/README.md)
- [开发与运行](docs/guides/development.md)
- [模型执行](docs/guides/model-execution.md)
- [实现状态](docs/design/status.md)

## 贡献与许可

欢迎提交 [Issue](https://github.com/1994/inference.rs/issues) 和 Pull Request；流程见[贡献指南](CONTRIBUTING.md)，安全问题见[安全策略](SECURITY.md)。

[Apache License 2.0](LICENSE)。
