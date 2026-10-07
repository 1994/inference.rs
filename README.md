# inference.rs

[![Quality](https://github.com/1994/inference.rs/actions/workflows/ci.yml/badge.svg)](https://github.com/1994/inference.rs/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.90%2B-orange?logo=rust)](rust-toolchain.toml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Rust 原生模型推理引擎：直接加载 Hugging Face Safetensors 模型包，推理、服务与诊断收进同一个二进制。单节点 SLO 吞吐目标对照 vLLM、SGLang 与 TensorRT-LLM。

## 为什么选择 inference.rs

- **Rust 原生运行时**：推理入口不依赖 Python、PyTorch，直接读取 HF 模型包（config、索引、Safetensors、tokenizer）。CUDA 当前使用 cuTile JIT，需要 CUDA Toolkit / tileiras；完整 AOT 发布制品仍待完成。
- **一个引擎覆盖多种工作负载**：Generate、Embed、Rerank、Decision（分类 / 打分 / 决策）原生共用同一条执行与调度流水线，无需拼装多个服务。
- **为并发服务而建**：paged KV、prefix cache、copy-on-write、多租户调度、SLO 感知准入与重算抢占；有界队列、背压、取消与在途资源排空内建于运行时。
- **可确定复现**：quiescent checkpoint 与 journal replay 能从请求、状态、成本三个层面重建一次执行，线上问题可以拿回本地回放。
- **自带诊断 Agent**：通过 JSON-RPC 暴露调度决策、状态所有权与源码关联等 typed 证据，可在隔离环境中运行 baseline / candidate 配置实验并按 SLO 裁决。
- **观测与正确性内建**：Prometheus、Chrome Trace、OTLP JSON 开箱可用；每项能力都配独立于实现的 golden / 参考验证与零分配门禁。

## 当前状态

> **项目处于开发阶段。**

- Apple GPU（Metal）可运行仓库自带的微型混合模型，覆盖 prefill / decode、prefix KV 与真实 HTTP 服务。
- CUDA：已实现 cuTile Rust 整步设备驻留图、辅助算子、状态缓存和 MTP 草稿图；RTX 5090 上通过真实 27B 模型与数据集验证。32-token BF16 tensor-core prefill 与同步 `BackendProvider` 已通过真实模型短验证；CLI/HTTP 服务和有界状态/图复用已实测；已消除逐 token metadata 临时上传同步，设备采样、服务内 MTP 和 prefill 辅助算子批量化仍未完成，详见[性能实测](docs/validation/cuda-resident.md)。
- Qwen3.8-27B 全模型与生产性能目标仍待验收。

已实现能力与剩余缺口的完整清单见[实现状态](docs/design/status.md)，测量边界见[验证记录](docs/validation/index.md)。

## 快速开始

环境要求：Rust 1.90+（仓库固定 1.99.0）与 C 编译器。

```sh
git clone https://github.com/1994/inference.rs.git
cd inference.rs
```

### 本机验证数值（无需 GPU）

CPU 对照后端仅在显式 feature 下可用，适合先确认环境正常：

```sh
cargo build --locked -p infer-cli --features test-backends
target/debug/infer --backend test-cpu verify \
  --package examples/qwen-hybrid-tiny \
  --golden examples/qwen-hybrid-tiny/golden.json --atol 0.000002 --rtol 0.00002
```

### 运行样例请求（macOS / Metal）

```sh
cargo build --locked --release -p infer-cli
target/release/infer --backend metal run \
  --package examples/qwen-hybrid-tiny \
  --requests examples/requests.json --config examples/runtime.json
```

### NVIDIA / CUDA 构建与服务

需要 Linux、NVIDIA 驱动、CUDA 13.4 Toolkit / tileiras 和 Rust。cuTile 首次捕获会编译 kernel，随后复用磁盘缓存。
以下配置用于本仓库的 RTX 5090 / Qwen3.8-27B-NVFP4 验证环境：

```sh
CUDA_TOOLKIT_PATH=/opt/cuda CARGO_BUILD_JOBS=1 cargo +stable build --release -p infer-cli --features cuda
bash tools/bench/safe-run.sh target/release/infer --backend cuda serve \
  --package /home/r/models/Qwen3.8-27B-NVFP4 \
  --config examples/cuda-runtime.json --device-memory-mib 30000 \
  --listen 127.0.0.1:8080
```

`--device-memory-mib` 限制权重与请求状态预算；默认 512 MiB 不适用于 27B 模型。
保护脚本还要求主机可用内存至少 48 GiB，并限制整个任务（含编译子进程）使用 32 GiB、禁用该任务 swap。
当前 CUDA 使用同步 BackendProvider 与 32-token BF16 prefill；GPU 连续批处理、设备采样及服务内 MTP 仍待接通。

### 启动服务

```sh
target/release/infer --backend metal serve \
  --package examples/qwen-hybrid-tiny --listen 127.0.0.1:8080
```

服务除原生 `/native/v1/*` 接口外，还提供 OpenAI 兼容的 `/v1/models`、`/v1/chat/completions` 与 `/v1/completions`。

## CLI 一览

```text
infer [--backend auto|metal|cuda|test-cpu] <command>
```

| 命令 | 用途 |
|---|---|
| `run` | 执行请求文件 |
| `verify` | 对照 golden 验证数值 |
| `serve` | 启动 HTTP/SSE 服务 |
| `agent` | 启动诊断 Agent（JSON-RPC） |
| `inspect-package` | 检查模型包权重绑定与内存需求 |
| `doctor` | 列出后端目录与可用性 |
| `benchmark` / `compare` | closed 测量 / 候选配置评估 |
| `replay` / `profile` | checkpoint 回放 / journal 性能分析 |

`--backend auto` 按 CUDA → Metal 顺序选择，无可用 GPU 时报错，不回退 CPU。完整参数见 `infer --help` 与[开发与运行](docs/guides/development.md)。

## 仓库结构

```text
crates/
  foundation/   core / ir / spi            身份、所有权、IR 与 15 类扩展接口
  backend/      api / kernel-api / metal / cuda
  model/        package / compiler         HF 包导入与 IR 编译
  engine/       state / scheduler / workloads / runtime
  diagnostics/  observe / quality          指标、trace、golden 与测量
  service/      frontdoor / agent / cli    HTTP/SSE、诊断 Agent 与命令行
  testing/      cpu 对照（仅测试 feature）
docs/           guides / architecture / design / validation / adr
examples/       微型模型包与请求样例
tools/          门禁、验证与基准工具
```

依赖方向与推理调用链见[代码布局](docs/architecture/layout.md)。

## 文档

- [文档导航](docs/README.md) — 指南、架构、设计与验证记录索引
- [开发与运行](docs/guides/development.md) — 构建、CLI、HTTP/SSE、checkpoint 与测量
- [模型执行](docs/guides/model-execution.md) — 模型包格式、权重加载、prefill / decode
- [技术方案](docs/design/technical-plan.md) — 总体目标、模块契约与验收标准
- [实现状态](docs/design/status.md) — 已实现能力与剩余缺口

## 参与贡献

欢迎提交 [Issue](https://github.com/1994/inference.rs/issues) 与 Pull Request。提交前请运行 `make check-rust`（完整门禁为 `make check`），流程详见[贡献指南](CONTRIBUTING.md)；安全问题请阅读[安全策略](SECURITY.md)。

## 许可

[Apache License 2.0](LICENSE)。

## RTX 5090 实测入口

原生路径只使用 Rust / cuTile；vLLM 仅作为独立环境中的外部对照。

```sh
CARGO_BUILD_JOBS=1 CUDA_TOOLKIT_PATH=/opt/cuda cargo +stable build --release \
  -p infer-backend-cuda --features cuda --example cuda-model-smoke

bash tools/bench/safe-run.sh target/release/examples/cuda-model-smoke \
  /home/r/models/Qwen3.8-27B-NVFP4 '只输出17乘23的结果。' 32 \
  --device-graph --prefill-batch 3 --mtp 0 --thinking false --temperature 0
```

不指定采样覆盖参数时，读取模型配置及已识别的模型推荐值；请求参数按字段覆盖。
`--device-graph` 同时启用 target / MTP 的设备图，输出报告列出精度、执行方式与限制。
这一路径尚不等价于生产 HTTP 服务；MTP 已支持批量验证与设备状态回退；吞吐与剩余限制见性能实测。

`--mtp 2` 启用两 token 草稿及批量验证；当前 5090 pilot 尚无净加速，因此不默认启用。`--fp8-kv` 显式启用模型标定 FP8 KV。

开发中的 `--prefill-batch 32` 使用 BF16 Tensor Core prefill，支持带掩码的尾块；其精度与 F32 路径不同，目前仅有数值及真实模型短验证，尚无正式吞吐结论。

平台打包入口：`make package TARGET=<rust-target>`，或 `make package-linux-cuda` / `make package-macos-metal`；由根 `build.rs` 管理契约，`cargo-zigbuild` 执行交叉编译。产物校验和 GPU 验收见[打包指南](docs/guides/packaging.md)。
