# inference.rs

[![Quality](https://github.com/1994/inference.rs/actions/workflows/ci.yml/badge.svg)](https://github.com/1994/inference.rs/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.90%2B-orange?logo=rust)](rust-toolchain.toml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

A native Rust inference engine for NVIDIA CUDA and Apple Metal. It loads Safetensors model packages, compiles them into device-resident execution graphs, and serves native token and OpenAI-compatible HTTP/SSE APIs from a single `infer` binary.

> **Status: active development.** Model coverage is limited and production performance is not yet certified. There is no CPU inference backend. The engine is suitable for development, fixture validation, and controlled benchmarking; it is not a drop-in replacement for vLLM or a general-purpose CPU runtime.

## What it provides

- **One deployable binary** for model loading, execution, HTTP serving, OpenAI compatibility, and diagnostics.
- **CUDA and Metal backends** with shared model semantics: the same package is compiled into a device-specific execution graph for either backend.
- **Bounded continuous decode batching** with admission control, backpressure, cancellation, and KV/state lifecycle management.
- **Greedy multi-token prediction (MTP)** on supported CUDA checkpoints, with automatic fast-profile selection when a validated configuration fits.
- **Native token API and OpenAI subset**: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, SSE streaming, tool calls, and strict-schema generation for the supported subset.
- **Operational tooling**: health and runtime inspection, event queries, Chrome Trace/OTLP export, checkpoint/replay, and a JSON-RPC diagnostic agent.

## Architecture

```text
CLI / HTTP+SSE / OpenAI subset / diagnostic agent
                        │
Runtime                 │  admission, scheduling, KV/state lifecycle,
                        │  cancellation, completion, observability
                        │
Model layer             │  package loading, provider registry,
                        │  execution graph, compiler
                        │
Backends                │  CUDA (cuTile / SIMT) and Metal (MSL) kernels,
                        │  device graphs, resource operations
                        │
Foundation              │  IR, SPI contracts, shared types
```

Model providers own architecture-specific behavior. Shared IR and kernel APIs describe operations independently of model names and GPU models. The scheduler operates on execution contracts and resource budgets, so model work and device work remain separable.

See [code layout](docs/architecture/layout.md), [backend capabilities](docs/architecture/backends.md), and the [runtime design](docs/architecture/runtime.md) for details.

## Requirements

| Component | Requirement |
|---|---|
| Rust | `1.99.0` pinned by `rust-toolchain.toml`; MSRV `1.90` |
| Linux / CUDA | NVIDIA GPU, CUDA Toolkit, `tileiras`, C compiler, and libclang |
| macOS / Metal | A Metal-capable Apple GPU and Xcode command-line tools |
| Model | A supported local Safetensors package with model config; tokenizer assets are required for text HTTP and OpenAI routes |

A CPU-only host can build and run the test and lint suites, but cannot execute a model. `auto` selects CUDA or Metal; if no supported GPU is available, startup fails instead of silently falling back to CPU.

## Quick start

### Build

```sh
git clone https://github.com/1994/inference.rs.git
cd inference.rs

# Native release binary with the platform backend:
make local-build
```

`make local-build` uses the same backend and feature plan as packaging: CUDA on Linux, Metal on macOS. For an explicit Cargo build:

```sh
# Linux + CUDA
cargo build --locked --release -p infer-cli --features cuda

# macOS + Metal
cargo build --locked --release -p infer-cli
```

CUDA builds need the environment described in [crates/backend/cuda/README.md](crates/backend/cuda/README.md). Direct Cargo builds require explicit feature selection because Cargo features are compile-time options.

### Serve

```sh
target/release/infer /path/to/model
```

The server listens on `http://127.0.0.1:8080`. Workspace size, input limits, KV budget, and CUDA memory admission are derived from the loaded model and device. Use `--listen`, `--gpu-memory-utilization`, or `--config` to override the defaults.

On supported CUDA checkpoints, an omitted execution profile may select BF16 prefill attention and MTP2 automatically. If that profile does not fit, startup retries the canonical profile without MTP. Pass `--num-speculative-tokens 0` to force canonical attention without speculative decoding.

### Make a request

Give the deployment a stable name for OpenAI clients:

```sh
target/release/infer /path/to/model --served-model-name qwen
```

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64
  }'
```

The OpenAI routes require a package with `tokenizer.json`; chat completions also require a chat template. Without those assets, the native token API remains available. See the [OpenAI API guide](docs/guides/openai-api.md) for supported parameters, streaming, tools, and error semantics.

### Inspect before serving

```sh
target/release/infer inspect-package --package /path/to/model
target/release/infer doctor
```

`inspect-package` validates shard headers, parameter bindings, and weight shapes. `doctor` reports the compiled backend catalog and host capability checks. Neither command proves that a model's full execution path or performance is supported.

## Supported models

The provider registry currently resolves Qwen-family text backbones:

- Qwen3 and Qwen3-MoE
- Qwen3-Next
- Qwen3.5 and Qwen3.5-MoE
- Qwen3-VL text backbones

Image and video inference paths are validated through dedicated examples; multimodal requests are not yet part of Engine/Scheduler service scheduling. Model and operator coverage is intentionally narrow: an arbitrary Hugging Face package is not guaranteed to load or run. Check [adding a model](docs/guides/adding-a-model.md), [model execution](docs/guides/model-execution.md), and the [MoE validation guide](docs/guides/moe-validation.md) before adapting a new package.

| Backend | Current scope |
|---|---|
| **CUDA on Linux** | Primary development target. Native Rust/cuTile and SIMT kernels, device-resident graphs, CLI/HTTP serving, bounded continuous decode, greedy MTP. The provider is synchronous and JIT-compiles kernels through the CUDA Toolkit. |
| **Metal on macOS** | Prefill/decode, KV and prefix caching, and service validation against repository fixtures. |
| **Host (no GPU)** | Unit/integration tests, lint, and scalar reference operators. No model execution backend. |

## Testing and packaging

```sh
make test              # Host workspace tests; no GPU required
make check-rust        # Lint, tests, Rustdoc, CPU allocation checks, golden verification
make check-tools       # Python linting and CI workflow checks
make test-cuda-hosted  # CUDA compile, lint, and CPU-side tests; needs the Toolkit, not a GPU
make test-cuda         # CUDA device checks; needs the Toolkit and a GPU
make test-metal        # Metal device checks; needs macOS and a GPU
make check-attention   # Numerical and performance comparison against Candle
```

`make check` is the full host gate. Device gates are separate because they require a specific GPU and driver.

Build platform packages with Zig and `cargo-zigbuild`:

```sh
make setup-build      # Install the pinned cargo-zigbuild; Zig must already be installed
make package TARGET=x86_64-unknown-linux-gnu.2.28
```

Packaging supports Linux CUDA and macOS Metal on x86_64 and AArch64. GitHub CI runs quality checks before building, verifying, and uploading the four target packages. Cross-compilation and CLI smoke tests do not establish GPU correctness or performance. See the [packaging guide](docs/guides/packaging.md).

## Repository layout

```text
crates/
  foundation/   Core types, IR, and provider contracts
  backend/      Shared kernel APIs, CUDA, and Metal
  model/        Model packages, recipes, and compiler
  engine/       State, scheduler, workloads, and runtime
  diagnostics/  Observability and quality tooling
  service/      HTTP frontdoor, diagnostic agent, and CLI
  testing/      Scalar reference operators for tests
docs/           Guides, architecture, ADRs, plans, and research
examples/       Small model fixtures and request examples
tools/          Checks, packaging, validation, and benchmarks
benchmarks/     Dataset manifests and machine-readable baselines
```

## Limitations

- **No CPU inference.** CPU paths exist for tests and reference operators only.
- **Limited model coverage.** Supported Qwen configurations are validated individually; arbitrary HF packages are not covered.
- **CUDA is synchronous.** There is no general asynchronous multi-flight execution path or full production SLO certification yet.
- **Multimodal serving is incomplete.** Image/video execution is exercised through examples, not the serving scheduler.
- **Performance validation is incomplete.** The generic attention path has not cleared the Candle performance gate, and the repository's current serving profiles have not passed the full performance contract.
- **DFlash2 is under development.** MTP is the production speculative path for supported CUDA checkpoints.

## Documentation and contributing

Start with the [documentation index](docs/README.md). In-depth guides are currently written in Chinese.

- [Development and operations](docs/guides/development.md)
- [Adding a model](docs/guides/adding-a-model.md)
- [Model loading and execution](docs/guides/model-execution.md)
- [OpenAI-compatible API](docs/guides/openai-api.md)
- [CUDA benchmarking and tuning](docs/guides/cuda-performance.md)
- [Quality gates](docs/guides/quality-gates.md)
- [Roadmap](docs/plans/README.md)
- [Contribution guidelines](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## License

[Apache License 2.0](LICENSE).
