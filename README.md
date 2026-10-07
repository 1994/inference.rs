# inference.rs

[![Quality](https://github.com/1994/inference.rs/actions/workflows/ci.yml/badge.svg)](https://github.com/1994/inference.rs/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.90%2B-orange?logo=rust)](rust-toolchain.toml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

A native Rust inference engine for CUDA and Metal. Load supported Hugging Face Safetensors model packages, run inference, serve requests, and inspect execution from a single binary.

## Architecture

The engine separates model semantics, kernel implementation, and request scheduling:

```text
Model provider   → execution graph, weights, precision, and draft topology
Compiler         → graph validation, memory lifetimes, and kernel selection
Backend          → device kernels, fusion, execution, and resource operations
Runtime          → request scheduling, budgets, KV lifecycle, and completion
Service          → CLI, HTTP/SSE, and diagnostic agent
```

Model providers own architecture-specific behavior. Shared kernel APIs describe operations independently of model names and GPU models. CUDA and Metal backends implement acceleration and fusion, while the scheduler works with execution contracts and resource budgets.

The runtime provides bounded queues, backpressure, cancellation, prefix caching, checkpoint/replay, and observability. Generate, Embed, Rerank, and Decision workloads share the execution pipeline; each requires a compatible model and backend.

See the [code layout](docs/architecture/layout.md) for module boundaries and extension points.

## Current support

**Under active development. Production performance validation is incomplete.**

| Backend | Scope |
|---|---|
| CUDA on Linux | Native Rust/cuTile kernels, resident execution graphs, CLI/HTTP serving, constrained continuous decode batching, and greedy multi-token prediction (MTP) |
| Metal on macOS | Prefill/decode, KV and prefix caching, and service validation using the repository's small model fixtures |
| CPU reference | Numerical verification through the explicit `test-backends` feature |

CUDA currently uses a synchronous provider and runtime JIT compilation through the CUDA Toolkit / `tileiras`. Model and operator coverage is limited; an arbitrary Hugging Face package is not guaranteed to run.

The generic attention implementation has **not passed its Candle performance gate**. Candle is an isolated benchmark baseline, outside the production runtime. Image generation is validated through dedicated examples and is not yet integrated into multimodal service scheduling.

Support and measurements apply to their tested configurations. See [backend capabilities](docs/architecture/backends.md), [quality gates](docs/guides/quality-gates.md), and [image model validation](docs/guides/vision.md) for details.

## Quick start

Install Rust and a C compiler. The repository pins Rust **1.99.0**; the portable workspace has an MSRV of **1.90**. GPU backends require their platform SDKs and runtime dependencies.

```sh
git clone https://github.com/1994/inference.rs.git
cd inference.rs
```

### Verify without a GPU

Build the CPU reference backend and compare the included model against independent golden outputs:

```sh
cargo build --locked -p infer-cli --features test-backends
target/debug/infer --backend test-cpu verify \
  --package examples/qwen-hybrid-tiny \
  --golden examples/qwen-hybrid-tiny/golden.json \
  --atol 0.000002 --rtol 0.00002
```

### Serve with Metal

On a Mac with a supported Metal device:

```sh
cargo build --locked --release -p infer-cli
target/release/infer examples/qwen-hybrid-tiny
```

### Serve with CUDA

Configure the [CUDA build environment](crates/backend/cuda/README.md), then build with the `cuda` feature. Replace `/path/to/model` with a supported local model package:

```sh
cargo build --locked --release -p infer-cli --features cuda
target/release/infer /path/to/model
```

Once `infer` is on your `PATH`, the default entry point is simply:

```sh
infer /path/to/model
```

This starts the server at `http://127.0.0.1:8080`. The backend is selected automatically in CUDA → Metal order; a missing GPU produces an error. Workspace and input limits are derived from the loaded model, and an exposed KV pool sets the logical page budget. CUDA derives its memory budget from device capacity and available memory; Metal uses the device's recommended working set. CUDA projection autotuning is enabled by default and reuses matching cached measurements.

Optional flags override these choices, for example `infer /path/to/model --listen 127.0.0.1:9000`. An explicit `--config` preserves the supplied runtime budgets; the original `infer serve --package ...` syntax remains supported. CPU execution requires explicit selection.

Automatic sizing and measured tile selection do not guarantee optimal end-to-end performance for every device, model, or workload. Performance changes must still pass the benchmark gates; MTP remains opt-in until its benefit is established for the workload.

### Logs

Startup and request logs go to stderr through `tracing`, so a command's JSON result on stdout stays parseable. At the default `info` level the log reports the device and derived budget, the parameters in effect, the identified model and its payload, every load phase with its duration, and one line per HTTP request with status and latency. Engine events cover the request lifecycle (`request accepted`, `request finished` with token count, TTFT and end-to-end time), so a request can be followed without a profiler.

Set `INFER_LOG` (or `RUST_LOG`) to change the level or filter by target, for example `INFER_LOG=infer=debug` adds per-weight load progress and finer engine detail, and `INFER_LOG=infer::load=debug` keeps it to loading.

The server exposes native HTTP/SSE routes and a subset of the OpenAI API. See the [development guide](docs/guides/development.md) for CLI commands and memory budgets, and the [OpenAI API guide](docs/guides/openai-api.md) for supported routes and parameters.

## Testing and packaging

```sh
make check-rust       # Lint, tests, Rustdoc, CPU allocation checks, and golden verification
make check-tools      # Python lint and GitHub Actions validation
make check-cuda       # CUDA device checks; requires the Toolkit and a GPU
make check-metal      # Metal device checks; requires macOS and a GPU
make check-attention  # Numerical and performance comparison against Candle
```

Build platform packages with Zig and `cargo-zigbuild`:

```sh
make setup-build      # Install the pinned cargo-zigbuild; requires Zig already installed
make package TARGET=x86_64-unknown-linux-gnu.2.28
```

Packaging supports Linux CUDA and macOS Metal on x86_64 and AArch64. Install the Rust target and required platform headers or SDK before cross-compiling. GitHub CI runs quality checks before building, verifying, and uploading the four target packages. Cross-compilation and CLI smoke tests do not establish GPU correctness or performance.

See the [packaging guide](docs/guides/packaging.md) for prerequisites, artifact verification, and device acceptance.

## Repository layout

```text
crates/
  foundation/   Core types, IR, and provider contracts
  backend/      Shared kernel APIs, CUDA, and Metal
  model/        Model packages, execution recipes, and compiler
  engine/       State, scheduler, workloads, and runtime
  diagnostics/  Observability and correctness tooling
  service/      HTTP frontdoor, diagnostic agent, and CLI
  testing/      CPU reference backends
docs/           Guides, architecture, and accepted design decisions
examples/       Small model fixtures and request examples
tools/          Checks, packaging, validation, and benchmarks
benchmarks/     Dataset manifests and machine-readable baselines
```

## Documentation and contributing

Start with the [documentation index](docs/README.md). Detailed guides are currently in Chinese.

- [Add a model](docs/guides/adding-a-model.md)
- [Model loading and execution](docs/guides/model-execution.md)
- [CPU performance measurement](docs/guides/cpu-performance.md)
- [CUDA benchmarking and tuning](docs/guides/cuda-performance.md)
- [Contribution guidelines](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## License

[Apache License 2.0](LICENSE).
