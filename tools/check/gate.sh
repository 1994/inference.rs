#!/bin/sh
# One fail-fast gate used locally and by CI. Never suppress failed checks.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

rust_checks() {
    python3 -m unittest discover -s tools/package -p 'test_*.py'
    python3 -m unittest discover -s tools/attention -p 'test_attention_gate.py'
    cargo fmt --manifest-path tools/bench/attention/Cargo.toml --check
    python3 -m unittest discover -s tools/check -p 'test_layout.py'
    python3 tools/check/layout.py
    python3 tools/check/policy.py
    cargo fmt --all --check
    # Isolated production checks prevent test features from merging into the CLI.
    cargo clippy --locked -p infer-cli --no-default-features --all-targets -- -D warnings
    cargo clippy --locked -p infer-ir --no-default-features --all-targets -- -D warnings
    cargo clippy --locked --all-targets -- -D warnings
    # Hosted CI has no CUDA Toolkit. Excluding the CUDA package alone is not
    # enough: --all-features also enables it through infer-cli/cuda.
    # CUDA builds belong to the package matrix; device checks to check-cuda.
    cargo clippy --locked -p infer-backend-cuda --no-default-features --all-targets -- -D warnings
    cargo clippy --locked --workspace --exclude infer-backend-cuda --all-targets --features infer-cli/test-backends -- -D warnings
    cargo test --locked -p infer-cli --no-default-features
    cargo test --locked -p infer-ir --no-default-features
    cargo test --locked -p infer-backend-cuda --no-default-features
    cargo test --locked --workspace --exclude infer-backend-cuda --features infer-cli/test-backends
    RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --exclude infer-backend-cuda --features infer-cli/test-backends --no-deps
    cargo build --locked --release -p infer-cli --no-default-features
    cpu_checks
    # Golden execution is independent of the Rust implementation.
    cargo build --locked -p infer-cli --features test-backends
    target/debug/infer --backend test-cpu verify
    for fixture in qwen-hybrid-tiny qwen-hybrid-grouped; do
        target/debug/infer --backend test-cpu verify --package "examples/$fixture" \
            --golden "examples/$fixture/golden.json" --atol 0.000002 --rtol 0.00002
    done
    target/debug/infer --backend test-cpu run --package examples/qwen-hybrid-tiny \
        --requests examples/requests.json --config examples/runtime.json
}

cpu_checks() {
    cargo fmt --manifest-path tools/bench/cpu/Cargo.toml --check
    CARGO_TARGET_DIR=target/cpu-bench cargo clippy --locked --manifest-path tools/bench/cpu/Cargo.toml --all-targets -- -D warnings
    mkdir -p artifacts/cpu-completion
    CARGO_TARGET_DIR=target/cpu-bench cargo run --locked --release --manifest-path tools/bench/cpu/Cargo.toml > artifacts/cpu-completion/allocation.json
}

tool_checks() {
    uv tool run --from ruff==0.15.7 ruff check tools
    uv tool run --from ruff==0.15.7 ruff format --check tools
    go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.7 -shellcheck= .github/workflows/*.yml
}

security_checks() {
    # The only accepted advisory is documented with its removal condition in deny.toml.
    cargo deny --locked check --deny warnings
    cargo deny --locked --manifest-path tools/bench/cpu/Cargo.toml --config tools/bench/cpu/deny.toml check --deny warnings
    cargo audit --deny warnings --ignore RUSTSEC-2024-0436
    cargo audit --file tools/bench/cpu/Cargo.lock --deny warnings --ignore RUSTSEC-2024-0436
    go run github.com/zricethezav/gitleaks/v8@v8.24.3 dir . --no-banner --redact=100 \
        --ignore-gitleaks-allow --exit-code 1
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        go run github.com/zricethezav/gitleaks/v8@v8.24.3 git . --no-banner --redact=100 \
            --ignore-gitleaks-allow --exit-code 1
    fi
}

cuda_checks() {
    cargo clippy --locked -p infer-backend-cuda --features cuda --all-targets -- -D warnings
    cargo test --locked -p infer-backend-cuda --features cuda
    RUSTDOCFLAGS="-D warnings" cargo doc --locked -p infer-backend-cuda --features cuda --no-deps
    cargo run --locked --release -p infer-backend-cuda --features cuda --example cuda-kernels
}

metal_checks() {
    # Explicit backend selection makes missing GPU support fail instead of skipping.
    cargo build --locked --release -p infer-cli --no-default-features
    cargo build --locked -p infer-cli --features test-backends
    python3 tools/check/metal.py
    python3 tools/validation/cpu-open-loop.py
}

msrv_checks() {
    CARGO_TARGET_DIR=target/msrv cargo +1.90.0 check --locked -p infer-cli --no-default-features
    CARGO_TARGET_DIR=target/msrv cargo +1.90.0 check --locked --workspace --exclude infer-backend-cuda --all-targets --features infer-cli/test-backends
    CARGO_TARGET_DIR=target/cpu-bench-msrv cargo +1.90.0 check --locked --manifest-path tools/bench/cpu/Cargo.toml --all-targets
}

linux_checks() {
    if [ "$(uname -s)" = Linux ]; then
        cargo clippy --locked -p infer-core --all-targets -- -D warnings
        cargo test --locked -p infer-core placement:: -- --test-threads=1
    else
        # Compile Linux-only FFI and tests on development hosts; runtime tests belong to Linux CI.
        CARGO_TARGET_DIR=target/linux-check cargo clippy --locked -p infer-core --all-targets --target x86_64-unknown-linux-gnu -- -D warnings
    fi
}

linux_numa_checks() {
    if [ "$(uname -s)" != Linux ]; then
        echo 'NUMA acceptance requires a Linux host with memory-policy syscalls enabled' >&2
        exit 1
    fi
    cargo test --locked -p infer-core placement::native::numa:: -- --ignored --test-threads=1
}

case "${1:-all}" in
    all)
        linux_checks
        rust_checks
        tool_checks
        security_checks
        msrv_checks
        if [ "$(uname -s)" = Darwin ]; then metal_checks; fi
        ;;
    rust) rust_checks ;;
    tools) tool_checks ;;
    security) security_checks ;;
    metal) metal_checks ;;
    cuda) cuda_checks ;;
    cpu) cpu_checks ;;
    msrv) msrv_checks ;;
    linux) linux_checks ;;
    linux-numa) linux_numa_checks ;;
    attention) bash tools/bench/safe-run.sh --memory-gib 16 bash tools/bench/attention.sh ;;
    *) echo "usage: $0 [all|rust|tools|security|metal|cuda|cpu|msrv|linux|linux-numa|attention]" >&2; exit 2 ;;
esac
