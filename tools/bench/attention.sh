#!/usr/bin/env bash
# Build first, then benchmark immutable binaries in alternating order on one GPU.
set -euo pipefail
cd "$(dirname -- "$0")/../.."
out=${INFER_ATTENTION_GATE_OUT:-artifacts/attention-gate}
mkdir -p "$out"
results=$(mktemp -d "$out/results-XXXXXXXX")
echo "Attention gate evidence: $results"
python3 -m unittest discover -s tools/attention -p 'test_attention_gate.py'
if [[ -n "${ATTENTION_PYTHON:-}" ]]; then
    "$ATTENTION_PYTHON" tools/attention/attention_fixtures.py --out "$out/fixtures.safetensors"
else
    uv run --with torch --with safetensors python tools/attention/attention_fixtures.py --out "$out/fixtures.safetensors"
fi
cargo test --locked --release -p infer-backend-cuda --features cuda --lib --no-run \
    --message-format=json > "$out/native-build.jsonl"
native=$(python3 - "$out/native-build.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
paths = [row['executable'] for row in rows if row.get('executable') and row.get('profile', {}).get('test') and row.get('target', {}).get('name') == 'infer_backend_cuda']
if len(paths) != 1:
    raise SystemExit('expected exactly one native test executable')
print(paths[0])
PY
)
CUDAFORGE_THREADS=2 CARGO_TARGET_DIR=target/attention-candidate cargo build --locked --release \
    --manifest-path tools/bench/attention/Cargo.toml
CUDAFORGE_THREADS=2 CARGO_TARGET_DIR=target/attention-candidate cargo clippy --locked --release \
    --manifest-path tools/bench/attention/Cargo.toml --all-targets -- -D warnings
python3 tools/attention/run_attention_gate.py --fixtures "$out/fixtures.safetensors" \
    --native "$native" --candle target/attention-candidate/release/infer-attention-candidate \
    --out "$results"
