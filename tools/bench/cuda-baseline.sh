#!/usr/bin/env bash
# Record exact build/environment identity next to the measured samples.
set -euo pipefail
if [[ $# != 3 && $# != 5 ]]; then
  echo 'usage: bash tools/bench/cuda-baseline.sh ROWS COLUMNS OUTPUT_DIRECTORY [TILE_ROWS TILE_COLUMNS]' >&2
  exit 2
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
rows="$1"
columns="$2"
report_dir="$3"
if [[ ! "$rows" =~ ^[1-9][0-9]*$ || ! "$columns" =~ ^[1-9][0-9]*$ ]]; then
  echo 'ROWS and COLUMNS must be positive integers' >&2
  exit 2
fi
# Refuse overwriting a prior baseline.
mkdir -- "$report_dir"
export CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_PATH:-/opt/cuda}"
profile="${CUDA_BASELINE_PROFILE:-release}"
case "$profile" in
  release) binary_dir=release ;;
  dev) binary_dir=debug ;;
  *) echo 'CUDA_BASELINE_PROFILE must be release or dev' >&2; exit 2 ;;
esac
cargo +stable build --locked -p infer-backend-cuda --features cuda \
  --profile "$profile" --example cuda-baseline > "$report_dir/build.log" 2>&1
binary="${CARGO_TARGET_DIR:-target}/$binary_dir/examples/cuda-baseline"
{
  date --utc --iso-8601=seconds
  rustc +stable --version --verbose
  "$CUDA_TOOLKIT_PATH/bin/tileiras" --version
  git rev-parse HEAD
  git status --short
  printf 'profile=%s\nrows=%s\ncolumns=%s\n' "$profile" "$rows" "$columns"
  printf 'CUDA_TOOLKIT_PATH=%s\nBINDGEN_EXTRA_CLANG_ARGS=%s\n' \
    "$CUDA_TOOLKIT_PATH" "${BINDGEN_EXTRA_CLANG_ARGS:-}"
  sha256sum -- "$binary" Cargo.lock
} > "$report_dir/environment.txt"
rg --files crates/backend/cuda crates/model/package Cargo.toml Cargo.lock \
  | sort | xargs sha256sum > "$report_dir/source-sha256.txt"
nvidia-smi -q > "$report_dir/gpu-before.txt"
"$binary" "$rows" "$columns" "${@:4}" > "$report_dir/samples.json"
nvidia-smi -q > "$report_dir/gpu-after.txt"
printf 'Saved CUDA baseline to %s\n' "$report_dir"
