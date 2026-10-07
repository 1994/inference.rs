# ModelScope real-data benchmark

## Sources and reproducibility

- [GSM8K](https://modelscope.cn/datasets/AI-ModelScope/gsm8k): `main/test`, arithmetic word problems with reference answers.
- [ShareGPT](https://modelscope.cn/datasets/AI-ModelScope/sharegpt_gpt4): `sharegpt_zh_38K_format.jsonl`, first human turn of Chinese conversations. Assistant responses are not correctness labels.

`tools/bench/prepare-modelscope.py` verifies fixed source SHA256 hashes, records revisions and row IDs, and samples with a fixed seed. Download URLs are constructed from `SOURCES` in that script. The manifest records the eligible population and length filter; prompts are never silently truncated. These are fixed subsets, not claims of completing the entire original datasets.

Example (Python dependencies live in the external benchmark environment):

```sh
artifacts/vllm-compare/bin/python tools/bench/prepare-modelscope.py \
  --gsm8k /tmp/gsm8k-test.parquet --sharegpt /tmp/sharegpt-zh.jsonl \
  --per-dataset 16 --max-chars 2000 --output artifacts/modelscope-suite.json
bash tools/bench/safe-run.sh target/release/examples/cuda-model-smoke /path/to/model unused 512 \
  --device-graph --prefill-batch 3 --dataset artifacts/modelscope-suite.json --mtp 0 --thinking false \
  --temperature 0 --presence-penalty 0 > artifacts/rust-suite.json
bash tools/bench/safe-run.sh artifacts/vllm-compare/bin/python tools/bench/vllm-dataset.py \
  --input artifacts/rust-suite.json --output artifacts/vllm-suite.json --mtp 0
python3 tools/bench/summarize-dataset.py artifacts/rust-suite.json \
  artifacts/vllm-suite.json --output artifacts/dataset-summary.json
```

Repeat each implementation with MTP disabled and depth 2, sequentially on an otherwise idle GPU. Change only `--mtp` between paired runs. Remove temperature/presence overrides for a separate model-default sampling experiment; retain an identical manifest, thinking mode and generation budget. Repeat measured suites at least three times before making optimization decisions.

## Metrics

The model stays resident. Each run warms one complete request and resets request state. Load and warmup are excluded from measured suite wall time. Concurrency is explicitly 1. vLLM replays Rust's exact input token IDs and resolved sampling settings; prefix caching is disabled.

- Request throughput = completed requests / suite wall seconds.
- Output throughput = actual output tokens excluding EOS / suite wall seconds.
- Total token throughput = (input tokens + output tokens excluding EOS) / suite wall seconds. This is reported separately from output throughput.
- Report length-limited outputs and completion counts. A capped response is not necessarily a completed answer. Partial runs must not enter the ranking.

Do not average individual request token/s. Different generated lengths and quantization numerics can change runtime; compare output counts and quality alongside throughput. This harness does not measure streaming TTFT/TPOT or concurrent serving capacity.

## Current implementation limits

The Rust example runner with `--device-graph` executes projections, auxiliary operations and recurrent state on CUDA; sampling remains on the CPU. MTP uses batched verification with device prefix checkpoints; `--sequential-verify` selects the control. `--prefill-batch 3` uses fused GEMV and a dedicated prompt graph, not tensor-core GEMM. The experimental `--prefill-batch 32` path uses BF16 tensor-core GEMM with F32 accumulation and masked padding; record this different prompt precision explicitly. The default precision is F32 activations/KV with weight-only quantization; `--fp8-kv` opts into model-scaled target KV. Without `--device-graph`, the legacy diagnostic path still uses CPU auxiliary operations. vLLM can use different activation quantization and optimized GPU execution. Results characterize these current implementations, not equal-precision kernel performance or an architectural limit of Rust/cuTile. ShareGPT throughput does not establish answer correctness; GSM8K accuracy needs answer extraction and sufficient generation budgets before quality claims.

## Hardware telemetry

Wrap every benchmark with `bash tools/bench/safe-run.sh /usr/bin/python3 tools/bench/hardware-monitor.py --output <new-telemetry.jsonl> -- <command>`. The collector refuses execution without a verified cgroup memory ceiling. It samples once per second and rejects overlapping jobs instead of queuing them. Output files must be new. The benchmark's stdout can be redirected to its own JSON report. Collect on the same GPU with the same monitor interval for all implementations.

The safety wrapper requires user systemd and cgroup v2, checks for 48 GiB available RAM at admission, sets `MemoryHigh=28G`, `MemoryMax=32G`, `MemorySwapMax=0`, `OOMPolicy=kill`, a 30-minute deadline, and whole-cgroup termination. The actual child checks the effective limits before executing the command. Compilation uses `MAX_JOBS=1`, `NVCC_THREADS=1`, `FLASHINFER_NVCC_THREADS=1`, `TORCHINDUCTOR_COMPILE_THREADS=1`, and `CARGO_BUILD_JOBS=1`. The monitor aborts when host available memory falls below 8 GiB. The printed unit name can be stopped with `systemctl --user stop <unit>`. These controls reduce host OOM risk; they do not bound device VRAM or memory consumed by unrelated applications. Do not bypass a failed limit check.

On 2026-10-06 the original unbounded FlashInfer FP4 JIT launched 16 CUDA compilation processes and triggered global OOM, killing desktop applications. That interrupted run and the earlier overlapping GPU run are invalid performance samples. Warmup/JIT must complete under the protected wrapper before measurements; any cgroup OOM, memory throttling or timeout must be reported rather than silently retried with higher limits.

Run `python3 tools/bench/summarize-hardware.py --report <report.json> --telemetry <telemetry.jsonl> --output <summary.json>` to select only the measured suite window and summarize:

- GPU and memory-controller busy time, VRAM allocation, power and power limit.
- Temperature, SM/memory clocks, PCIe link width/generation and clock event reasons.
- Process-tree CPU use, resident memory and host memory/load snapshots.

GPU counters are device-wide, including desktop processes. Memory-controller utilization is not achieved memory bandwidth. Sampled CPU process-tree totals may miss short-lived children. The collector does not change clock or power settings. Use Nsight Systems separately for launch gaps, copies and synchronization; use Nsight Compute on representative kernels for DRAM GB/s, achieved occupancy, Tensor Core activity, register pressure and spills. Profiler overhead must not be included in throughput rankings.
