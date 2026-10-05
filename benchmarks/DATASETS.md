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
target/release/examples/cuda-model-smoke /home/r/models/Qwen3.8-27B-NVFP4 unused 512 \
  --dataset artifacts/modelscope-suite.json --mtp 0 --thinking false \
  --temperature 0 --presence-penalty 0 > artifacts/rust-suite.json
artifacts/vllm-compare/bin/python tools/bench/vllm-dataset.py \
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

The Rust runner is a diagnostic: GPU projections, Rust CPU auxiliary operations, weight-only quantization and sequential MTP verification. vLLM can use different activation quantization and optimized GPU execution. Results characterize these current implementations, not equal-precision kernel performance or an architectural limit of Rust/cuTile. ShareGPT throughput does not establish answer correctness; GSM8K accuracy needs answer extraction and sufficient generation budgets before quality claims.

## Hardware telemetry

Wrap every benchmark with `python3 tools/bench/hardware-monitor.py --output <new-telemetry.jsonl> -- <command>`. The wrapper samples once per second and serializes benchmark wrappers with a workspace lock. Output files must be new. The benchmark's stdout can be redirected to its own JSON report. Collect on the same GPU with the same monitor interval for all implementations.

Run `python3 tools/bench/summarize-hardware.py --report <report.json> --telemetry <telemetry.jsonl> --output <summary.json>` to select only the measured suite window and summarize:

- GPU and memory-controller busy time, VRAM allocation, power and power limit.
- Temperature, SM/memory clocks, PCIe link width/generation and clock event reasons.
- Process-tree CPU use, resident memory and host memory/load snapshots.

GPU counters are device-wide, including desktop processes. Memory-controller utilization is not achieved memory bandwidth. Sampled CPU process-tree totals may miss short-lived children. The collector does not change clock or power settings. Use Nsight Systems separately for launch gaps, copies and synchronization; use Nsight Compute on representative kernels for DRAM GB/s, achieved occupancy, Tensor Core activity, register pressure and spills. Profiler overhead must not be included in throughput rankings.
