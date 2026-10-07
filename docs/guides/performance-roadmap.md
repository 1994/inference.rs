# Inference performance roadmap

This is the requested follow-up work plan, dated 2026-10-07. It records the
remaining engineering work and acceptance criteria, not a claim of performance
parity. Update this file as work is qualified; keep raw experiments in `artifacts/`.

## Goal and boundaries

Bring native serving latency and throughput within 10% of the best valid vLLM
and SGLang configurations on comparable workloads, without sacrificing correctness.
Passing a kernel microbenchmark does not satisfy the serving gate.

- Keep the generic kernel API independent of model names and device names.
- Model-specific inference semantics belong in model providers/recipes. Scheduling
  owns request and KV lifecycles; backend implementations own fusion and acceleration.
- Select implementations using declared precision, operation semantics, tensor
  geometry, measured device capabilities and memory budgets.
- The RTX 5090 is a test environment, not the engine's supported-device boundary.
- Study and reuse proven kernels and scheduling techniques from vLLM, SGLang,
  FlashInfer and Candle where appropriate. Preserve licenses and source provenance.
  Introducing Candle as a runtime layer is not a prerequisite for reusing kernels.

## Qualified starting point

Final native release executable SHA256:
`f6b11d49b6f5fd28b5f680e5d25f5e45dacda4bf23ffcb906d162619d224c925`.

The following are median serving times in seconds. Each case has one excluded
warmup and three measured trials. Batch4 measures completion of four requests;
other cases measure one request. Each request generates 64 non-EOS tokens.

| Model | Engine | Short | Long | Batch4 | Hot prefix |
|---|---|---:|---:|---:|---:|
| Qwen3.8-27B-NVFP4 | Native | 0.7539 | 1.2505 | 1.5909 | 0.8358 |
| Qwen3.8-27B-NVFP4 | vLLM | 0.6350 | 0.5891 | 0.7198 | 0.8013 |
| Qwen3.8-27B-NVFP4 | SGLang | 0.8297 | 0.9671 | 1.1034 | 0.8599 |
| qwen3vl-2b | Native | 0.2378 | 0.3186 | 0.4502 | 0.2865 |
| qwen3vl-2b | vLLM | 0.2231 | 0.2286 | 0.2765 | 0.2458 |
| qwen3vl-2b | SGLang | 0.2043 | 0.2101 | 0.2563 | 0.2181 |

All four model/reference comparisons fail the overall 10% gate. The largest
27B gaps against vLLM are long inputs (2.12x) and batch4 (2.21x). The 2B batch4
case is 1.63x vLLM and 1.76x SGLang.

Workload alignment and limitations:

- Models were loaded from `/home/r/models`. Engines receive identical input token
  arrays and EOS IDs, with greedy sampling. 27B uses MTP depth 2; 2B uses no MTP.
- Prefix caching is enabled, and hot-prefix cache hits are checked in metrics.
  27B hot requests reuse 3584 tokens in native/SGLang but 1600 in vLLM.
- Native uses Cargo release and CUDA JIT O3. Profilers are disabled during formal
  serving measurements. GPU workloads run sequentially, with hardware telemetry.
- Native GPU memory utilization is 0.88; vLLM uses 0.85 after 0.88 failed startup
  with desktop VRAM usage. SGLang 27B uses NEXTN, ReplaySSM, FP8 KV, four running
  requests, a 12-slot lazy SSM cache and an explicit 8192-token KV cap.
- Native retains F32 intermediates and F32 KV for 2B; reference engines use BF16
  execution paths. SGLang reports default FP8 KV scales of 1.0 for 27B. Cross-engine
  precision and model-quality equivalence have **not** been certified.
- Three trials do not establish production tail latency. A same-session A/B test
  reproduced timing changes in an unchanged binary; do not attribute every change
  between distant runs to the patch being tested.

Local evidence is intentionally ignored by Git. These paths identify existing
artifacts on the development machine; the table above survives a fresh clone:

| Evidence | Local path |
|---|---|
| Final results, hashes and gates | `artifacts/best-20261007/qualified/qualification.json` |
| Native commands, request traces and telemetry | `artifacts/best-20261007/qualified/` |
| Refreshed vLLM 27B results | `artifacts/best-20261007/final-reference/` |
| Other refreshed reference results | `artifacts/best-20261007/reference-refresh/` |
| SGLang 27B results | `artifacts/best-20261007/sglang/` |
| HTML report and reference source snapshots | `artifacts/engine-reference-20261007/` |
| Native boundary-only profile | `artifacts/profile-20261007/admission-fixed/` |
| Verified vLLM CUDA trace | `artifacts/profile-20261007/vllm-kernels-verified/` |
| Final device regressions and checkpoint comparison | `artifacts/profile-20261007/final-validation/` |

## Retained implementation

Parallel causal convolution processes prefill rows together and commits history
once. Batched draft execution shares projections, and state-only draft catch-up
omits the vocabulary head and its readback. Accepted proposals refresh draft KV
using target hidden states, including after full acceptance.

Shared verification releases redundant private graphs/checkpoints and credits
admission memory. Capacity failures from asynchronous reservations wait for
resource or execution progress instead of failing the request immediately. Release
notifications wake waiters even when the backend has no byte counters. Cancellation,
timeout, rollback, inactive slots and subsequent decoding have regression coverage.

Final checks passed: 54 runtime tests, 27 CUDA host tests, 15 resident GPU tests,
a real 511-token checkpoint comparison including verification rollback and continued
decode, five serving-gate tests, CUDA/CLI Clippy, layout and repository policy checks.
The final native runs preserve all 28 request outputs per model, including warmups,
against their preceding native baseline. This does not imply cross-engine token parity.

## Work sequence

### P0 — Make comparisons reproducible from a clean checkout

- [x] Promote the reusable serving harness and workload definitions out of the local
  `artifacts/best-20261007/qualified/compare.py` experiment into `tools/bench/` and
  `benchmarks/`. Accept model paths and executable paths as arguments. The promoted
  harness is [serve-compare.py](../../tools/bench/serve-compare.py) with its canonical
  matrix in [serve-workloads.py](../../tools/bench/serve-workloads.py); `serve-cases.py`
  filters a prepared file for focused runs.
- [x] Record engine/package versions, executable hash, effective MTP, precision,
  cache configuration, prompt hashes, warmups, repetitions and hardware telemetry.
  Retain failed runs; never silently replace a failed reference with a weaker setup.
  The harness writes `binary_sha256`, `inputs_sha256`, engine version and command, and
  `/native/v1/runtime` now reports `execution_profile` (`prefill_width`, `batch_width`,
  `mtp_depth`, `automatic_prefill`, `arena_budget_bytes`) plus the effective
  `scheduler` chunk budgets, so a run can prove which graph geometry really ran.
- [ ] Support sequential paired runs and report wall time, output throughput, TTFT
  and TPOT. Add longer-context and concurrency sweeps beyond the initial batch4 case.
- [ ] Establish a separate quality/precision comparison before adopting a faster
  numerical mode. Keep precision-matched and best-valid-configuration results distinct.

Acceptance: another developer can reproduce the initial cases without the local
virtual environments or an untracked script. The existing
[serving gate](../../tools/bench/compare-results.py) rejects incomplete runs,
misaligned workloads, absent hot-prefix reuse and performance regressions.

### P1a — Qualified state after the first pass (2026-10-07)

A same-session paired A/B of the preserved baseline binary
(`artifacts/infer-perf-baseline`, SHA256 `f6b11d49…`) against the current build over
the canonical matrix, both at `--gpu-memory-utilization 0.88`, one excluded warmup and
three measured trials, gives:

| Model | Case | vLLM wall | Native wall | wall ratio | TTFT ratio |
|---|---|---:|---:|---:|---:|
| 27B | short | 0.5890 | 0.7523 | 1.277 | 1.301 |
| 27B | long | 0.6216 | 1.2524 | 2.015 | 6.628 |
| 27B | batch4 | 1.2397 | 1.7255 | 1.392 | 2.783 |
| 27B | hot_long | 0.7770 | 0.8316 | 1.070 | 0.399 |
| 2B | short | 0.2268 | 0.2390 | 1.054 | 0.853 |
| 2B | long | 0.2303 | 0.2894 | 1.257 | 3.409 |
| 2B | batch4 | 0.2783 | 0.4554 | 1.636 | 2.408 |
| 2B | hot_long | 0.2505 | 0.2979 | 1.189 | 1.775 |

Two load-time defects were fixed and are reflected above:

1. Automatic prompt-graph width divided the *remaining* device memory by the expected
   resident state count. A single request owns one arena, so that conflated two budgets
   and rejected the 128-lane graph. `crates/backend/cuda/src/loading/mod.rs` now bounds
   the width by the profiled arena budget only. Measured effect: `qwen3vl-2b` selects
   128 lanes and its 507-token TTFT falls from a 0.0835 s median to 0.0542 s (1.54x)
   with no 27B regression.
2. The CUDA prefill chunk stayed at the generic 64-token runtime default even when the
   captured graph was wider, and a chunk wider than the graph adds replays without
   helping. `crates/service/cli/src/support/serving.rs` now sets the logical chunk from
   the backend's reported `prefill_width`. The chunk size itself is **not** a lever:
   a 64/128/192/256 sweep on `qwen3vl-2b` moved the 507-token TTFT by under 3%.

The 27B is unchanged by both fixes because its prompt graph is already width 64 and the
profiled arena budget cannot fund 128 lanes at 0.88 utilization. Its dominant gap is
per-chunk prompt throughput:

- A 64-token prompt replay measures ~47.7 ms, of which linear projections are 26.1 ms
  (496 nodes, ~52.6 us each), the GDN `Delta` op is 15.7 ms (48 nodes, ~327 us each,
  48 CTAs per replay), and all elementwise/attention traffic is the remainder.
- 19.1 GB of weights are read exactly once per replay, so the memory-bandwidth floor is
  10.7 ms. Measured effective bandwidth is 0.69–0.71 TB/s (38–40% of the 1.79 TB/s
  device peak) and compute is ~1.6% of the dense BF16 peak: the gap is grid parallelism,
  not redundant weight traffic. The same kernels at two row blocks (32-lane prefill)
  reach 1.17 TB/s, and the slot-verify graph at 12 lanes is 27.7 ms against the same
  10.7 ms floor.
- The 27B `long` TTFT is therefore ~9.4 us per prompt token against vLLM's ~1.3 us.

A fresh 12-lane `slot_verify` profile (99 replays, batch4) reproduces the same shape
and attributes the 27.28 ms median replay:

| Bucket | ms / replay | Share | Notes |
|---|---:|---:|---|
| linear | 17.55 | 64% | 497 nodes; 19.1 GB of weights read once, 1.09 TB/s |
| delta | 3.45 | 13% | 48 nodes, 12 per-lane launches each |
| conv | 2.25 | 8% | 48 nodes, per-lane |
| norm | 0.84 | 3% | 161 nodes |
| attention | 1.25 | 5% | 16 nodes |
| add / multiply / rope / silu / gated_norm / split / sigmoid | 1.92 | 7% | per-lane elementwise |
| embedding | 0.02 | — | |

The linear bucket carries the replay and runs at 1.09 TB/s against a 10.7 ms
weight-only floor at the device's 1.79 TB/s. Forward projections use a `[16, 64]`
output tile, so a 12-lane replay has exactly one row block and the grid is bounded by
`N / 64` (288 CTAs for `N = 18432`) with a 40-to-136-step dependent K loop per CTA;
the whole-replay bottleneck is CTA count and K-chain depth, not bytes. A 4-request
step measures a 47.3 ms median wall (p10 46.5, p90 48.8) against vLLM's ~21.7 ms,
with 2.29 accepted tokens per request per step.

Concretely, the next lever is split-K for the small-M activation-quantized
projections, reusing `gemm::nvfp4_split`/`gemm::reduce_split` and the
`PREFILL_SPLIT_K` partial-sum pattern that the prompt path already validates. It must
be scoped by measured CTA count, because the record already notes that broad NVFP4
split-K dispatch regressed other shapes. A second, independent lever is batching the
per-lane `Conv`/`Delta`/`Norm`/`Rope` verification nodes over the verify lanes, which
is 9.95 ms of per-replay launches.

#### Split-K is measured and rejected for the target graph (2026-10-08)

A split-K twin of the FP8 projection kernel was implemented and measured in isolation:
`fp8_gemm::kernels::matmul_split` plus `reduce_split`, splitting the K loop across four
CTAs. The kernel-level evidence is a **clean win**, and the ignored regression test
`fp8_small_m_split_k_preserves_values_and_measures_better` reproduces it (paired CUDA
events, cold L2, 12 rows):

| Geometry | Unsplit | Split-K | Speedup | Worst relative gap |
|---|---:|---:|---:|---:|
| 12x17408x5120 | 0.0655 ms | 0.0370 ms | **1.77x** | 0.0 |
| 12x18432x5120 | 0.0702 ms | 0.0383 ms | **1.83x** | 0.0 |

Wired into the production dispatch for the narrow-row activation-quantized path, the
same change **regressed the measured serving case**: the 27B `batch4` wall moved from a
1.55 s median (unsplit) to a 2.75 s median with split-K, a 1.8x regression, and was
reverted. So a kernel that wins its microbenchmark by 1.8x lost end to end by 1.8x.

The reason is that the target graph is not a GEMM benchmark: a K-split adds a second
kernel and a partials round trip, and the replay's non-GEMM traffic (quantization
barriers, per-lane `Conv`/`Delta`/`Norm`, snapshot copies) does not shrink. This is the
concrete form of the warning already recorded below: a kernel-level speedup does not
imply a serving improvement. Do not re-attempt split-K here without an end-to-end
serving measurement first, and do not treat the isolated result as a pending win.

The L2/partial traffic also scales with the output width, which is large for the target
graph (`N` up to 18432), unlike the prompt path where the split-K windows are small.

### P1 — Separate target kernels from graph-external work

The successful native profile measures a median 27.72 ms target graph for 12
verification rows, 1.56 ms per four-request draft graph, and 39.85 ms for an entire
four-request backend execution. These medians are not additive CPU attribution.
Native four-request steps accept about 1.24 proposal tokens per request, versus
roughly 1.3 in the reference trace: acceptance alone does not explain the gap.

- [ ] Add bounded, optional timings around proposal sampling, target readback,
  acceptance, draft repair, KV copies and host/device synchronization.
- [ ] Map expensive GEMMs to dimensions, dtype, active rows and padding. Compare
  against the corresponding reference kernels rather than whole-trace averages.
- [ ] Measure inactive-slot and short-chain overhead separately. Reference CUPTI
  timings and native event timings have different instrumentation overheads.

Acceptance: a repeatable time breakdown explains the serving gap and identifies
which proposed change can materially reduce it. Profiling stays opt-in.

### P2 — Improve recurrent verification and small-batch projections

- [ ] Introduce owned contiguous per-slot recurrent state with disjoint views,
  retaining parent allocations for every captured graph. Preserve private state,
  prefix export, checkpoint rollback, slot reuse and cancellation semantics.
- [ ] Integrate batched GDN verification only after independent serial-state tests,
  real-checkpoint tests and end-to-end gates pass. Consider convolution and
  checkpoint fusion as part of the same measured state-traffic problem.
- [ ] Benchmark mature GEMM implementations against native FP8/NVFP4 projections
  by geometry. Evaluate projection/activation/quantization fusion and measured
  dispatch choices; retain portable fallback implementations.
- [ ] Add smaller graph buckets or masked-tail strategies where profiling proves
  that executing a full pool for fewer active requests is material.

The unshipped contiguous-state GDN prototype matched serial outputs/state exactly
on tested geometries and improved the large-head microbenchmark about 1.38x. Its
source is in `artifacts/profile-20261007/contiguous-gdn-prototype/`. Integration
requires proper ownership/layout work; do not fake aliases to bypass cuTile grid checks.

Acceptance: independent numerical tests, mixed active slots, partial acceptance,
full acceptance, repeated slot reuse, prefix restore and continued decode pass.
The full-model serving gate must improve without regressing other cases.

### P3 — Reduce readbacks and draft repair traffic

- [ ] Evaluate device-side greedy selection and acceptance for compatible sampling
  parameters, retaining the general sampling path for penalties and stochastic modes.
- [ ] Preserve non-finite rejection, signed-zero ordering and lowest-index tie rules.
  Do not treat every temperature-zero request as an unmodified argmax.
- [ ] Keep target hidden rows on device when possible, and avoid repeated draft KV
  export/import when a request can retain a draft slot lease safely.

Acceptance: sampling agrees with the CPU oracle on ties, signed zero, invalid
logits and supported parameter combinations. MTP outputs, rollback and subsequent
state remain correct. Measure transfer bytes and serving latency, not just argmax time.

### P4 — Improve long prefill and memory reuse

- [ ] Evaluate shared prefill workspace and captured-graph reuse across serially
  scheduled requests, with explicit state binding and ownership.
- [ ] Measure chunk widths and batched/chunked prefill with decode traffic. Account
  for activation storage, recurrent checkpoints, KV capacity and admission pressure.
- [ ] Validate longer contexts, prefix eviction, mixed request sizes and low-memory
  admission. Do not increase memory fractions merely to hide an accounting issue.

Acceptance: lower cold-long TTFT and end-to-end latency, with bounded memory,
forward progress and no short-request or concurrent-serving regression.

## Experiments that must not be enabled without new evidence

- Persistent recurrent prefill remains disabled for activation-quantized graphs.
  Its isolated FP64 oracle passed, but checkpoint drift exceeded the gate.
- Narrow-output SIMT projections passed the isolated oracle and sped up selected
  shapes by 15–32%, but real-checkpoint verification hidden-state relative L2 reached
  0.272. The candidate was withdrawn; do not loosen the threshold to ship it.
- Wider dense tiles and broad NVFP4 split-K dispatch regressed other measured shapes.
- Contiguous GDN microbenchmark results are not production serving improvements.

## Acceptance workflow

1. Add meaningful correctness tests before changing dispatch. Include independent
   references, partial shapes, inactive lanes and relevant state transitions.
2. Measure paired release/O3 kernel performance across affected shapes. The current
   microbenchmark regression allowance is 5%; investigate rather than average away failures.
3. Run real-checkpoint prefill, verification/rollback and teacher-forced continuation.
   The current hidden/logit gate is relative L2 <= 1e-4 and max absolute error <= 0.01.
   Changing numerical policy requires a separately justified quality evaluation.
4. Run paired native serving tests with identical tokens where the numerical contract
   is unchanged, then compare aligned vLLM/SGLang runs. Require all measured wall,
   TTFT and TPOT ratios <= 1.10; preserve the failed gate if this is not achieved.
5. Check release builds, relevant tests, Clippy, formatting, layout and repository
   policy. Run one GPU workload at a time through `tools/bench/safe-run.sh`; avoid
   compiler work during formal performance measurement.

```sh
python3 -m unittest discover -s tools/bench -p test_compare_results.py
python3 tools/bench/compare-results.py baseline.json candidate.json \
  --require-identical-tokens --max-ratio 1.10
python3 tools/check/layout.py
python3 tools/check/policy.py
cargo fmt --all --check
```

The work is complete only when correctness and serving gates pass across the agreed
matrix, reference configuration differences are disclosed, and another supported
GPU/platform validates the implementation or is explicitly recorded as unqualified.
