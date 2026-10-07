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

- [ ] Promote the reusable serving harness and workload definitions out of the local
  `artifacts/best-20261007/qualified/compare.py` experiment into `tools/bench/` and
  `benchmarks/`. Accept model paths and executable paths as arguments.
- [ ] Record engine/package versions, executable hash, effective MTP, precision,
  cache configuration, prompt hashes, warmups, repetitions and hardware telemetry.
  Retain failed runs; never silently replace a failed reference with a weaker setup.
- [ ] Support sequential paired runs and report wall time, output throughput, TTFT
  and TPOT. Add longer-context and concurrency sweeps beyond the initial batch4 case.
- [ ] Establish a separate quality/precision comparison before adopting a faster
  numerical mode. Keep precision-matched and best-valid-configuration results distinct.

Acceptance: another developer can reproduce the initial cases without the local
virtual environments or an untracked script. The existing
[serving gate](../../tools/bench/compare-results.py) rejects incomplete runs,
misaligned workloads, absent hot-prefix reuse and performance regressions.

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
