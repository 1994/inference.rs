# Inference performance roadmap

> **Measurement discipline.** This file has repeatedly gone wrong by combining two
> numbers that came from different populations: a cost measured on one captured graph
> applied to another graph's event count; a device replay compared with a whole-execution
> wall; a draft-head figure used as a target figure. Four separate errors of this kind are
> recorded below, two of them across the target and MTP-draft prompt graphs, which are
> both reported as `graph = "prefill"` in the same profile file. Before using any number
> here, check which program, which captured graph, and which population produced it — and
> prefer a like-for-like re-measurement over a ratio built from two sources.


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

**Refreshed on the current build (2026-10-08)**, same workload and configuration, after
the phase instrumentation and with every rejected experiment reverted. Only the ratios
are shown; the raw figures are in `artifacts/perf-r13/`.

| Model | Case | wall ratio | TTFT ratio | TPOT ratio | Gate |
|---|---|---:|---:|---:|---|
| 2B | short | 1.000 | 1.056 | 0.999 | **passes all three** |
| 2B | hot_long | 1.093 | 1.865 | 1.032 | 2 of 3 |
| 2B | long | 1.141 | 3.230 | 0.986 | TPOT passes |
| 2B | batch4 | 1.459 | 2.537 | 1.356 | fails |
| 27B | hot_long | 0.952 | 0.381 | 1.186 | wall and TTFT pass |
| 27B | short | 1.146 | 1.296 | 1.134 | fails, all near |
| 27B | batch4 | 1.140 | 2.517 | 1.855 | fails |
| 27B | long | 1.763 | 5.839 | 1.271 | fails |

Decode (`TPOT`) is at parity for `qwen3vl-2b` (0.99-1.03 on three of four cases) and
1.13-1.27 for the 27B outside batch4. The two largest remaining gaps are 27B long TTFT
(5.84x) and both models' batch4 TTFT (2.52x), i.e. prefill, not decode.

#### Like-for-like: the target prompt replay vs the target verify replay

Both graphs measured in **one profile run on one process**, which is what the discipline
note above requires and what no earlier comparison had done:

| Graph | tokens | nodes | n | median | weights at 23.4 GB |
|---|---:|---:|---:|---:|---:|
| `prefill` (target prompt) | 64 | 1154 | 7 | **42.88 ms** | **546 GB/s** |
| `prefill_last` (target prompt) | 31-63 | 1155 | 10 | **40.76 ms** | 574 GB/s |
| `slot_verify` (target verify) | 3-12 lanes | 1155 | 36 | **24.71 ms** | **947 GB/s** |
| `slot_decode` | 2-4 lanes | 24 | 64 | 1.54 ms | — |
| `prefill` (draft head) | 9-32 | 23 | 36 | **1.86 ms** | — |

This is the clean comparison: **the same target weights stream at 947 GB/s in the verify
graph and 546 GB/s in the prompt graph, a 1.7x gap**, with both figures from the same
process and the same model. The verify graph does more rows of work per replay (3-12 lanes
across 3 candidate positions) and finishes faster.

> **Superseded below.** This entry's verdict was wrong. It is retained because the error is
> instructive, but read the correction before using any of it.

**The bandwidth framing was the wrong axis.** A roofline consistency check on the same
numbers settles which limit governs. At M=64 the plan's arithmetic intensity is
`2 x 27e9 FLOP / 23.4 GB = 148 FLOP/byte` against a machine balance of
`105 TFLOP/s / 1.79 TB/s = 59 FLOP/byte`. Intensity is **2.5x the balance**, so a 64-token
prompt replay is **compute-bound, not memory-bound**:

| Roofline | Value | Reading |
|---|---:|---|
| memory | 23.4 GB / 42.88 ms = 546 GB/s | 31% of 1.79 TB/s — not the limiter |
| compute | 3.46 TFLOP / 42.88 ms = 80.6 TFLOP/s | **77% of ~105 TFLOP/s dense BF16** |

The breakeven is `M = 59/2 = 29` tokens: below that the weights cost more than the
arithmetic, above it the arithmetic dominates. At 64 tokens the prompt replay is past
breakeven, so **the 546 GB/s figure is a consequence of being compute-limited, not a
cause** — a compute-bound kernel necessarily leaves bandwidth idle.

The 23.4 GB denominator was worth checking against the checkpoint rather than a byte total,
since an unused vision tower would inflate it and weaken the conclusion. Reading the
safetensors headers directly:

| Component | size |
|---|---:|
| text backbone | 20.16 GiB |
| MTP head | 0.79 GiB |
| vision tower (**4.1% of the shard, unused by these benchmarks**) | 0.86 GiB |

Using the text backbone alone as the resident volume gives `3.46 TFLOP / 20.16 GiB =
171 FLOP/byte`, against a balance of 58.7 — **2.9x the balance, so the compute-bound
conclusion holds and strengthens** when the unused tower is excluded.

**Correction, and it reverses the verdict.** That balance of 58.7 is the *BF16* machine
balance, but the weights are 4-bit (168 MLP projections) and 8-bit (233 attention and
linear-attention projections), not BF16. The balance depends on which pipeline's peak is
used:

| Assumed pipeline | Peak | Balance | vs intensity 160 | Verdict |
|---|---:|---:|---|---|
| BF16 | 105 TFLOP/s | 59 | intensity wins | compute-bound |
| FP8 | 419 TFLOP/s | 234 | balance wins | **memory-bound** |
| NVFP4 | 838 TFLOP/s | 468 | balance wins | **memory-bound** |

Because the bulk of the FLOPs run on 4-bit and 8-bit tensor cores, the quantized balances
are the right ones and the verdict is **memory-bound**. My error was taking the "77% of
dense BF16 peak" figure as evidence of compute saturation: an achieved-throughput
percentage is not a utilization figure when the peak in the denominator is the wrong
pipeline's.

So the earlier bandwidth framing was correct after all, and this entry's retraction is
itself retracted. What the numbers actually support: the prompt replay achieves
`20.16 GiB / 42.88 ms = 505 GB/s`, or **28% of the device's 1.79 TB/s**, while the same
weights in the 12-lane verify graph reach 947 GB/s (53%). A ~1.9x bandwidth gap on
identical weights remains the standing observation, and the achieved-BF16-FLOP percentage
should not be cited as evidence either way.

**Mechanism, traced in the code.** The two graphs reach the *same* kernel but with
different row counts, and that kernel tiles the row axis by 16:

| | rows (M) | row blocks | weight reads per projection |
|---|---:|---:|---:|
| target prompt (`batch.rs:506` -> `prefill_projection::record`) | 64 | **4** | **4** |
| target verify (`slot_verify.rs` -> `record_slots`) | 12 | 1 | 1 |

Both land in `nvfp4_gemm::kernels::packed`, whose output tile is `[16, 64]` and whose
weight partition is `[64, 128]` with `pid.0` selecting the row block and `pid.1` the column
tile. Every column tile is therefore loaded once per row block, so a 64-row projection
reads each weight tile four times where a 12-row projection reads it once.

The quantities do not match exactly — 4x predicted traffic against a 1.9x observed
bandwidth gap — and the discrepancy is consistent with L2 absorbing the repeat passes
(each matrix is 42-90 MiB against a 96 MiB L2), which is also why the DRAM-only accounting
showed no amplification. **This is a hypothesis fitted to the code and the observed ratio,
not a measurement**: it has not been confirmed by counting weight bytes read per
projection.

If it holds, the fix is to widen the M tile so one row block covers all 64 prompt rows,
which would remove the repeat passes; `QUANT_GEMM_TILE = [16, 64]` is the constant to
change and `nvfp4_gemm::tests` plus the serving gate are the checks. It also finally gives
a coherent account of why split-K regressed: on a memory-bound plan already re-reading
weights, splitting K adds partial traffic without reducing weight reads.

This retracts the framing this file has carried for several entries: "the prompt graph
runs at 31% of device bandwidth" is true but is not an inefficiency to attack, and every
bandwidth-motivated lever tried against it (split-K, wider tiles, L2 residency) was aimed
at the non-limiting resource. The relevant question is why a compute-bound plan reaches
only 77% of dense BF16 peak — and note that the weights are quantized (NVFP4/FP8), whose
peaks are several times higher, so 77%-of-BF16 is a much smaller fraction of what the
hardware can do on this data.

The caveat that remains: both rows assume each replay reads all 23.4 GB exactly once, which
is untested for either path.

The draft head's 1.86 ms at 23 nodes is a one-layer network, so it remains non-comparable
and is listed only to keep it out of future ratios.

#### The 22x "cliff" compared two different programs, not two chunk sizes

The 23-node / 1.94 ms and 1154-node / 42.5 ms records do not belong to one program at two
chunk widths. A model holds **two resident programs**, each with its own prompt graph:

| Program | `prefill_width` | capture | layers | nodes | replay |
|---|---:|---|---:|---:|---:|
| target | 64 | `build32` -> `capture32` | 64 | 1154 | 42.5 ms |
| draft (MTP head) | 32 | `build32` -> `capture32` | 1 | 23 | 1.94 ms |

Both programs take the same path (`batch.rs:251` routes any width >= `PREFILL_LANES`
through `build32`), so the node difference is **the layer count of the network being
captured**, not the capture routine and not the chunk size.

`prime_draft` chunks by `PREFILL_LANES = 32` (`execution.rs:752`), so the 23-node records
are **draft KV priming**, and only the 1154-node records are target prefill. The profile
reports both as `graph = "prefill"` because each program has its own `PrefillProfile`
writing to the same path.

So the "cliff" was the draft graph measured against the target graph. That is a
cross-population comparison of the same kind that produced the previous correction — the
eighth time this session that two numbers from different sources have been combined, and
the fourth time it has happened across these two graph kinds specifically.

**What actually remains**, with only real target-prefill numbers:

- A 511-token prompt is 8 target replays of 64 tokens at **42.5 ms** each = 340 ms, and the
  measured TTFT is 398 ms. That is the whole long-prompt cost, confirmed on both the device
  and host sides.
- The target replay reads 23.4 GB of resident weights. At the device's 1.79 TB/s that is
  **13.1 ms**, so 42.5 ms is **3.2x over the memory floor**.
- The draft program's 1.94 ms for 32 tokens is therefore 1.94 ms for a much smaller
  network, and says nothing about target prefill cost.

The live question is now narrow and well-posed: **why does the 64-lane target prompt replay
run at 31% of device bandwidth**, when the same weights stream at 61% in the 12-lane slot
path. Both are target weights, both are measured, and the gap is 2x.

#### Prefill attribution: the replay is 94% of TTFT (host instrumentation)

`host_phases_us` now covers prefill (`prefill_chunk` around
`DeviceProgram::prefill_batch_readout`, `prefill_step` around `step_readout`). One
511-token prompt, 16 generated tokens, quiet server:

| Phase | samples | median |
|---|---:|---:|
| `prefill_chunk` | 25 | **43.66 ms** |
| `prefill_step` | 1 | 16.13 ms |
| whole execution wall | 47 | 46.53 ms |

The phase covers **94% of the execution wall**, and the 42.5 ms device replay measured
independently for the same 64-token shape sits inside it. So `prefill_batch_readout` is
almost entirely GPU work with a synchronous readback at the end, and `8 x ~43 = 344 ms`
reproduces the measured 0.398 s steady-state TTFT.

This **reinstates the device-side conclusion** that the previous entry retracted. That
entry reasoned "16 chunks x 1.94 ms = 31 ms, so 94% of the 0.506 s forced-width run is not
replay, therefore TTFT is host-side". The error was in the premise: the 1.94 ms figure
belongs to a *different* captured graph (23 nodes) than the one a long prompt actually
runs (1154 nodes, 42.5 ms). Applying the cheap graph's cost to the expensive graph's chunk
count produced a nonsense total, and I read the resulting gap as host overhead instead of
as evidence that the cost figure did not apply.

So the corrected position, with the host phase measured rather than inferred:

- 27B long-prompt TTFT is **device-bound**: ~43 ms per 64-token prompt replay, 8 replays.
- The host adds only ~3 ms around that call, i.e. submission and readout are not the gap.
- The open question is what makes a 64-token prefill replay cost 42.5 ms while a 32-token
  one costs 1.94 ms, which is a genuine 22x and has never been explained by any hypothesis
  tested so far.

#### The 27B long-prompt prefill cliff: 32 tokens costs 1.9 ms, 64 tokens costs 42.5 ms

This is the missing long-prompt measurement. One 511-token prompt, profiled end to end on
a quiet server with graph-only replay timings (111 records):

| Captured graph | tokens | nodes | median |
|---|---:|---:|---:|
| `prefill` | 32 | 23 | **1.9 ms** |
| `prefill` | 64 | **1154** | **42.5 ms** |
| `prefill_last` | 63 | 1155 | 44.5 ms |

**Doubling the chunk from 32 to 64 tokens multiplies the captured node count by 50 and the
replay time by 22.** The prompt's 511 tokens are processed as 8 chunks at positions
0/64/128/192/256/320/384/448, and **every one uses the 64-token shape**, so the device
replay is `8 x ~42.5 = 340 ms` against a measured steady-state TTFT of 0.399-0.413 s. That
closes the budget: the replay *is* the TTFT, and the 64-token shape is essentially all of
it.

This resolves the reconciliation gap above and supersedes the retraction that preceded it.
It also means the long-prompt gap is a **discrete dispatch cliff**, not a bandwidth,
launch-count or scheduling problem:

- The two shapes come from different ways of recording the same model.
  `batch.rs::build_pair` builds the prompt graph at `weights.prefill_width`; a 2x-large
  chunk that multiplies node count by 50 is consistent with a per-lane or per-row-group
  recording loop that the 32-token shape does not take.
- A 2x token increase cannot plausibly cost 50x the nodes in a batched graph. Something in
  the 64-lane capture records ~18 nodes per token against ~0.36 for the 23-node shape.

The node counts are **not two captures of the same graph**; they are two different graphs,
which narrows the cause to a dispatch choice:

| `prefill` chunk | captured nodes | samples | median | per token |
|---|---:|---:|---:|---:|
| 19-32 tokens | **23** | 74 | 1.94 ms | 61-102 us |
| 64 tokens | **1154** | 31 | 42.56 ms | 665 us |

74 samples at 23 nodes and 31 at 1154, with no overlap and no intermediate count, is a
branch — not a scaling curve. `batch.rs::build_pair` is the likely fork: when
`weights.prefill_width >= PREFILL_LANES` it builds the batched prompt graph
(`BatchBuilder::build`, the 1154-node shape), while a chunk that does not qualify falls to
`BatchBuilder::capture32` or to `DeviceProgram::step_readout` (the small shape). The two
are then 22x apart in *absolute* time and 6-11x apart *per token*.

Note which way the branch hurts: the **smaller** chunk is the efficient one per token
(61-102 us/token) and the larger chunk is the inefficient one (665 us/token). So this is
not "small chunks under-utilize the GPU"; the 32-token path is simply 6-11x better per
token and is not being used for the 64-token chunks.

**The prediction was tested and FALSIFIED.** `--max-num-batched-tokens 32` did force the
branch — the runtime reported `prefill_width: 32, automatic_prefill: false`, chunk 32, so
16 chunks instead of 8, all through the small shape. Long TTFT came out **worse**:

| Configuration | chunks | long TTFT (steady) |
|---|---:|---:|
| default, width 64 | 8 | **0.399-0.413 s** |
| forced, width 32 | 16 | **0.506 s** |

Worse still for the hypothesis, the arithmetic does not close either way. If the small
shape really cost 1.94 ms per replay, 16 chunks would be ~31 ms of replay and predict a
TTFT near 0.03-0.05 s; the measured 0.506 s means roughly **94% of TTFT is not replay at
all**. So the 23-node 1.94 ms figure does not describe what a long prompt actually runs,
and **neither width configuration is replay-bound**.

What this leaves is stronger than the hypothesis it replaced: doubling the chunk count and
halving the chunk width changed TTFT by only 25%, and both configurations spend the large
majority of TTFT outside the measured replays. The 27B long-prompt TTFT is **host-side**,
not device-side. The next measurement is therefore not a kernel diff but a decomposition of
the 0.40 s itself — per-chunk submission, readback and scheduling gaps on the path from
`run` to first token. The host-phase instrumentation added for the decode path
(`host_phases_us`) covers `slot_speculate` only and does not yet instrument prefill.

Also measured, and worth keeping: the **first** long prompt after a short warmup costs
1.12 s TTFT while later ones cost 0.40 s, and re-running the first prompt costs 0.20 s
(prefix reuse). So there is a further one-time ~0.7 s warm cost on the first long prompt
that the serving gate's single warmup absorbs.

#### An unreconciled 8x: per-replay prefill costs do not sum to the measured TTFT

Two independent runs agree on the shape — the last chunk of a prompt graph costs ~21x a
mid chunk — but neither reconciles with the end-to-end number:

| Measurement | mid chunk | last chunk |
|---|---:|---:|
| direct HTTP, one request in flight | 1.98 ms | 42.84 ms |
| harness, 4 concurrent | 2.14 ms | 42.54 ms |

A 511-token prompt at width 64 is 8 chunks. If only the tail were expensive the total
would be `7 x 1.98 + 42.84 = 56.7 ms`; if every chunk cost the tail price it would be
`8 x 42.5 = 340 ms`. The **measured long TTFT is 398-427 ms**, which matches neither.

So one of these is wrong: either most chunks in a real long-prompt run cost near the tail
price (~50 ms each), or a large part of the TTFT is outside the profiled replays. The
profile runs above were short-prompt dominated and never captured a single long prompt
end to end, so they cannot settle it.

**Resolved: the profile never contained a long prompt.** Inspecting the captured
positions settles it. Every `prefill` record in the batch4 profile sits in the 0-127
position band (33 samples, median 2.14 ms), and every `prefill_last` record carries 30-63
tokens across positions 40-52 — i.e. the 52-token batch4 prompts. There is **no mid-chunk
sample above position 127 at all**, which is where a 511-token prompt's 8 chunks would
live.

So the 42.5 ms `prefill_last` figure that this file has been using as "the last chunk of a
long prompt" is in fact **the last chunk of a ~52-token prompt**. It says nothing about
long-context prefill. That is the specific gap in the prefill evidence: no run to date has
profiled one long prompt from submission to first token.

Consequences, which supersede several earlier entries:

- The claim that "the 27B prefill cost is concentrated in the last chunk" is unsupported:
  it was inferred from short-prompt tails.
- The claim that the prefill replay is dominated by captured node count came from a
  correlation between 1.98 ms/23-node and 43 ms/1154-node graphs — but the 1154-node
  graphs are the short-prompt width-32 captures, not long-prompt chunks.
- The 398-427 ms long TTFT has never been decomposed by replay. Its dominant term is
  **unknown**.

What is established is narrower and still useful: a 27B prefill replay costs 1.98-2.14 ms
when the captured graph has 23 nodes and ~43 ms when it has 1154, and the 64-token
width-32 capture is the expensive shape. Whether a long prompt uses the expensive shape
for all eight chunks is exactly what the missing measurement would show.

#### The 27B prefill cost is concentrated in the last chunk

Graph-only measurement with one request in flight at a time (no batching), which settles
the 1.98 ms versus 42.5 ms discrepancy that earlier tables carried without explanation:

| Graph | samples | median |
|---|---:|---:|
| `prefill` (mid chunks) | 25 | **1.98 ms** |
| `prefill_last` (final chunk, carries the logits readout) | 2 | **42.84 ms** |

The last chunk is **21x** the mid chunk. For a 511-token prompt at width 64 that is 7
mid chunks plus the last one: `7 x 1.98 + 42.84 = 56.7 ms` of device replay. The measured
long TTFT is 398-427 ms in the gated runs (and 1131 ms for the first of two back-to-back
requests, which includes one-time state work).

Grouping those replays by captured node count relocates the cause entirely:

| Graph | captured nodes | samples | median | per node |
|---|---:|---:|---:|---:|
| `prefill` | 23 | 18 | 1.96 ms | 85.2 us |
| `prefill` | 1154 | 7 | 43.32 ms | 37.5 us |
| `prefill_last` | 1155 | 2 | 42.84 ms | 37.1 us |

**The cost tracks captured node count, not which chunk is running.** The 23-node graph and
the 1154-node graph differ by 50x in nodes and 22x in time; the 1155-node `prefill_last`
is within 1% of the 1154-node `prefill`, so the logits node is *not* the cost.

The 36-37 us per node for the large graphs is **average GPU time per node, not per-node
overhead** — a graph-only replay records one boundary segment covering all 1154 nodes, so
it cannot attribute time to individual nodes. What the 42.2 ms replay buys, measured both
ways:

| Roofline | Value | Utilization |
|---|---:|---:|
| compute, 64 tokens x 2 x 27 GFLOP = 3.46 TFLOP | 81.9 TFLOP/s | 78% of ~105 TFLOP/s dense BF16 peak |
| memory, 23.4 GB of resident weights read once | **555 GB/s** | **31% of 1.79 TB/s** |

The plan is memory-bound: 23.4 GB at 555 GB/s *is* the 42.2 ms, so the replay is limited by
weight-streaming efficiency, not by arithmetic. Per-node launch overhead cannot be read
out of these numbers and should not be inferred from them.

The same weights stream at a different rate in the other two graphs, which is the useful
comparison:

| Graph | width | achieved weight bandwidth | utilization |
|---|---:|---:|---:|
| prompt graph (this section) | 64 | 555 GB/s | 31% |
| 12-lane target verification | 12 | ~1.09 TB/s | 61% |

A wider M is *slower* per byte, which is the opposite of the usual tradeoff. The two
numbers are not perfectly like-for-like — 555 GB/s is 23.4 GB over the whole 42.2 ms
replay (1154 nodes, so it includes the non-linear nodes too), while 1.09 TB/s is the
linear bucket of the 12-lane replay — so treat the factor as approximate and the ordering
as the reliable part: the prompt graph streams weights at roughly half the rate.

**The obvious mechanical candidate does not survive checking, so this is recorded as
unresolved rather than explained.** The batched prompt path partitions by
`QUANT_GEMM_TILE = [16, 64]`, so at 64 tokens it uses **4 row blocks** against **1** for
the 12-lane slot path, which would amplify weight reads 4x. That hypothesis fails its own
arithmetic:

- every individual weight matrix fits in the device's 96 MiB L2 (gate/up 42.5 MiB, down
  85.0 MiB, QKV 90.0 MiB), so repeat passes would be served by L2, not DRAM — DRAM traffic
  would stay at 1x either way;
- 4 x 23.4 GB over 42.2 ms is 2.2 TB/s of L2 traffic, comfortably inside L2 bandwidth;
- the 23.4 GB DRAM figure is already consistent with both the 42.2 ms prompt replay
  (555 GB/s) and the 26.5 ms verify replay (883 GB/s) with no amplification at all.

So row-block amplification neither explains the 1.6x gap between those two rates nor is
ruled out cleanly by them. It is an open discrepancy, and the two rates are not
like-for-like in any case: one is whole-replay bytes over whole-replay time including all
non-linear nodes, the other is a linear-only bucket.

What would settle it is a **per-projection weight-byte count at matched geometry** for the
two paths, i.e. instrumenting the bytes each projection actually reads rather than
inferring them from tile constants. Until that exists, treat the prompt graph's streaming
rate as a measured number without a confirmed cause.

The 23-node replays are the small-token chunks (19, 31, 32 tokens at widths where the
batched nodes collapse). The 1154/1155-node replays are the 64-token width-32 captures,
where 64 layers contribute roughly 18 nodes each.

So the 27B prefill replay is **dominated by how many nodes the captured graph launches,
at ~37 us per node**, not by the vocabulary head, not by the last chunk, and not by any
single kernel. 1154 nodes x 37 us is the 43 ms.

This supersedes two earlier claims in this file: that the last chunk's logits readout is
the cost, and (from an instrumented run whose per-op shares were scaled onto the wrong
total) that per-lane `delta` launches dominate. The `delta` launch count is real but is
one contributor inside the 37 us per node, not the explanation.

The remaining device-to-TTFT gap is still host-side: 56.7 ms of replay against 398 ms of
measured TTFT means roughly 340 ms is not in the profiled replays at all.

#### batch4 TTFT is admission sequencing, not prefill throughput

Profiling the 2B batch4 case with `host_phases_us` and the execution record shows the
whole 47 ms TTFT is three executions, and the steady standalone prefill is only 8.4 ms:

| Step | Work in the execution | Wall |
|---|---|---:|
| 1 | 48-token prefill alone | 8.4 ms |
| 2 | 3 prefills + 1 decode | 27.1 ms |
| 3 | 3 decodes + 1 prefill (17 of the 48 tokens) | 14.4 ms |
| | **total to first token** | **49.9 ms** |

The measured median TTFT is 47 ms, so this accounts for it. Against a 4-lane
decode-only step of **5.0 ms**, step 2 costs **5.4x a decode-only step while carrying
exactly one decode** — a stage that mixes prefill with decode is far more expensive than
either alone. The requests are also admitted one at a time: each takes about three
executions to finish its 48-token prompt, and the per-slot TTFTs measured directly are
0.020/0.047/0.047/0.061 s, i.e. a 3x spread across four requests that arrive together.

This is why batch4 TTFT sits at ~2.5x while single-request short TTFT is at parity
(1.06): the single-request path never mixes stages. It also explains the bimodality seen
across 17 historical 2B batch4 measurements of functionally identical binaries
(0.042-0.047 s versus 0.050-0.053 s) — that is queue variation in a 50 ms sequence, not
measurement noise or a code effect.

The one-time cost of building the decode pool is visible too: the first 4-lane
decode-carrying execution is **287.6 ms**. It is excluded by the warmup, but it means the
first concurrent request after startup pays it.

So batch4 is not a kernel problem. It needs either prefill/decode stage separation or a
prompt graph wide enough to finish a request's prompt in one execution.

**The prefill replay cost is fixed, not token-dependent.** Raising
`DEFAULT_FAIR_QUANTUM_TOKENS` from 8 to 128 — the cap that limits how many prompt tokens
one request takes per scheduling turn — changed nothing measurable on `qwen3vl-2b`:

| Case | fq=8 | fq=128 |
|---|---:|---:|
| short TTFT / wall | 0.0151 / 0.227 | 0.0143 / 0.220 |
| long TTFT / wall | 0.0513 / 0.263 | 0.0508 / 0.261 |
| batch4 TTFT / wall | 0.0495 / 0.406 | **0.0490 / 0.412** |
| hot TTFT / wall | 0.0335 / 0.274 | 0.0335 / 0.274 |

Two conclusions. First, the quantum was never the binding constraint: a 48-token prompt
was already completing in 2-3 executions, i.e. ~16-24 tokens per turn rather than 8.
Second, and more useful, a prefill execution costs the same whether it carries 8, 17 or
48 tokens — consistent with step 3 of the table above costing 14.4 ms for 17 tokens while
step 1 costs 8.4 ms for 48. That is the signature of a **fixed per-replay cost**
(weight traffic plus graph node launches) rather than a throughput limit, and it is why
no scheduling-threshold change moves batch4 TTFT.

The lever is therefore the number of prefill *executions* per request, not the tokens in
them: one execution per request would give 47 ms -> about 8.4 ms of TTFT.

#### The 27B prefill replay is dominated by per-lane recurrence launches

The 27B prompt graph runs at width 64, so it processes a 511-token prompt in 8 replays.
Graph-only measurement gives a **42.5 ms** replay (`prefill_last`, 12 samples) and
`8 x 42.5 = 340 ms`, which accounts for the measured 398 ms TTFT. Scaling the
instrumented per-op shares to that true total puts **`delta` at ~14.0 ms, 33% of the
replay**, second only to the linear bucket.

`delta` is a per-lane launch, not a chunked one:
`recurrent_prefill::Workspace::new` returns an empty workspace when
`!weights.input_scales.is_empty() || !weights.fp8_inputs.is_empty()`, and the 27B
checkpoint declares activation scales. `Workspace::record` then declines, and
`batch.rs::dispatch32` sends `TensorOp::Delta` down `Dispatch32::Row`, recording one
`recurrent::delta` per lane. At width 64 that is **64 launches per node x 48 nodes =
3072 launches per replay**, and 14.0 ms of replay time at ~4.5 us per launch.

The chunked implementation that replaces this already exists in
`resident/recurrent_prefill/` and reduces the 64 per-lane launches to a handful of
chunk kernels — a ~64x launch reduction on a third of the replay. It is **gated off by a
documented numerical rejection**, not by a missing implementation:

> Persistent recurrent prefill remains disabled for activation-quantized graphs. Its
> isolated FP64 oracle passed, but checkpoint drift exceeded the gate.

**Measured, then rejected: opening the gate is not worth it.** Bypassing the condition
and running the same 27B matrix on a clean GPU:

| 27B case | gate closed | gate open | delta |
|---|---:|---:|---:|
| long TTFT | 0.3980 s | 0.3253 s | **-18%** |
| long wall | 1.096 s | 0.995 s | -9% |
| short TTFT | 0.0568 s | 0.0487 s | +17% |
| short wall | 0.675 s | 0.736 s | +9% |

So the launch-count model over-predicted: replacing 3072 per-lane launches recovered
18% of long TTFT, not the ~33% of replay time that the delta share implied. The chunked
path costs more per launch at width 1 (the short case regresses), and the win is
confined to the one case with a long prompt. Against a **documented numerical rejection**
for exactly this configuration, an 18% gain on one case plus a 17% regression on another
is not evidence to overturn it. The gate stays closed and `workspace.rs` is unchanged.

If this is ever revisited, the evidence bar is the roadmap's own acceptance criteria
(relative L2 <= 1e-4 and max absolute error <= 0.01 on the real checkpoint, plus partial
acceptance, full acceptance, rollback and continued decode), and the
`recurrent_prefill/model_check.rs` harness that drives `LEGACY_CAPTURE` against the
chunked path already exists to produce that evidence.

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
| linear | 17.55 | 64% | 497 nodes; per-node events inflate this ~12%, see the graph-only table below |
| delta | 3.45 | 13% | 48 nodes, 12 per-lane launches each |
| conv | 2.25 | 8% | 48 nodes, per-lane |
| norm | 0.84 | 3% | 161 nodes |
| attention | 1.25 | 5% | 16 nodes |
| add / multiply / rope / silu / gated_norm / split / sigmoid | 1.92 | 7% | per-lane elementwise |
| embedding | 0.02 | — | |

The linear bucket carries the replay. These per-node figures are inflated by the
instrumentation; the graph-only table further down supersedes them, and gives the
projections ~62% of the device's 1.79 TB/s. Forward projections use a `[16, 64]`
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

#### Widening the quantized recurrent prompt graph is also rejected (2026-10-08)

The 27B prompt graph is capped at 64 lanes by two guards in `loading/mod.rs`: the
profiled activation-arena budget, and a recurrent cap applied when the graph contains
`Delta` nodes and declares activation scales. Removing both — funding the 128-lane
graph from live free device memory instead — was measured on a clean GPU with the full
matrix, one excluded warmup and three trials:

| 27B case | width 64 (kept) | width 128 (rejected) |
|---|---:|---:|
| short TTFT | 0.0579 s | 0.0837 s |
| long TTFT | 0.4357 s | 0.4238 s |
| batch4 TTFT | 0.2473 s | 0.2908 s |
| hot-prefix TTFT | 0.0900 s | 0.1150 s |
| short / batch4 / hot wall | 0.736 / 1.505 / 0.797 s | 0.757 / 1.640 / 0.886 s |

Widening won only the long case, by 3%, and lost the other three by 8–45% in TTFT and
11–45% in batch4/hot wall. A wider prompt graph also enlarges every state's resident
arena, which is what the batch4 and hot-prefix cases pay for. The 1.6x gain that width
128 gives `qwen3vl-2b` does not transfer: that model has no recurrent graph and its
arena budget funds 128 lanes without lifting the cap. Keep both guards and re-measure
end to end before touching either.

#### Where the decode step actually goes (2026-10-08)

A fresh batch4 profile with the current binary separates device work from host work for
the 27B. Medians are per 4-lane decode-only backend execution (62 samples):

Graph-only measurement (`INFER_CUDA_PROFILE_GRAPH_ONLY=1`), which removes the per-node
event inflation that earlier rows in this file quoted:

| Component | ms/step | Share | Evidence |
|---|---:|---:|---|
| whole backend execution (host wall) | 38.7 | 100% | `INFER_CUDA_EXECUTION_PROFILE` |
| 12-lane `slot_verify` replay (device, true) | **26.5** | 69% | graph-only profile, 108 replays |
| — of which `linear` (derived) | ~17.1 | 44% | 23.4 GB resident weights, 1.12 TB/s of 1.79 |
| — of which per-lane `delta`/`conv`/`attention`/`norm`/rest | ~9.4 | 24% | one launch per lane |
| 4-lane `slot_decode` replays (device) | ~3.0 | 8% | 186 replays, 1.59 ms each |
| host `verify` phase (includes its readback) | 28.3 | 73% | `host_phases_us` |

The earlier figure of 29.8 ms for this replay was **12% inflated by the per-node event
instrumentation** and is superseded. The corrected split matters because it moves ~3.3 ms
from the per-lane bucket into the linear bucket: the projections are a larger share of
the replay than previously recorded, and the per-lane work is smaller.

Two further corrections from the same run:

- The checkpoint is **23.4 GB on disk** (two safetensors shards), not the 19.1 GB this
  file previously used as the decode weight volume. Used as the resident volume, the
  replay's linear time implies **1.12 TB/s, or 62% of the device's 1.79 TB/s** — better
  utilization than the 1.09 TB/s previously recorded, and less headroom than assumed.
- `draft_propose` measures 5.9 ms host around ~3.0 ms of device replays, so roughly half
  of it is its two sync edges rather than GPU work.

Three facts constrain what can help:

1. **One step reads every target weight exactly once.** 19.1 GB at the device's
   1.79 TB/s is a 10.7 ms floor, so a step cannot go below that however few requests
   are active, and per-step cost is nearly flat in lane count (`slot_decode` 1.55 ms at
   1 lane versus 1.56 ms at 4). Throughput therefore comes from tokens accepted per
   step, not from shrinking a step.
2. **The decode pool is capped at 4 slots** (`CB_DECODE_SLOTS`). At MTP depth 2 that is
   12 verification lanes, or 36 with the maximum supported depth. This ceiling, not
   bandwidth, bounds concurrent-service throughput: 8 slots would pay one weight read
   for twice the accepted tokens.
3. **~30% of the step never touches the device usefully.** Each step performs three
   graph replays that end in a synchronous readback: two sequential draft depths
   (`drafting::propose` loops over `step` and calls `run_external` per depth) and one
   target verify. Each readback moves full vocabulary logits — 17.9 MB per step across
   the 2×3 draft lanes and 12 verify lanes — and `Readbacks::run` blocks in
   `graph.launch().then(copies).sync_on(...)` before copying every source out of pinned
   staging with `as_slice().to_vec()`. Raw transfer is not the limit (17.9 MB is ~0.3 ms
   at PCIe Gen5 rates) and neither is the host argmax — a direct Rust measurement of the
   production `greedy` shape (12 rows x 248320 values, same `total_cmp` tie rule) is
   **1.67 ms per step, 0.56 ns/value**, about 4% of the step — so the remaining cost is
   the three serialising sync edges plus the per-source staging copies and their
   allocations.

The MTP-depth sweep confirms where the time is not:

| 27B case | MTP 0 | MTP 1 | MTP 2 |
|---|---:|---:|---:|
| short wall | 1.184 | 0.876 | **0.733** |
| short TTFT | 0.0531 | 0.0570 | 0.0584 |
| batch4 wall | 1.768 | 1.537 | **1.577** |
| batch4 TTFT | 0.2096 | 0.2376 | 0.2553 |

Speculation earns its keep on wall time, and TTFT is flat across depth — so the
prefill/TTFT gap is independent of the draft path and must be fixed in prefill.

Concrete next steps, in the order the evidence ranks them:

1. **Host attribution is instrumented, and it removes the readback hypothesis.**
   `INFER_CUDA_EXECUTION_PROFILE` now records `host_phases_us` per backend execution
   (`executor::profiling::phase`), so a step's host time is attributed by construction
   instead of inferred. For the 4-lane 27B decode step (60 samples, median 38.5 ms wall):

   | Host phase | ms/step | Share |
   |---|---:|---:|
   | `verify` (the 12-lane target replay plus its readback) | 28.4 | **74%** |
   | `draft_propose` (two sequential depths, each with its own replay and readback) | 5.8 | 15% |
   | remainder (decide, commit, snapshots, prefix) | 4.2 | 11% |

   The decisive comparison: `verify` costs **28.4 ms of host wall around a 29.8 ms device
   replay**. The host is not adding meaningful time on top of the GPU work for the
   dominant phase — it is waiting for it. So the D2H volume, the staging `to_vec()` and
   the greedy scan are all **not** what gates the step; the 12-lane target replay does.
   The earlier entries below that ranked readback work above replay work were wrong about
   the ordering, and their measurements are retained only as component costs.

2. **The remaining ~15 ms is `draft_propose` plus the small remainder, not readback
   overhead.** `draft_propose` is 5.8 ms for two depth replays whose combined device time
   is ~3.3 ms, so ~2.5 ms of it is the two sync edges; the last 4.2 ms is host work in
   decide/commit/snapshots/prefix. Both are worth attacking only after the verify replay.

Component costs along the same path, each a direct Rust measurement on the production
shape (12 verify lanes + 2x3 draft lanes, 248320-wide rows). They are real but none is
large enough to explain the step:

| Item | ms/step | Share of step |
|---|---:|---:|
| host greedy scan (`greedy`, same `total_cmp` rule) | 1.67 | 3% |
| staging `to_vec()` per source | 3.55 | 7% |

An earlier entry in this file claimed the scan was the largest cost at 90 ms; that was a
Python proxy running 50x slower than the real loop and is retracted. `Readbacks::prepare`
already reuses pinned staging, so the 3.55 ms is the unavoidable copy out of it, not
allocation churn.

The verify replay is already the floor: 23.4 GB of resident weights read once at ~62% of
the device's 1.79 TB/s. The other half of the replay is the per-lane bucket, and scaling
the instrumented per-op shares onto the true graph-only total puts it at **11.5 ms
(43% of the replay)**, at a measured efficiency far below the projections:

| Per-lane op | ms/replay (true) | Memory traffic floor | Achieved |
|---|---:|---:|---:|
| `delta` | 3.22 | 2.03 ms (3456 MiB state read+write) | 1126 GB/s |
| `conv` | 2.11 | 0.12 ms (203 MiB) | **101 GB/s** |
| `attention` | 1.15 | — | — |
| `norm`/`rope`/`add`/`multiply`/`silu`/`gated_norm`/`split`/`sigmoid` | 4.98 | well under 0.5 ms combined | — |

`delta` is at 63% of peak and is close to its traffic floor, so it is not the outlier.
**`conv` is: 203 MiB of traffic should take 0.12 ms and takes 2.11 ms, 17x over its
floor and 11x below the bandwidth `delta` reaches in the same replay.** The whole
non-linear bucket is ~11.5 ms of a 26.5 ms replay, i.e. ~30% of a decode step, against a
combined floor well under 3 ms.

The mechanism is **launch count, and it is confirmed by a controlled comparison**. The
same 48 convolution nodes cost, per launch, in three captured graphs measured by the same
profiler:

| Graph | lanes | conv ms | launches | us / launch |
|---|---:|---:|---:|---:|
| `slot_decode` | 2 | 0.323 | 48 | 6.7 |
| `prefill` | 32 | 2.311 | 48 | 48.1 |
| 12-lane verify (scaled) | 12 | 2.11 | **576** | 3.7 |

The launch count is what differs, and the geometry is exact: the hybrid mixer
convolution has `2*16*128 + 48*128 = 10240` channels, so at `CONV_KERNEL_TILE = 128`
every launch is 80 CTAs. `capture_state.rs::conv` records one `conv4` **per lane**, so the
verify graph issues `48 nodes x 12 lanes = 576` launches = **46,080 CTAs** where the
prefill graph issues 48 launches for the same layers. Per-launch cost is 3.7-6.7 us either
way, which is why the per-lane path costs 2.11 ms on 203 MiB of traffic.

The same per-lane shape applies to `delta` (48 nodes x 12), `norm` (161 x 12) and `rope`
(32 x 12), so the bucket is roughly 5400 launches per replay at a few microseconds each,
which is the 11.5 ms.

Two mechanical fixes were tried and **both failed**, which narrows the real solution:

1. **Widening the tile** to make one block own the whole 10240-channel row is rejected by
   cuTile: `make_partition_view: tile dimensions must be positive powers of two`, and
   10240 is not. (An earlier note in this file said 12288; that was arithmetic error. The
   real figure 10240 divides evenly by 128..2048, but the *tile* itself must still be a
   power of two, which it is — the blocker is that the **per-lane state and output
   tensors** are sized `[channels]` and cannot be partitioned by a tile larger than the
   row.)
2. **Parameterizing `conv4` as `conv4<const W: i32>`** compiles and the prefill twin
   `conv_prefill::forward` hardcodes `[1, 128]`, so the two paths share a tile width and
   cannot diverge without also parameterizing the batched kernel and updating its tests.

So the per-lane launch count is not removable by changing the tile; it needs the batched
kernel to own several lanes, i.e. the contiguous per-lane state layout the prototype was
built for. That is the scoped next step, and it is a layout change with a silent
cross-lane corruption failure mode, so it needs its own round with the full state
verification suite.

`batch.rs::dispatch32` sends `Conv`/`Delta`/`Norm`/`GatedNorm` down `Dispatch32::Row`,
which records one kernel per lane, and `capture::Capture` allocates per-lane state, so the
launch count and the state layout are both per-lane. Recovering this bucket needs the
contiguous per-lane state layout the prototype in
`artifacts/profile-20261007/contiguous-gdn-prototype/` was built for. Start with `conv`:
it has the largest gap, the smallest and most self-contained kernel, and no recurrence to
get wrong.

2. **Concurrency scaling is measured, and the slot ceiling is not the lever.** Two
   experiments settle this:

   | Workload | concurrency 4 | concurrency 8 | scaling |
   |---|---:|---:|---:|
   | 511-token prompts, 64 generated, wall | 3.61 s | 6.00 s | 1.65x for 2x work |
   | 52-token prompts, 64 generated, wall | 1.50 s | 4.10 s | 2.7x for 2x work |

   The long-prompt row shows only a 1.65x cost for 2x work, i.e. concurrency is worth
   roughly **1.15x per doubled batch**, nowhere near the 2x that paying one weight read
   for twice the tokens would imply. Raising `CB_DECODE_SLOTS` from 4 to 8 was therefore
   tried and **measured worse**: the same 8-way batch went from a 4.10 s median at 4 slots
   to 6.6 s at 8 slots. Keep the constant at 4.

   The reason is the same fact as the rest of this section: each request's decode is
   already near the one-weight-read floor, so extra concurrent requests mostly queue
   behind that floor instead of amortising it. An earlier entry here predicted 2x from
   eight slots; that prediction was wrong and is retracted. Throughput is bounded by
   per-request step cost, not by the number of slots the pool can hold.
3. **Batch the per-lane verify nodes** (`delta` 3.6 + `conv` 2.4 + `norm` 0.9 + `rope`
   0.5 + elementwise ≈ 10.7 ms/replay) into one launch per node across lanes, reusing the
   contiguous-state prototype in `artifacts/profile-20261007/contiguous-gdn-prototype/`.
4. **Cut the draft round-trips.** `drafting::propose` replays one graph per depth and each
   replay ends in a blocking readback, so depth 2 costs two host round-trips before the
   verify replay even starts. Removing them needs the draft token choice to stay on
   device, which is only worth doing once step 1 shows the volume actually matters.

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
