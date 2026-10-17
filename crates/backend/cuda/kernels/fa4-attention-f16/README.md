# SM120 FA4 plain-FP16 AOT artifact

This directory contains the only validated FP16 FA4 artifact: the SM120
(RTX 5090) plain-FP16 inputs, F32 accumulator, FP16 output core and its
paged-cache bridge. It is an explicit, opt-in numerical profile. Canonical
attention remains the default and unsupported architectures or geometry fall
back to canonical. The MTP draft path is unchanged.

The embedded AOT library is
`sm120/attention.so` (sha256 `1cd25cc4...c91a`). The bridge is
`bridge.ptx` (sha256 `caf06f30...9baad`) built from `bridge.cu`
(sha256 `6e129e49...74c26`). `prepare_fa4_f16` widens FP8 E4M3 K/V to F16,
applies the checkpoint F32 scale and rounds to F16; `finish_fa4_f16` widens
the F16 output to F32 and keeps the inactive-row zero contract.
`attention.so` implements the same fixed `infer_fa4_initialize` /
`infer_fa4_forward` host ABI as the BF16 artifacts.

Provenance is frozen in `provenance.json` (the probe's
`artifact-hashes.json`) and `manifest.json`: upstream
`e9d8944c855489e03c66262d07ba4c81d9d8dd31`, CuTe DSL `4.8.0`, CUDA
`13.4.92`, source-tree sha256 `06cc6745...fb0`. The compile record has
`GPU_execution=false`; the separate 20-case operator probe
(`staging/fp16-fa4-probe-20261018/logs/run-1-envbug/probe-report.md`)
ran on SM120 and is the hardware/operator evidence. It is not a model
task-quality acceptance. `task_quality_accepted` stays false until the
fixed 1,024-question gate passes.

Run `python3 verify-artifact.py` to check the file hashes, fixed provenance,
FP16 instruction family, ELF/host ABI and runtime dependencies without a GPU.
