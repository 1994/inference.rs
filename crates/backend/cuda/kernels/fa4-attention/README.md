# Architecture-specific FA4 AOT prefill

This candidate lowers FP8 paged KV and F32 queries to BF16, executes pinned FA4, then
converts its BF16 output to F32. It is a separate numerical profile from canonical F32
attention. Task acceptance is pending; the canonical exact and rejected tiled gates
are unchanged. Serving imports no Python.

| Hardware target | Execution | Validation |
| --- | --- | --- |
| SM90a / H100, H200 | WGMMA, TMA, warp specialization | Cross-compiled; hardware unavailable |
| SM100a / B200 | tcgen05, TMEM, TMA; single and paired CTA candidates | Cross-compiled; hardware unavailable |
| SM103a / B300 | tcgen05, TMEM, TMA; single and paired CTA candidates | Cross-compiled; hardware unavailable |
| SM120 / RTX 5090 | Warp MMA, cp.async | Operator and dynamic CUDA Graph checks passed |

Candidate selection uses numerical sanity plus cold/warm A/B and B/A cost checks.
Caches bind hardware, driver, artifacts, geometry and numerical profile. Cold searches
run while loading the model; request capture only reuses completed choices. No new
user switch is introduced.

The current specialization is 24 query heads, 4 KV heads and dimension 256, with a
full causal mask. Query width and KV capacity are dynamic ABI arguments; there is no
32K or 64K context ceiling. Ineligible geometry retains the generic path. One BF16
Q/O pair and K/V pair are shared across a program's layers, charged to admission
before allocation. GPU metadata controls valid lengths without a host readback.

Sources are pinned to the commit and file hashes in upstream-files.json.
LICENSE.upstream retains the upstream BSD license. host.cpp.in implements only
public CUDA API forwarding for the generated host ABI; it does not bundle the
proprietary CuTe runtime. Linux x86_64 artifacts use the installed CUDA 13 runtime.

Regenerate offline using a Python environment with Torch, nvidia-cutlass-dsl 4.8.0
and its CUDA 13 libraries, plus CUDA 13.4.92 and a C++ compiler:

    python regenerate.py --upstream /path/to/flash-attention \
      --sdk-site /path/to/site-packages --output /new/artifact-directory
    python verify-artifact.py

Generation verifies all pinned CuTe source files and never executes a GPU kernel.
manifest.json binds all six artifacts, their actual instruction families and common
build sources. Cross-compilation does not establish hardware performance or task quality.
