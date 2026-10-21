# Issue #34 native FP8 calibration

[issue-34-sm120.json.gz](issue-34-sm120.json.gz) is the complete validated tuning
table used for the Qwen3.8-27B-NVFP4 C16/MTP2 qualification in
[PR #35](https://github.com/1994/inference.rs/pull/35). It retains the original
baseline table and adds seven M48 native FP8 entries. Other table sections and
the original six FP8 entries are unchanged.

The archive includes all four-fixture finite/bitwise gates, cold/warm paired
trials and reverse-order confirmation. The existing selector checks the exact
hardware, driver, library and kernel identity before accepting an entry.
This calibration matches RTX 5090 / sm_120 / 170 SMs, driver 615.78.08,
CUDA runtime 13000 and cuBLASLt 130800; incompatible identities do not reuse it.

| M | N | K | Selected native tile M×N |
|---|---|---|---|
| 48 | 10240 | 5120 | 64×64 |
| 48 | 6144 | 5120 | 64×64 |
| 48 | 5120 | 6144 | 16×128 |
| 48 | 12288 | 5120 | 64×128 |
| 48 | 1024 | 5120 | 16×64 |
| 48 | 17408 | 5120 | 64×64 |
| 48 | 5120 | 17408 | 16×128 |

Decompress it into a writable benchmark cache:

~~~sh
gzip -dc benchmarks/tuning/issue-34-sm120.json.gz > /tmp/inference-issue34-tuning.json
sha256sum /tmp/inference-issue34-tuning.json
export INFER_CUDA_TUNING_TABLE=/tmp/inference-issue34-tuning.json
~~~

The decompressed SHA256 is
6ada4d52af7dd41855d454e4968e7132e30d3f9b4886d619c2bb1e271fd8d4df.
Run the existing tools/bench/serve-compare.py workload with this environment.
The measured configuration uses C16 × 4096 input tokens, 256 output tokens,
MTP2, utilization 0.88, maximum length 32768, maximum sequences 16, prefix caching,
and --extra --cuda-attention-profile bf16 --no-autotune. Its automatic native
token budget is 2063; the vLLM comparison uses 2048.

On frozen executable
6f25cfddde47862a499d9bd8290e5e41876e9234eefed7e7bc9b24a5d0e68303,
adjacent three-trial medians were 12.086 s with the baseline table and 11.910 s
with this table. All 64 complete token streams matched. These are configuration
results on the same executable; source-code gains are reported separately in
the PR. Final independent serving, long-context and task-quality gates are
reported there as they complete.
