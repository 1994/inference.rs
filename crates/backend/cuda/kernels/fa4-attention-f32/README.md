# F32 KV staging for the existing FA4 AOT cores

This independent source-storage profile gathers the existing paged F32 KV cache and
rounds Q/K/V to BF16 on the GPU. F32 cache data has no FP8 descale. The 13-argument
preparation ABI, GPU sequence lengths and original BF16-to-F32 readout are shared
with the existing FA4 AOT path. The six Hopper, Blackwell and SM120 cores are unchanged.

The measured workload is MTP draft prefill with 128 rows, 24 query heads,
4 KV heads and dimension 256. The bridge supports matching full-causal F32 KV
geometry; automatic integration requires the existing MTP fusion program role.
A target program cannot allocate or capture this BF16 workspace. It introduces no
user switch or additional context ceiling.

The tuning record has its own numerical/storage profile and binds the F32 bridge,
host launch code, F64 oracle, capture code, tuner, original F32 append/decode compiler
identities, hardware and driver. It requires complete short-output checks, independent
long-context rows, relative L2 checks, unchanged cache bits and cold/warm A/B and B/A
cost confirmation. Searches run at model loading; request capture only reads choices.

The RTX 5090 staging implementation passed 20 frozen graph/canary/conversion
cases at 32K, 64K and 262K capacity, with full valid-prefix BF16 conversion bits
checked independently on the CPU. The source-bound Rust tuning check covers every
short output, 72 independent long-context F64 rows, raw append-cache bits, and both
cold/warm timing orders. Its maximum relative L2 error was 0.260% on the tested
fixtures. Those checks establish the declared BF16 operator profile, not F32 equality.

Full task-output preservation and formal service performance remain separate,
pending acceptance gates. Hopper and datacenter Blackwell hardware were unavailable;
their six instruction-family artifacts were cross-compiled and byte-reproduced.
No H/B runtime speedup is claimed.
