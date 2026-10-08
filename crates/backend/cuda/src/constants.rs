//! Named launch geometry, encoding format and admission budget constants.

/// FP4 values sharing one E4M3 block scale.
pub const NVFP4_GROUP_SIZE: usize = 16;
/// Fallback decode GEMV tile when no measurement applies.
pub const DEFAULT_TILE_ROWS: usize = 16;
/// Fallback decode GEMV tile when no measurement applies.
pub const DEFAULT_TILE_COLUMNS: usize = 256;

/// Device-only geometry and budget constants; empty without the `cuda` feature.
#[cfg(all(target_os = "linux", feature = "cuda"))]
mod device {
    /// Middle prompt capture bucket, including quantized recurrent models.
    pub const MID_PREFILL_LANES: usize = 64;
    /// Head-major KV cache axes: heads, capacity, head dimension.
    pub const KV_CACHE_RANK: usize = 3;
    /// Minimum KV span where split decoding amortizes its reduction.
    pub const ATTENTION_SPLIT_THRESHOLD: usize = 1024;
    /// Minimum tokens assigned to a split attention partition.
    pub const ATTENTION_SPLIT_MIN_TOKENS: usize = 128;
    /// Maximum split attention partitions and their scratch budget.
    pub const ATTENTION_SPLIT_MAX_PARTS: usize = 16;
    /// External hidden, embedding and two normalized rows used by draft fusion.
    pub const MTP_FUSION_ROWS: u64 = 4;
    /// MMA output tile used by dense and scaled verification projections.
    pub const SMALL_GEMM_TILE: [usize; 2] = [16, 32];
    /// Maximum row count before MMA replaces the vocabulary GEMV path.
    pub const SMALL_GEMV_MAX_ROWS: usize = 4;
    /// Vocabulary-like expansion ratio where narrow-batch GEMV remains useful.
    pub const GEMV_WIDE_OUTPUT_RATIO: usize = 4;
    /// Output tile shared by the native activation-quantized GEMMs on decode-width graphs.
    pub const QUANT_GEMM_TILE: [usize; 2] = [16, 64];
    /// Row tile for graphs wider than one decode tile.
    ///
    /// **Measured (RTX 5090)**: on the 27B long-prompt graph the 64-row tile cut the replay from
    /// 46.9 ms to 43.1 ms and the prompt `linear` bucket from 27.3 ms to 23.1 ms (serving A/B,
    /// identical token sequences). The 16-row tile splits a prompt graph into four row blocks
    /// that each re-read the same weight tile.
    ///
    /// **Also measured**: widening the tile globally to `[64, 128]` wins the isolated
    /// `nvfp4_packed_tile_sweep` kernel benchmark at every row count but loses end to end
    /// (27B short wall 1.14x -> 1.30x, batch4 TPOT 1.86x -> 2.05x): a single 40-CTA decode
    /// GEMM cannot fill the device, and a 64-row CTA on a one-row decode wastes 63 rows of
    /// tensor-core work. Keep the row-dependent split and re-measure serving, not just the
    /// kernel, before changing either dimension.
    pub const PROMPT_GEMM_TILE_ROWS: usize = 64;
    /// Output tile of the activation-quantized GEMM for a graph with `rows` output rows.
    ///
    /// Decode-width graphs keep the 16-row tile; wider ones take the prompt tile so one CTA
    /// covers the whole width and reads each weight once. Selection is a pure function of the
    /// output row count, so one shared `Workspace` can serve prompt and decode graphs without any
    /// mutable per-graph state.
    pub const fn quant_gemm_tile(rows: usize) -> [usize; 2] {
        if rows > QUANT_GEMM_TILE[0] {
            [PROMPT_GEMM_TILE_ROWS, QUANT_GEMM_TILE[1]]
        } else {
            QUANT_GEMM_TILE
        }
    }
    /// Packed activation codes written by one NVFP4 quantization block.
    pub const NVFP4_QUANT_CODES_TILE: [usize; 2] = [1, 256];
    /// Activation scales written by one NVFP4 quantization block.
    pub const NVFP4_QUANT_SCALES_TILE: [usize; 2] = [1, 32];
    /// QKV, beta and alpha activation inputs to gated delta recurrence.
    pub const GDN_PROJECTED_INPUTS: usize = 3;
    /// Fallback decode GEMV tile used by the untiled `matvec` entry point.
    pub const MATVEC_TILE_ROWS: usize = 4;
    /// Fallback decode GEMV tile used by the untiled `matvec` entry point.
    pub const MATVEC_TILE_COLUMNS: usize = 128;
    /// One MiB in bytes, for readable budget expressions.
    pub const MIB: usize = 1024 * 1024;
    /// One GiB in bytes, for device-memory budget expressions.
    pub const GIB: usize = 1024 * MIB;
    /// Bytes per F32 element on device and in pinned host readbacks.
    pub const F32_BYTES: usize = size_of::<f32>();
    /// Elementwise and fused kernels process one tile of this many elements per block.
    pub const AUX_KERNEL_TILE: usize = 256;
    /// Tokens per batched prefill graph; the GEMM M tile matches the lane count.
    pub const PREFILL_LANES: usize = 32;
    /// Largest automatically selected prompt batch, bounded by the activation budget.
    pub const MAX_PREFILL_LANES: usize = 128;
    /// Column tile of the batched prefill GEMM.
    pub const PREFILL_GEMM_TILE_N: usize = 64;
    /// K tile of the batched prefill GEMM: columns of input per loop iteration.
    pub const PREFILL_GEMM_TILE_K: usize = 64;
    /// NVFP4 prompt GEMM output tile. **Measured: 32 is 2.8x worse than 64** (196-token prompt,
    /// 32-lane chunks: 132.1 vs 46.7 ms per chunk, numerics still correct). A narrower N tile
    /// shrinks every MMA and doubles the instruction count, so the prompt GEMM is bound by
    /// tensor-core shape efficiency, not by CTA occupancy. Do not lower this again without
    /// re-measuring the §5e ladder.
    pub const PREFILL_NVFP4_TILE_N: usize = 64;
    /// K-windows a prefill fp4 GEMM splits across CTAs before the reduction pass.
    /// **Measured: 4 and 8 tie on TTFT (within noise) but 4 halves partial traffic and
    /// scratch memory, and wins the in-graph linear bucket (32.3 vs 33.2 ms/chunk).**
    pub const PREFILL_SPLIT_K: usize = 4;
    /// Candidates verified by the fused shared-weight GEMV.
    pub const FUSED_VERIFY_LANES: usize = 3;
    /// Maximum draft candidates accepted by the verification graph builder.
    pub const MAX_VERIFICATION_WIDTH: usize = 9;
    /// Maximum speculative draft depth accepted when loading an MTP model. The verify batch
    /// carries the submitted token plus every candidate, so the depth is bounded by its width.
    pub const MAX_MTP_DEPTH: usize = MAX_VERIFICATION_WIDTH - 1;
    /// Target positions the MTP draft's KV cache lags behind, matching the draft head's
    /// `kv_offset`: the draft attends to the position before the token it consumes.
    pub const MTP_KV_OFFSET: usize = 1;
    /// Per-lane metadata carries position, token, KV position and the external-hidden flag.
    pub const METADATA_FIELDS: usize = 4;
    /// Tap count of the hybrid mixer convolution kernel.
    pub const CONV_KERNEL_SIZE: usize = 4;
    /// Stateful conv taps retained per mixer; the current input is separate.
    pub const CONV_STATE_TENSORS: usize = 3;
    /// Convolution kernels process one tile of channels per block.
    pub const CONV_KERNEL_TILE: usize = 128;
    /// Sequence length bound for resident graphs and admission.
    pub const MAX_CAPACITY_TOKENS: usize = 32768;
    /// Hidden size bound for resident programs.
    pub const MAX_HIDDEN_SIZE: usize = 32768;
    /// Maximum shared decode slots. Independent of the per-sequence draft depth;
    /// the batched projection kernels handle row tails and reuse weights across slots.
    ///
    /// **Measured: raising this to 8 made the same 8-way batch slower, 4.10 s -> 6.6 s.**
    /// A step does read every target weight once (`slot_decode` costs 1.55 ms at one lane
    /// and 1.56 ms at four), but extra concurrent requests queue behind that floor rather
    /// than amortising it: going from 4 to 8 concurrent long prompts cost 1.65x for 2x the
    /// work. Each slot also keeps a full state set plus its verification checkpoints, so a
    /// wider pool spends device memory for a throughput gain that was not observed. Do not
    /// raise this without an end-to-end measurement at the concurrency being targeted.
    pub const CB_DECODE_SLOTS: usize = 4;
    /// Token rows in each slot's KV cache. Sequences admitted above this stay on the
    /// per-sequence serial path; 4096 keeps the pool's upfront allocation (one state set per
    /// slot) inside the device budget next to resident sequences and the prefix cache.
    pub const CB_SLOT_TOKENS: usize = 4096;
    /// Metadata state-position sentinel masking one lane's state writes in a batched replay.
    pub const INACTIVE_LANE_STATE_POSITION: i32 = -1;
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
pub use device::*;
