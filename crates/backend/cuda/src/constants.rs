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
    /// Decode slots captured into the shared continuous-batching graph. Four lanes cover the
    /// concurrency the admission budget grants on the reference device (`resident_states`
    /// floor is 4) without baking more state copies than traffic can occupy; retune only
    /// against the §4 concurrency measurement in docs/design/continuous-batching-plan.md.
    /// **Must equal `FUSED_VERIFY_LANES`**: slot decode reuses the 3-lane fused
    /// shared-weight GEMV (`linear_batch::batched` is hardcoded to three lanes plus
    /// padding), which is what makes a batched replay read weights once.
    pub const CB_DECODE_SLOTS: usize = FUSED_VERIFY_LANES;
    /// Token rows in each slot's KV cache. Sequences admitted above this stay on the
    /// per-sequence serial path; 4096 keeps the pool's upfront allocation (one state set per
    /// slot) inside the device budget next to resident sequences and the prefix cache.
    pub const CB_SLOT_TOKENS: usize = 4096;
    /// Metadata state-position sentinel masking one lane's state writes in a batched replay.
    pub const INACTIVE_LANE_STATE_POSITION: i32 = -1;
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
pub use device::*;
