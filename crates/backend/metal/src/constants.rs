//! Named Metal buffer bindings, kernel geometry and byte-budget constants.
//!
//! Binding indices and launch geometry mirror the shader ABI declared in
//! `kernels.metal`, so they must stay in lockstep with that file.

/// Bytes per F32 element on device and in host readbacks.
pub const F32_BYTES: usize = size_of::<f32>();
/// [`F32_BYTES`] widened for byte-budget arithmetic performed in `u64`.
pub const F32_BYTES_U64: u64 = F32_BYTES as u64;
/// One MiB in bytes, for readable device-budget expressions.
pub const MIB: usize = 1024 * 1024;
/// One MiB in bytes widened for budgets stored as `u64`.
pub const MIB_U64: u64 = MIB as u64;
/// Leading buffer bindings that carry node inputs (`x`, `w`, `v`, `z`, `bias`).
pub const INPUT_BINDING_COUNT: usize = 5;
/// Buffer binding holding recurrent or attention state.
pub const STATE_BINDING: u64 = 5;
/// Buffer binding holding the node output.
pub const OUTPUT_BINDING: u64 = 6;
/// Buffer binding holding the inline `Params` struct bytes.
pub const PARAMS_BINDING: u64 = 7;
/// Buffer binding holding the KV page table.
pub const PAGE_TABLE_BINDING: u64 = 8;
/// Buffer binding holding the sequence token buffer.
pub const TOKEN_BINDING: u64 = 9;
/// SIMD execution width the tiled prefill kernel is compiled for.
pub const TILED_PREFILL_SIMD_WIDTH: u64 = 32;
/// Threads per threadgroup of the tiled prefill GEMM dispatch.
pub const TILED_PREFILL_THREADGROUP: u64 = 128;
/// Token rows one tiled prefill threadgroup computes.
pub const TILED_PREFILL_TILE_ROWS: u64 = 4;
/// Output columns one tiled prefill threadgroup computes.
pub const TILED_PREFILL_TILE_COLUMNS: u64 = 4;
/// Largest threadgroup requested for an untiled decode dispatch.
pub const MAX_DECODE_THREADGROUP: u64 = 64;
/// Nanoseconds per second, for Metal command-buffer timestamp conversion.
pub const NANOS_PER_SECOND: f64 = 1e9;
/// Nanoseconds per microsecond, for integer duration conversion.
pub const NANOS_PER_MICROSECOND: u64 = 1000;
/// Nanoseconds per microsecond as `f64`, for trace timestamp scaling.
pub const NANOS_PER_MICROSECOND_F64: f64 = 1000.0;
/// Task metadata a single in-flight ticket can hold.
pub const MAX_TICKET_TASKS: usize = 64;
/// Reusable task-output vectors kept in the completion pool.
pub const COMPLETION_POOL_SIZE: usize = 4;
/// Prefix entries the KV prefix cache retains at once.
pub const MAX_PREFIX_ENTRIES: usize = 4096;
