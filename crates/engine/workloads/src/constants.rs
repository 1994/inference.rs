//! Shared element-width constants for workload planning and output handling.

/// Bytes per F32 element, where element counts are `usize`.
pub const F32_BYTES: usize = size_of::<f32>();
/// Bytes per F32 element as `u64`, for byte-budget arithmetic.
pub const F32_BYTES_U64: u64 = F32_BYTES as u64;
