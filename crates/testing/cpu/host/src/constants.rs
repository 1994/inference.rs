//! Named byte widths and shared limits for the host reference backend.

/// Size in bytes of one `f32` tensor element.
pub const F32_BYTES: usize = size_of::<f32>();
/// Size in bytes of one `f32` element in a `u64` byte-budget expression.
pub const F32_BYTES_U64: u64 = F32_BYTES as u64;
/// Size in bytes of one `f32` element in a `u128` byte-budget expression.
pub const F32_BYTES_U128: u128 = F32_BYTES as u128;
/// Size in bytes of one `u32` token id in a `u64` byte-budget expression.
pub const TOKEN_BYTES_U64: u64 = size_of::<u32>() as u64;
/// Size in bytes of one `u32` token id in a `u128` byte-budget expression.
pub const TOKEN_BYTES_U128: u128 = size_of::<u32>() as u128;
/// Maximum entries retained by a host prefix cache before LRU eviction.
pub const PREFIX_CACHE_MAX_ENTRIES: usize = 4096;
