//! Shared byte-size and element-width constants for package handling.

/// One mebibyte in bytes, for readable host-side size budgets.
pub const MIB: usize = 1024 * 1024;
/// One mebibyte as `u64`, for APIs that take byte budgets as `u64`.
pub const MIB_U64: u64 = MIB as u64;
/// Maximum accepted `config.json` metadata size.
pub const CONFIG_MAX_BYTES: u64 = MIB_U64;
/// Bytes per F32 element, where element counts are `usize`.
pub const F32_BYTES: usize = size_of::<f32>();
/// Bytes per F32 element as `u64`, for byte-offset arithmetic.
pub const F32_BYTES_U64: u64 = F32_BYTES as u64;
