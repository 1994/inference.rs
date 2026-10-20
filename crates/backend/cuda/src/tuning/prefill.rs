//! Default projection policies; kernel modules retain only layout and ABI constraints.

/// Vendor BF16 projection wins from 2048x8192 weights upwards in prompt measurements.
const MIN_DELEGATED_WEIGHT_ELEMENTS: usize = 8 * 1024 * 1024;
/// Narrowing tile chosen from prompt measurements (1024 elements: 5.2 us versus 8.1 us at 8192).
pub const DENSE_CAST_TILE: usize = 1024;
const NARROW_DENSE_SHAPE: (usize, usize, usize) = (64, 48, 5120);

pub const fn delegate_dense(m: usize, n: usize, k: usize) -> bool {
    m.saturating_mul(k) > 0 && n.saturating_mul(k) >= MIN_DELEGATED_WEIGHT_ELEMENTS
}

/// The fused draft projection has been qualified on this architecture.
pub fn fused_draft(architecture: &str) -> bool {
    architecture == "sm_120"
}

pub fn narrow_dense_layout(architecture: &str, shape: (usize, usize, usize)) -> bool {
    architecture == "sm_120" && shape == NARROW_DENSE_SHAPE
}
