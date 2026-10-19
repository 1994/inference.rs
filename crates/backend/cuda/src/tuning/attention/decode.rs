//! Decode launch geometry derived from hardware occupancy and the attention window.

/// Padded query rows required by the grouped MMA kernel.
pub const GROUPED_ROWS: usize = 16;
/// Upper bound shared with admission's scratch reservation.
pub const MAX_GROUPED_PARTS: usize = 64;
const WAVE_NUMERATOR: usize = 3;
const WAVE_DENOMINATOR: usize = 2;

/// Match the existing F32 reduction tree using hardware/window facts, never KV storage capacity.
pub fn partitions(heads: usize, window: Option<usize>, multiprocessors: usize) -> usize {
    if window.is_some_and(|tokens| tokens < crate::constants::ATTENTION_SPLIT_THRESHOLD) {
        return 1;
    }
    let limit = multiprocessors
        .div_ceil(heads)
        .clamp(1, crate::constants::ATTENTION_SPLIT_MAX_PARTS);
    1 << limit.ilog2()
}

/// Target about 1.5 waves of KV-head CTAs, keeping the scratch bound independent of hardware.
/// On 170 SMs and four KV heads this retains the measured 64-way split.
pub fn grouped_parts(heads: usize, kv_heads: usize, multiprocessors: usize) -> Option<usize> {
    if heads == 0
        || kv_heads == 0
        || !heads.is_multiple_of(kv_heads)
        || heads / kv_heads > GROUPED_ROWS
    {
        return None;
    }
    let blocks = multiprocessors
        .saturating_mul(WAVE_NUMERATOR)
        .div_ceil(WAVE_DENOMINATOR);
    let limit = blocks.div_ceil(kv_heads).clamp(1, MAX_GROUPED_PARTS);
    Some(limit.next_power_of_two())
}

#[cfg(test)]
#[path = "../../../tests/unit/resident_attention_decode_workspace.rs"]
mod tests;
