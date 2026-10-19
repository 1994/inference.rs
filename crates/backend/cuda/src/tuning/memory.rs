//! Hardware-derived workspace budgets shared by admission and graph allocation.

const MIB: u64 = 1024 * 1024;
/// Funds the 2048-row prompt graph on a 32 GiB device, bounded on smaller/larger devices.
const ARENA_DIVISOR: u64 = 32;
const MIN_ARENA_BUDGET: u64 = 128 * MIB;
const MAX_ARENA_BUDGET: u64 = 1024 * MIB;

pub fn arena_budget(total_memory_bytes: u64) -> u64 {
    (total_memory_bytes / ARENA_DIVISOR).clamp(MIN_ARENA_BUDGET, MAX_ARENA_BUDGET)
}
