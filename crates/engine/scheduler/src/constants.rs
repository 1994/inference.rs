//! Named scales shared by scheduler validation, packing and cost calibration.

/// Denominator for percent-valued scheduler and cost configuration fields.
pub const PERCENT: u32 = 100;
/// Parts-per-million denominator for calibrated cost and service-charge ratios.
pub const PARTS_PER_MILLION: u64 = 1_000_000;
