//! Explicit numerical profiles for the target prefill attention path.
//!
//! The profile is resolved once when the model loads and is immutable for every sequence the
//! loaded model creates. It never changes weights, KV storage or the draft proposal path.
use serde::Serialize;

/// Target prefill attention arithmetic selected by the operator.
///
/// `Canonical` keeps the existing exact attention path bit-for-bit. `Bf16` opts the target's
/// FP8-KV prefill into the existing FA4 BF16-QKV / F32-accumulation / BF16-output kernel. `Fp16`
/// opts only the validated SM120 target FP8-KV prefill into the plain-FP16-QKV /
/// F32-accumulation / FP16-output kernel. Both experimental profiles are never selected
/// automatically and are not production accuracy claims until the fixed task-quality gate passes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum AttentionProfile {
    /// Existing canonical attention, unchanged.
    #[default]
    Canonical,
    /// Existing FA4 BF16 attention for target FP8-KV prefill; experimental, not quality-approved.
    Bf16,
    /// FA4 plain-FP16 attention for validated SM120 target FP8-KV prefill; not quality-approved.
    Fp16,
}
impl AttentionProfile {
    /// Stable readback/log name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Bf16 => "bf16",
            Self::Fp16 => "fp16",
        }
    }
    /// Whether the explicit BF16 target profile is selected.
    #[must_use]
    pub const fn is_bf16(self) -> bool {
        matches!(self, Self::Bf16)
    }
    /// Whether the explicit FP16 target profile is selected.
    #[must_use]
    pub const fn is_fp16(self) -> bool {
        matches!(self, Self::Fp16)
    }
    /// Whether a target prefill node may use one of the experimental FA4 profiles.
    ///
    /// The caller still decides node-level FP8-KV, architecture and geometry eligibility; this
    /// only separates the canonical default from the opt-in target profiles.
    #[must_use]
    pub const fn is_target_mma(self) -> bool {
        matches!(self, Self::Bf16 | Self::Fp16)
    }
}
