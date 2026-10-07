//! Named constants for the scalar reference fixture and its bounds.
//!
//! The fixture dimensions are part of the reference contract: every expected
//! logit and hidden tensor in the correctness tests is derived from them, so
//! they are named rather than repeated inline.

/// Hidden width of every reference fixture tensor.
pub const FIXTURE_HIDDEN_SIZE: usize = 8;
/// Dense feed-forward intermediate width of the reference fixture.
pub const FIXTURE_INTERMEDIATE_SIZE: usize = 16;
/// Vocabulary size (embedding and LM-head rows) of the reference fixture.
pub const FIXTURE_VOCAB_SIZE: usize = 32;
/// Number of decoder layers in the reference fixture.
pub const FIXTURE_LAYER_COUNT: usize = 2;
/// Maximum sequence length admitted by the reference fixture position spec.
pub const FIXTURE_MAX_SEQUENCE: usize = 128;
/// `RoPE` base frequency used by the reference fixture.
pub const FIXTURE_ROPE_THETA: f64 = 10000.0;
/// RMS-norm epsilon used by the reference fixture.
pub const FIXTURE_NORM_EPSILON: f32 = 1e-5;
/// Key/value element pairs stored per attention head in the fixture state.
pub const FIXTURE_KV_ELEMENTS_PER_HEAD: usize = 2;

/// `SplitMix64` state increment (the 64-bit golden-ratio constant).
pub const SPLITMIX_GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
/// First `SplitMix64` mixing multiplier (low half of the golden-ratio constant).
pub const SPLITMIX_MIX_MULTIPLIER_A: u64 = 0xbf58_476d_1ce4_e5b9;
/// Second `SplitMix64` mixing multiplier (high half of the golden-ratio constant).
pub const SPLITMIX_MIX_MULTIPLIER_B: u64 = 0x94d0_49bb_1331_11eb;
/// First right shift of the `SplitMix64` finalizer.
pub const SPLITMIX_MIX_SHIFT_A: u32 = 30;
/// Second right shift of the `SplitMix64` finalizer.
pub const SPLITMIX_MIX_SHIFT_B: u32 = 27;
/// Final right shift of the `SplitMix64` finalizer.
pub const SPLITMIX_FINAL_MIX_SHIFT: u32 = 31;
/// Number of high `SplitMix64` bits converted into the uniform random value.
pub const SPLITMIX_OUTPUT_BITS: u32 = 24;
/// Right shift that selects the high bits used by the uniform random value.
pub const SPLITMIX_OUTPUT_SHIFT: u32 = 64 - SPLITMIX_OUTPUT_BITS;
/// Divisor that normalizes the extracted `SplitMix64` bits to `[0, 1)`.
pub const SPLITMIX_UNIFORM_BASE: u64 = 1_u64 << SPLITMIX_OUTPUT_BITS;
/// Center subtracted from the normalized fixture value before scaling.
pub const FIXTURE_UNIFORM_CENTER: f64 = 1.0;
/// Amplitude of the fixture's symmetric `[-1, 1)` random weight range.
pub const FIXTURE_WEIGHT_SCALE: f32 = 0.15;

/// Largest hidden width the reference executor accepts.
pub const REFERENCE_MAX_HIDDEN_SIZE: usize = 256;
/// Largest vocabulary size the reference executor accepts.
pub const REFERENCE_MAX_VOCAB_SIZE: usize = 65536;
/// Largest dense feed-forward intermediate width the reference executor accepts.
pub const REFERENCE_MAX_INTERMEDIATE_SIZE: usize = 1024;
/// Largest sequence length the reference executor accepts.
pub const REFERENCE_MAX_SEQUENCE_LENGTH: usize = 4096;
/// Largest number of mixers (layers) the reference executor accepts.
pub const REFERENCE_MAX_MIXER_COUNT: usize = 64;
