//! Fixed device activations and captured, model-independent tensor operations.
pub(crate) mod arena;
mod batch;
pub use batch::BatchOutput;
mod attention;
mod attention_decode;
pub(crate) mod attention_oxide;
pub(crate) mod attention_prefill;
mod batch_projection;
mod capture;
mod capture_attention;
mod capture_checkpoint;
mod capture_rope;
mod capture_state;
mod fp8_cache;
pub(crate) mod fp8_gemm;
pub(crate) mod kernels;
mod linear_batch;
mod metadata;
pub(crate) mod nvfp4_gemm;
#[cfg(test)]
pub(crate) mod prefill_demand;
mod prefill_gemm;
mod prefill_projection;
mod profile;
mod program;
mod recurrent;
mod recurrent_checkpoint;
pub(crate) mod recurrent_prefill;
pub(crate) mod slot_batch;
mod slot_verify;
mod small_gemm;
mod validation;
pub use arena::ActivationArena;
pub(crate) use program::ProgramGeometry;
pub use program::{DeviceProgram, FusionWeights, ProgramWeights};

mod conv_prefill;

pub mod moe;

mod reset;

pub(crate) mod greedy;

mod token_readout;

#[cfg(test)]
pub(crate) use attention_decode::taps as attention_prefill_taps;

#[cfg(test)]
pub(crate) use small_gemm::{pool_f32_capture_records, with_pool_f32, with_precise_dense};

pub(crate) mod fp8_quant_reduction;

pub(crate) mod pool_gemv;

#[cfg(test)]
pub(crate) use small_gemm::kernels as small_gemm_kernels;

#[cfg(test)]
#[path = "../../tests/unit/resident_gated_norm_discovery.rs"]
pub(crate) mod gated_norm_discovery;
