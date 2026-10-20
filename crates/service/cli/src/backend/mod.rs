#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "cuda")))]
mod cuda;
#[cfg(target_os = "macos")]
mod metal;
#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "cuda")))]
mod selected;
#[cfg(all(target_os = "linux", feature = "cuda"))]
use infer_backend_cuda::device::CudaDevice;
#[cfg(any(test, target_os = "macos", all(target_os = "linux", feature = "cuda")))]
use infer_core::{Error, Result};

#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "cuda")))]
pub use selected::{SelectedBackend, load};
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum BackendChoice {
    #[default]
    Auto,
    Metal,
    Cuda,
}
/// Explicit target attention numerical profile.
///
/// `Canonical` is the existing exact path. `Bf16` and `Fp16` are experimental target FP8-KV
/// prefill profiles that have not passed the fixed task-quality gate; neither is enabled
/// implicitly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum CudaAttentionProfile {
    #[default]
    Canonical,
    Bf16,
    Fp16,
}
impl CudaAttentionProfile {
    /// Stable startup-log name.
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Bf16 => "bf16",
            Self::Fp16 => "fp16",
        }
    }
}
#[cfg(all(target_os = "linux", feature = "cuda"))]
impl From<CudaAttentionProfile> for infer_backend_cuda::AttentionProfile {
    fn from(value: CudaAttentionProfile) -> Self {
        match value {
            CudaAttentionProfile::Canonical => Self::Canonical,
            CudaAttentionProfile::Bf16 => Self::Bf16,
            CudaAttentionProfile::Fp16 => Self::Fp16,
        }
    }
}
#[cfg(target_os = "macos")]
pub fn metal_available() -> bool {
    metal::available()
}
#[cfg(not(target_os = "macos"))]
pub const fn metal_available() -> bool {
    false
}
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub fn cuda_available() -> bool {
    CudaDevice::new(0).is_ok()
}
#[cfg(not(all(target_os = "linux", feature = "cuda")))]
pub const fn cuda_available() -> bool {
    false
}
pub fn catalog() -> serde_json::Value {
    serde_json::json!({"primary_target":"cuda","supported":["cuda","metal"],"auto_priority":["cuda","metal"],
        "cuda":{"implemented":cfg!(all(target_os="linux", feature="cuda")),"available":cuda_available(),"execution":"resident CUDA graphs, synchronous provider", "build_feature":"cuda"},
        "metal":{"implemented":cfg!(target_os="macos"),"available":metal_available()},
        "testing_backends":{"enabled":false,"supported_for_deployment":false}})
}
#[derive(Debug, Clone)]
#[cfg_attr(
    not(any(target_os = "macos", all(target_os = "linux", feature = "cuda"))),
    allow(
        dead_code,
        reason = "The portable CLI parses GPU allocation options even when no native execution backend is compiled"
    )
)]
pub struct Selection {
    pub kind: BackendChoice,
    pub num_gpu_blocks_override: Option<usize>,
    pub block_size: Option<usize>,
    pub max_num_batched_tokens: Option<usize>,
    /// Total context per sequence, prompt and output together (`--max-model-len`).
    pub max_model_len: Option<usize>,
    /// Service cap on generated tokens per request (`--max-output-tokens`).
    pub max_output_tokens: Option<usize>,
    /// Cap on sequences the scheduler runs at once (`--max-num-seqs`).
    pub max_num_seqs: Option<usize>,
    /// Name clients address this deployment by (`--served-model-name`).
    pub served_model_name: Option<String>,
    pub upload_staging_mib: Option<usize>,
    pub num_speculative_tokens: usize,
    /// Target attention numerical profile, experimental for `bf16`.
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub attention_profile: CudaAttentionProfile,
    pub gpu_memory_utilization: f64,
    /// Measure the best GEMV tile per projection at load time; false keeps the built-in tile.
    pub autotune: bool,
}
#[cfg(any(test, target_os = "macos", all(target_os = "linux", feature = "cuda")))]
fn resolve(
    choice: BackendChoice,
    cuda_available: bool,
    metal_available: bool,
) -> Result<BackendChoice> {
    if choice != BackendChoice::Auto {
        return Ok(choice);
    }
    if cuda_available {
        return Ok(BackendChoice::Cuda);
    }
    if metal_available {
        return Ok(BackendChoice::Metal);
    }
    Err(Error::unsupported(
        "no supported GPU backend available (CUDA or Metal); CPU execution is test-only",
    ))
}
#[cfg(test)]
#[path = "../../tests/unit/backend_mod.rs"]
mod tests;
