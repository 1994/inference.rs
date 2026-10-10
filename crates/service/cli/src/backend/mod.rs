#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
mod cuda;
#[cfg(target_os = "macos")]
mod metal;
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
mod selected;
#[cfg(feature = "test-backends")]
mod testing;
#[cfg(any(
    test,
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
use infer_core::{Error, Result};

#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
pub use selected::{SelectedBackend, load};
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum BackendChoice {
    #[default]
    Auto,
    #[cfg(feature = "test-backends")]
    TestCpu,
    Metal,
    Cuda,
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
    infer_backend_cuda::device::CudaDevice::new(0).is_ok()
}
#[cfg(not(all(target_os = "linux", feature = "cuda")))]
pub const fn cuda_available() -> bool {
    false
}
pub fn catalog() -> serde_json::Value {
    serde_json::json!({"primary_target":"cuda","supported":["cuda","metal"],"auto_priority":["cuda","metal"],
        "cuda":{"implemented":cfg!(all(target_os="linux", feature="cuda")),"available":cuda_available(),"execution":"resident CUDA graphs, synchronous provider", "build_feature":"cuda"},
        "metal":{"implemented":cfg!(target_os="macos"),"available":metal_available()},
        "testing_backends":{"enabled":cfg!(feature="test-backends"),"supported_for_deployment":false}})
}
#[derive(Debug, Clone)]
#[cfg_attr(
    not(any(
        target_os = "macos",
        feature = "test-backends",
        all(target_os = "linux", feature = "cuda")
    )),
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
    pub gpu_memory_utilization: f64,
    /// Measure the best GEMV tile per projection at load time; false keeps the built-in tile.
    pub autotune: bool,
    /// Capture the chunked gated-delta prefill kernel instead of the exact per-token one.
    pub chunked_recurrent: bool,
}
#[cfg(any(
    test,
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
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
