#[cfg(any(target_os = "macos", feature = "test-backends"))]
mod cuda;
#[cfg(target_os = "macos")]
mod metal;
#[cfg(any(target_os = "macos", feature = "test-backends"))]
mod selected;
#[cfg(feature = "test-backends")]
mod testing;
#[cfg(any(test, target_os = "macos", feature = "test-backends"))]
use infer_core::{Error, Result};

#[cfg(any(target_os = "macos", feature = "test-backends"))]
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
pub fn catalog() -> serde_json::Value {
    serde_json::json!({"primary_target":"cuda","supported":["cuda","metal"],"auto_priority":["cuda","metal"],
        "cuda":{"implemented":false,"available":false,"reason":"native CUDA execution deferred to RTX 5090 migration"},
        "metal":{"implemented":cfg!(target_os="macos"),"available":metal_available()},
        "testing_backends":{"enabled":cfg!(feature="test-backends"),"supported_for_deployment":false}})
}
#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    not(any(target_os = "macos", feature = "test-backends")),
    allow(
        dead_code,
        reason = "The portable CLI parses GPU allocation options even when no native execution backend is compiled"
    )
)]
pub struct Selection {
    pub kind: BackendChoice,
    pub kv_cache_blocks: Option<usize>,
    pub page_tokens: Option<usize>,
    pub prefill_chunk_tokens: Option<usize>,
    pub upload_staging_mib: Option<usize>,
}
#[cfg(any(test, target_os = "macos", feature = "test-backends"))]
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
mod tests {
    use super::*;
    #[test]
    fn automatic_selection_requires_a_gpu_and_never_selects_a_test_backend() {
        assert!(resolve(BackendChoice::Auto, false, false).is_err());
        assert_eq!(
            resolve(BackendChoice::Auto, false, true).unwrap(),
            BackendChoice::Metal
        );
        assert_eq!(
            resolve(BackendChoice::Auto, true, true).unwrap(),
            BackendChoice::Cuda
        );
        assert_eq!(
            resolve(BackendChoice::Cuda, false, true).unwrap(),
            BackendChoice::Cuda
        );
    }
}
