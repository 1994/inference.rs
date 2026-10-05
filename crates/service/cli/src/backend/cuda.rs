use super::SelectedBackend;
use infer_core::{Error, Result};

pub fn load() -> Result<SelectedBackend> {
    Err(Error::unsupported(
        "native NVIDIA CUDA executor is not installed; CUDA implementation is deferred to RTX 5090 migration",
    ))
}
