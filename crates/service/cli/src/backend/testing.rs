use super::{SelectedBackend, Selection};
use infer_backend_host::{HostBackend, HostConfig};
use infer_core::{Error, ModelId, Result};
use std::path::Path;

pub fn load(path: &Path, memory_bytes: u64, selection: Selection) -> Result<SelectedBackend> {
    if selection.kv_cache_blocks.is_some() {
        return Err(Error::unsupported(
            "fixed GPU KV pool blocks require a device backend",
        ));
    }
    if selection.prefill_chunk_tokens.is_some() || selection.upload_staging_mib.is_some() {
        return Err(Error::unsupported(
            "GPU prefill/upload options require a native device backend",
        ));
    }
    let mut package = infer_models::QwenPackage::open(path, ModelId::new(1)?)?;
    Ok(SelectedBackend::Host(Box::new(HostBackend::from_package(
        &mut package,
        HostConfig {
            memory_bytes,
            page_tokens: selection.page_tokens.unwrap_or(16),
            ..Default::default()
        },
    )?)))
}
