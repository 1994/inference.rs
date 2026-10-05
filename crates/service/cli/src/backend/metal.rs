use super::{SelectedBackend, Selection};
use infer_backend_metal::{MetalBackend, MetalConfig};
use infer_core::{Error, ModelId, Result};
use std::path::Path;

pub fn available() -> bool {
    MetalBackend::available()
}
pub fn load(path: &Path, memory_bytes: u64, selection: Selection) -> Result<SelectedBackend> {
    let mut package = infer_models::QwenPackage::open(path, ModelId::new(1)?)?;
    Ok(SelectedBackend::Metal(Box::new(
        MetalBackend::from_package(
            &mut package,
            MetalConfig {
                memory_bytes,
                page_tokens: selection.page_tokens.unwrap_or(16),
                kv_cache_blocks: selection.kv_cache_blocks,
                prefill_chunk_tokens: selection.prefill_chunk_tokens.unwrap_or(32),
                upload_staging_bytes: selection
                    .upload_staging_mib
                    .unwrap_or(4)
                    .checked_mul(1024 * 1024)
                    .ok_or_else(|| Error::invalid("upload staging size overflow"))?,
                ..Default::default()
            },
        )?,
    )))
}
