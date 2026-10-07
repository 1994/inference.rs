use super::{SelectedBackend, Selection};
use infer_backend_metal::{MetalBackend, MetalConfig};
use infer_core::{Error, ModelId, Result};
use std::path::Path;

pub fn available() -> bool {
    MetalBackend::available()
}
pub fn load(path: &Path, memory_bytes: u64, selection: &Selection) -> Result<SelectedBackend> {
    // Sizing a budget against a device profile and measuring kernels are CUDA-path concepts;
    // host-memory backends simply report the selection values as unused.
    let _ = (selection.autotune, selection.gpu_memory_utilization);
    // Without an explicit budget, Metal and the CPU test backend keep the host-memory default.
    let memory_bytes = if memory_bytes == 0 {
        crate::constants::DEFAULT_HOST_MEMORY_MIB * crate::constants::MIB_U64
    } else {
        memory_bytes
    };
    if selection.num_speculative_tokens != 0 {
        return Err(Error::unsupported(
            "MTP draft depth requires the CUDA resident backend",
        ));
    }
    let mut package = infer_models::ModelPackage::open(path, ModelId::new(1)?)?;
    Ok(SelectedBackend::Metal(Box::new(
        MetalBackend::from_package(
            &mut package,
            MetalConfig {
                memory_bytes,
                block_size: selection
                    .block_size
                    .unwrap_or(crate::constants::DEFAULT_KV_PAGE_TOKENS),
                kv_cache_blocks: selection.num_gpu_blocks_override,
                prefill_chunk_tokens: selection
                    .max_num_batched_tokens
                    .unwrap_or(crate::constants::DEFAULT_PREFILL_CHUNK_TOKENS),
                upload_staging_bytes: selection
                    .upload_staging_mib
                    .unwrap_or(crate::constants::DEFAULT_STAGING_MIB)
                    .checked_mul(crate::constants::MIB)
                    .ok_or_else(|| Error::invalid("upload staging size overflow"))?,
                ..Default::default()
            },
        )?,
    )))
}
