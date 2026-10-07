use super::{SelectedBackend, Selection};
use infer_backend_host::{HostBackend, HostConfig};
use infer_core::{Error, ModelId, Result};
use std::path::Path;

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
    if selection.num_gpu_blocks_override.is_some() {
        return Err(Error::unsupported(
            "fixed GPU KV pool blocks require a device backend",
        ));
    }
    if selection.max_num_batched_tokens.is_some() || selection.upload_staging_mib.is_some() {
        return Err(Error::unsupported(
            "GPU prefill/upload options require a native device backend",
        ));
    }
    if selection.num_speculative_tokens != 0 {
        return Err(Error::unsupported(
            "MTP draft depth requires the CUDA resident backend",
        ));
    }
    let mut package = infer_models::ModelPackage::open(path, ModelId::new(1)?)?;
    Ok(SelectedBackend::Host(Box::new(HostBackend::from_package(
        &mut package,
        HostConfig {
            memory_bytes,
            block_size: selection
                .block_size
                .unwrap_or(crate::constants::DEFAULT_KV_PAGE_TOKENS),
            ..Default::default()
        },
    )?)))
}
