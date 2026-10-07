//! Derive cold-start service limits from the loaded graph and backend contracts.
use infer_core::{ProgramId, Result};
use infer_ir::{ModelIr, PrecisionPlan};
use infer_kernel_api::KernelRegistry;
use infer_runtime::RuntimeConfig;
use infer_spi::BackendProvider;

/// Logical CUDA prefill chunk; the backend selects smaller captured kernel batches.
const CUDA_PREFILL_CHUNK_TOKENS: usize = 256;

pub fn derive_config(
    config: &mut RuntimeConfig,
    backend: &impl BackendProvider,
    model: &ModelIr,
    registry: &KernelRegistry,
) -> Result<()> {
    let capabilities = backend.capabilities();
    let budget = backend
        .free_state_bytes()?
        .unwrap_or(capabilities.memory_bytes);
    let program = infer_compiler::compile(
        ProgramId::ONE,
        infer_compiler::lower(model, backend.execution_graph(model)?, PrecisionPlan::f32())?,
        registry,
        &capabilities,
        budget,
    )?;
    config.workspace_bytes = program.workspace_bytes.max(1);
    if capabilities.backend_kind() == infer_ir::BackendKind::Cuda {
        // The backend subdivides this logical chunk to its selected kernel width.
        config.scheduler.prefill_chunk_tokens =
            CUDA_PREFILL_CHUNK_TOKENS.min(config.max_num_batched_tokens);
    }
    config.max_input_tokens = config.max_input_tokens.min(model.max_sequence);
    // Cold graph capture can take as long as a device submission. The generic resource
    // default is intended for already-warm pools and is too short for JIT-backed loading.
    config.resource_timeout_us = config.submission_timeout_us;
    if let Some(cache) = backend.kv_cache() {
        config.num_gpu_blocks = cache.total_blocks.max(1);
    }
    config.validate()
}

#[cfg(all(test, feature = "test-backends"))]
#[path = "serving/tests.rs"]
mod tests;
