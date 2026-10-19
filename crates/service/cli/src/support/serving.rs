//! Derive cold-start service limits from the loaded graph and backend contracts.
use infer_core::{ProgramId, Result};
use infer_ir::{ModelIr, PrecisionPlan};
use infer_kernel_api::KernelRegistry;
use infer_runtime::RuntimeConfig;
use infer_spi::BackendProvider;

/// Fallback logical CUDA prefill chunk when the backend does not report graph geometry.
const CUDA_PREFILL_CHUNK_TOKENS: usize = 64;

/// Step GPU-time budget for CUDA serving: wide enough that the scheduler packs every ready
/// decode and a prompt chunk, and leaves `max_num_batched_tokens` to bound a step. The
/// calibrated per-token cost mixes decode and prompt steps, so it prices a 2048-token chunk
/// at tens of seconds; any tighter budget trims or defers prompt work.
const CUDA_STEP_BUDGET_US: u64 = 3_600_000_000;

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
        // One logical chunk costs one captured prompt graph replay: a chunk wider than the
        // captured graph is wasted work, and a graph wider than the chunk never runs. Ask
        // the backend for the width it resolved at load and hand the scheduler exactly
        // that, instead of leaving the generic 64-token default in place.
        let width = backend
            .execution_profile()
            .map_or(CUDA_PREFILL_CHUNK_TOKENS, |profile| profile.prefill_width)
            .max(1);
        config.scheduler.prefill_chunk_tokens = config.scheduler.prefill_chunk_tokens.max(width);
        config.max_num_batched_tokens = config
            .max_num_batched_tokens
            .max(config.scheduler.prefill_chunk_tokens);
        // One decode step of a large model already exceeds the generic 10 ms step budget, and
        // the cost model prices an unseen batch shape linearly, so a second decode was always
        // deferred and batched shapes were never observed: concurrent requests ran one at a
        // time. Batched decode is bandwidth-bound, so the token budget is the real limit.
        config.gpu_budget_us = config.gpu_budget_us.max(CUDA_STEP_BUDGET_US);
    }
    // Cold graph capture can take as long as a device submission. The generic resource
    // default is intended for already-warm pools and is too short for JIT-backed loading.
    config.resource_timeout_us = config.submission_timeout_us;
    if let Some(cache) = backend.kv_cache() {
        config.num_gpu_blocks = cache.total_blocks.max(1);
        // The engine's logical pages mirror the backend's physical blocks one for one; keeping
        // a different page size would silently re-denominate every page-based limit.
        config.block_size = cache.block_size.max(1);
    }
    config.validate()
}
