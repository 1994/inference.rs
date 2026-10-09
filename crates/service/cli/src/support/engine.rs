//! Engine CLI support.
use super::read_json;
use crate::backend;
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
use crate::backend::SelectedBackend;
#[cfg(feature = "test-backends")]
use infer_backend_reference::{ReferenceBackend, ReferenceKernels, ReferenceModel};
#[cfg(feature = "test-backends")]
use infer_core::ModelId;
use infer_core::{Error, Result};
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
use infer_ir::PrecisionPlan;
#[cfg(feature = "test-backends")]
use infer_kernel_api::KernelRegistry;
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
use infer_runtime::{Engine, RuntimeConfig};
use std::path::Path;

/// Deterministic seed for the CPU reference fixture model.
#[cfg(feature = "test-backends")]
const REFERENCE_FIXTURE_SEED: u64 = 7;

#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
pub fn selected_engine(
    config: RuntimeConfig,
    model: Option<&Path>,
    package: Option<&Path>,
    memory_mib: u64,
    choice: backend::Selection,
) -> Result<Engine<SelectedBackend>> {
    configured_engine(Some(config), model, package, memory_mib, choice)
}

pub fn configured_engine(
    config: Option<RuntimeConfig>,
    model: Option<&Path>,
    package: Option<&Path>,
    memory_mib: u64,
    choice: backend::Selection,
) -> Result<Engine<SelectedBackend>> {
    let automatic = config.is_none();
    let mut config = config.unwrap_or_default();
    config.block_size = choice.block_size.unwrap_or(config.block_size);
    if let Some(tokens) = choice.max_num_batched_tokens {
        config.max_num_batched_tokens = tokens;
    }
    if let Some(tokens) = choice.max_model_len {
        config.max_model_len = Some(tokens);
    }
    if let Some(tokens) = choice.max_output_tokens {
        config.max_output_tokens = Some(tokens);
    }
    if let Some(sequences) = choice.max_num_seqs {
        config.max_num_seqs = sequences;
    }
    if let Some(name) = choice.served_model_name.clone() {
        config.served_model_name = Some(name);
    }
    // The refined selection is the only one the backend sees, and building it here consumes the
    // caller's value on every path instead of leaving it borrowed in host-only builds.
    let choice = backend::Selection {
        block_size: Some(config.block_size),
        ..choice
    };
    if let Some(package) = package {
        let backend = config
            .cpu
            .placement
            .device
            .scope(|| backend::load(package, memory_mib, &choice))?;
        let ir = backend.model_ir().clone();
        let registry = backend.registry()?;
        if automatic {
            super::serving::derive_config(&mut config, &backend, &ir, &registry)?;
        }
        let mut engine = Engine::new(backend, ir, PrecisionPlan::f32(), &registry, config)?;
        if package.join("readouts.safetensors").exists() {
            let provider = infer_workloads::ProjectionWorkloads::open(
                package,
                engine.model().hidden_size,
                engine.model().vocab_size,
            )?;
            engine = engine.with_workloads(provider)?;
        }
        Ok(engine)
    } else {
        #[cfg(feature = "test-backends")]
        if choice.kind == backend::BackendChoice::TestCpu {
            return new_engine(config, model);
        }
        let _ = model;
        Err(Error::invalid(
            "device backend requires --package; CPU fixtures require a test-backends build and --backend test-cpu",
        ))
    }
}
#[cfg(feature = "test-backends")]
pub fn registry() -> Result<KernelRegistry> {
    let mut r = KernelRegistry::default();
    r.register(&ReferenceKernels)?;
    Ok(r)
}
#[cfg(feature = "test-backends")]
pub fn new_engine(config: RuntimeConfig, model: Option<&Path>) -> Result<Engine<SelectedBackend>> {
    let model = if let Some(path) = model {
        read_json(path)?
    } else {
        ReferenceModel::fixture(ModelId::new(1)?, REFERENCE_FIXTURE_SEED)
    };
    let ir = model.ir.clone();
    Engine::new(
        SelectedBackend::Reference(Box::new(ReferenceBackend::new(model)?)),
        ir,
        PrecisionPlan::f32(),
        &registry()?,
        config,
    )
}
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
pub fn config(path: Option<&Path>) -> Result<RuntimeConfig> {
    path.map(read_json)
        .transpose()
        .map(Option::unwrap_or_default)
}
