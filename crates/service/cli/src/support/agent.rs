//! Agent CLI support.
#[cfg(any(target_os = "macos", feature = "test-backends"))]
use super::selected_engine;
#[cfg(any(target_os = "macos", feature = "test-backends"))]
use crate::backend;
#[cfg(any(target_os = "macos", feature = "test-backends"))]
use infer_core::{Error, Result};
#[cfg(any(target_os = "macos", feature = "test-backends"))]
use infer_runtime::RuntimeConfig;
#[cfg(any(target_os = "macos", feature = "test-backends"))]
use std::{io, path::Path};

#[cfg(any(target_os = "macos", feature = "test-backends"))]
pub fn agent(
    package: Option<&Path>,
    memory_mib: u64,
    probe_memory_mib: u64,
    backend_choice: backend::Selection,
) -> Result<()> {
    let mut engine = selected_engine(
        RuntimeConfig::default(),
        None,
        package,
        memory_mib,
        backend_choice,
    )?;
    engine.backend_mut().enable_layer_probes(
        probe_memory_mib
            .checked_mul(1024 * 1024)
            .ok_or_else(|| Error::invalid("probe budget overflow"))?,
    )?;
    let mut service = infer_agent::AgentService::new(engine, backend::catalog())?;
    infer_agent::transport::serve(
        &mut service,
        &mut io::stdin().lock(),
        &mut io::stdout().lock(),
    )
}
