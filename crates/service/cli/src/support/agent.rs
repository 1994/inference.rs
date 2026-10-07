//! Agent CLI support.
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
use super::selected_engine;
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
use crate::backend;
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
use infer_core::{Error, Result};
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
use infer_runtime::RuntimeConfig;
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
use std::{io, path::Path};

#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
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
            .checked_mul(crate::constants::MIB_U64)
            .ok_or_else(|| Error::invalid("probe budget overflow"))?,
    )?;
    let mut service = infer_agent::AgentService::new(engine, backend::catalog())?;
    infer_agent::transport::serve(
        &mut service,
        &mut io::stdin().lock(),
        &mut io::stdout().lock(),
    )
}
