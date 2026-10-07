use super::*;
use crate::{backend, support::configured_engine};
use std::path::Path;

const fn selection() -> backend::Selection {
    backend::Selection {
        kind: backend::BackendChoice::TestCpu,
        num_gpu_blocks_override: None,
        block_size: Some(2),
        max_num_batched_tokens: None,
        upload_staging_mib: None,
        num_speculative_tokens: 0,
        gpu_memory_utilization: 0.9,
        autotune: true,
    }
}

#[test]
fn automatic_service_uses_loaded_graph_without_a_config_file() -> Result<()> {
    let package = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let engine = configured_engine(None, None, Some(&package), 0, selection())?;
    assert_eq!(
        engine.config().workspace_bytes,
        engine.program().workspace_bytes.max(1)
    );
    assert_eq!(
        engine.config().max_input_tokens,
        engine.model().max_sequence
    );
    assert_eq!(
        engine.config().num_gpu_blocks,
        RuntimeConfig::default().num_gpu_blocks
    );
    assert_eq!(engine.config().block_size, 2);
    assert_eq!(
        engine.config().max_num_batched_tokens,
        RuntimeConfig::default().max_num_batched_tokens
    );
    assert_eq!(
        engine.config().resource_timeout_us,
        engine.config().submission_timeout_us
    );
    Ok(())
}

#[test]
fn explicit_runtime_budget_is_preserved_and_invalid_budget_still_fails() -> Result<()> {
    let package = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let mut config = RuntimeConfig {
        resource_timeout_us: 12345,
        ..Default::default()
    };
    let engine = configured_engine(Some(config.clone()), None, Some(&package), 0, selection())?;
    assert_eq!(
        engine.config().resource_timeout_us,
        config.resource_timeout_us
    );
    assert_eq!(engine.config().workspace_bytes, config.workspace_bytes);
    config.workspace_bytes = 1;
    assert!(configured_engine(Some(config), None, Some(&package), 0, selection()).is_err());
    Ok(())
}
