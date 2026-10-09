use super::*;
use crate::{backend, support::configured_engine};
use std::path::Path;

const fn selection() -> backend::Selection {
    backend::Selection {
        kind: backend::BackendChoice::TestCpu,
        num_gpu_blocks_override: None,
        block_size: Some(2),
        max_num_batched_tokens: None,
        max_model_len: None,
        max_output_tokens: None,
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

#[test]
fn deployment_length_limits_reach_the_engine_and_are_readable() -> Result<()> {
    let package = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let limited = backend::Selection {
        max_model_len: Some(32),
        max_output_tokens: Some(8),
        ..selection()
    };
    let engine = configured_engine(None, None, Some(&package), 0, limited)?;
    let model_limit = engine.model().max_sequence;
    assert!(
        model_limit > 32,
        "the fixture must exceed the deployment cap"
    );
    assert_eq!(engine.config().max_model_len, Some(32));
    let limits = engine.length_limits();
    assert_eq!(limits.model_limit, model_limit);
    assert_eq!(limits.total, 32);
    assert_eq!(limits.input_cap, 32);
    assert_eq!(limits.output_cap, 8);
    assert_eq!(limits.sources.total, infer_runtime::LimitSource::Service);
    assert_eq!(
        limits.sources.output_cap,
        infer_runtime::LimitSource::Service
    );
    // Every entry point reads the same resolved values through inspection.
    assert_eq!(engine.inspect().lengths, limits);
    // An explicit limit the model cannot fulfil fails startup instead of shrinking silently.
    let unattainable = backend::Selection {
        max_model_len: Some(model_limit + 1),
        ..selection()
    };
    assert!(configured_engine(None, None, Some(&package), 0, unattainable).is_err());
    let zero = backend::Selection {
        max_model_len: Some(0),
        ..selection()
    };
    assert!(configured_engine(None, None, Some(&package), 0, zero).is_err());
    Ok(())
}
