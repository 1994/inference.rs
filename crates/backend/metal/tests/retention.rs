#![cfg(target_os = "macos")]
use infer_backend_metal::{MetalBackend, MetalConfig, MetalKernels};
use infer_core::{Error, ModelId, RequestId, Result, StateId};
use infer_ir::{CanonicalRequest, OutputReadout, Pooling, PrecisionPlan, Workload, WorkloadOutput};
use infer_kernel_api::KernelRegistry;
use infer_models::ModelPackage;
use infer_runtime::{Engine, RuntimeConfig};
use infer_spi::BackendProvider;
use std::{path::Path, time::Duration, time::Instant};

fn backend(cache: u64) -> Result<MetalBackend> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let mut package = ModelPackage::open(root, ModelId::ONE)?;
    MetalBackend::from_package(
        &mut package,
        MetalConfig {
            block_size: 2,
            kv_cache_blocks: Some(16),
            prefix_cache_bytes: cache,
            ..Default::default()
        },
    )
}
fn engine(cache: u64) -> Result<Engine<MetalBackend>> {
    let b = backend(cache)?;
    let model = b.model().clone();
    let mut registry = KernelRegistry::default();
    registry.register(&MetalKernels)?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let workloads =
        infer_workloads::ProjectionWorkloads::open(root, model.hidden_size, model.vocab_size)?;
    Engine::new(
        b,
        model,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig::default(),
    )?
    .with_workloads(workloads)
}
fn request(id: u64, workload: Workload) -> Result<CanonicalRequest> {
    let mut request: CanonicalRequest = serde_json::from_value(serde_json::json!({"id":id,"model":1,"input":{"Sequence":{"tokens":[1,2,3,5,8,13]}},"workload":{"Generate":{"max_new_tokens":5}}})).map_err(|e| Error::invalid(e.to_string()))?;
    request.workload = workload;
    Ok(request)
}
fn finish(engine: &mut Engine<MetalBackend>, id: RequestId) -> Result<Option<WorkloadOutput>> {
    let limit = Instant::now() + Duration::from_secs(10);
    while !engine.is_idle() {
        if Instant::now() >= limit {
            return Err(Error::invariant("Metal retention test stalled"));
        }
        engine.tick(engine.now_us() + 1)?;
        std::thread::sleep(Duration::from_micros(100));
    }
    Ok(engine.take_completed(id)?.and_then(|done| done.output))
}
#[test]
fn generation_retains_one_hidden_row_instead_of_max_context_history() -> Result<()> {
    if !MetalBackend::available() {
        return Ok(());
    }
    let mut backend = backend(0)?;
    let capacity = backend.model().max_sequence;
    let full = backend
        .state_reservation_bytes_for(capacity, OutputReadout::Full)?
        .ok_or_else(|| Error::invariant("missing full budget"))?;
    let compact = backend
        .state_reservation_bytes_for(capacity, OutputReadout::Logits)?
        .ok_or_else(|| Error::invariant("missing generation budget"))?;
    assert_eq!(
        full - compact,
        ((capacity - 1) * backend.model().hidden_size * 4) as u64
    );
    backend.reserve_state_for(StateId::ONE, capacity, OutputReadout::Logits)?;
    assert_eq!(backend.inspect().reserved_bytes, compact);
    backend.reset_state(StateId::ONE)?;
    assert_eq!(backend.inspect().reserved_bytes, compact);
    backend.release_state(StateId::ONE)?;
    assert_eq!(backend.inspect().reserved_bytes, 0);
    Ok(())
}
#[test]
fn compact_generation_prefix_never_corrupts_full_readout_and_can_be_upgraded() -> Result<()> {
    if !MetalBackend::available() {
        return Ok(());
    }
    let workload = Workload::Embed {
        pooling: Pooling::Mean,
        dimensions: None,
        normalize: false,
    };
    let mut baseline = engine(0)?;
    baseline.submit(request(1, workload.clone())?)?;
    let expected = finish(&mut baseline, RequestId::ONE)?;
    let mut cached = engine(1 << 20)?;
    cached.submit(request(1, Workload::Generate { max_new_tokens: 5 })?)?;
    assert_eq!(
        finish(&mut cached, RequestId::ONE)?,
        Some(WorkloadOutput::Tokens(vec![25, 3, 3, 3, 3].into()))
    );
    cached.submit(request(2, workload.clone())?)?;
    assert_eq!(finish(&mut cached, RequestId::new(2)?)?, expected);
    cached.submit(request(3, workload)?)?;
    assert_eq!(finish(&mut cached, RequestId::new(3)?)?, expected);
    assert!(
        cached
            .backend()
            .inspect()
            .kv_cache
            .is_some_and(|kv| kv.reused_tokens >= 4)
    );
    cached.check_invariants()
}
