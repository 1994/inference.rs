use infer_backend_reference::{ReferenceBackend, ReferenceKernels, ReferenceModel};
use infer_core::{Error, ModelId, RequestId, Result};
use infer_ir::{CanonicalRequest, PrecisionPlan};
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, RuntimeConfig, RuntimeSnapshot};
use std::{time::Duration, time::Instant};

fn fixture() -> Result<(ReferenceBackend, KernelRegistry)> {
    let mut registry = KernelRegistry::default();
    registry.register(&ReferenceKernels)?;
    Ok((
        ReferenceBackend::new(ReferenceModel::fixture(ModelId::ONE, 7))?,
        registry,
    ))
}
fn request() -> Result<CanonicalRequest> {
    serde_json::from_value(serde_json::json!({"id":1,"model":1,"input":{"Sequence":{"tokens":[1,2,3]}},"workload":{"Generate":{"max_new_tokens":3}}}))
        .map_err(|error| Error::invalid(error.to_string()))
}
fn wait(limit: Instant) -> Result<()> {
    if Instant::now() >= limit {
        return Err(Error::invariant("checkpoint owner did not converge"));
    }
    std::thread::sleep(Duration::from_micros(100));
    Ok(())
}
#[test]
fn checkpoint_is_acknowledged_by_owner_and_restored_states_rejoin_async_pipeline() -> Result<()> {
    let (backend, registry) = fixture()?;
    let model = backend.model().ir.clone();
    let engine = Engine::new(
        backend,
        model,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig::default(),
    )?;
    let mut engine = engine.into_threaded(4, Duration::from_millis(20))?;
    let prepared = engine.request_preparer()?.prepare(request()?)?;
    let mut quote = engine.admission_quote(&prepared)?;
    let limit = Instant::now() + Duration::from_secs(5);
    let bytes = loop {
        if let Some(infer_spi::ResourceReply::ReservationBytes(bytes)) = quote.poll()? {
            break bytes;
        }
        wait(limit)?;
    };
    engine.submit_quoted_with_trace(prepared, bytes, None)?;
    let snapshot = loop {
        let (_, checkpoint) = engine.quiesce(engine.now_us() + 1)?;
        if let Some(checkpoint) = checkpoint {
            break checkpoint;
        }
        wait(limit)?;
    };
    assert!(snapshot.checkpoint_supported);
    assert!(snapshot.inflight.is_none());
    let serialized = serde_json::to_vec(&snapshot).map_err(|e| Error::invalid(e.to_string()))?;
    let snapshot: RuntimeSnapshot =
        serde_json::from_slice(&serialized).map_err(|e| Error::invalid(e.to_string()))?;
    let (backend, registry) = fixture()?;
    let mut resumed = Engine::restore(backend, &registry, snapshot)?
        .into_threaded(4, Duration::from_millis(20))?;
    while !resumed.is_idle() {
        resumed.tick(resumed.now_us() + 1)?;
        wait(limit)?;
    }
    resumed.check_invariants()?;
    let done = resumed
        .take_completed(RequestId::ONE)?
        .ok_or_else(|| Error::invariant("restored completion lost"))?;
    assert_eq!(done.measurement.output_tokens, 3);
    assert!(done.measurement.successful);
    while !engine.is_idle() {
        engine.tick(engine.now_us() + 1)?;
        wait(limit)?;
    }
    Ok(())
}
