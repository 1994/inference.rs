//! Execution CLI support.
#[cfg(any(target_os = "macos", feature = "test-backends"))]
use crate::backend::SelectedBackend;
#[cfg(any(target_os = "macos", feature = "test-backends"))]
use infer_core::{Error, RequestId, Result};
#[cfg(any(target_os = "macos", feature = "test-backends"))]
use infer_ir::CanonicalRequest;
#[cfg(feature = "test-backends")]
use infer_ir::WorkloadOutput;
#[cfg(any(target_os = "macos", feature = "test-backends"))]
use infer_runtime::Engine;

#[cfg(any(target_os = "macos", feature = "test-backends"))]
pub fn run_to_idle(
    engine: &mut Engine<SelectedBackend>,
) -> Result<Vec<infer_core::event::SemanticEvent>> {
    let mut events = engine.drain_events();
    let start = engine.inspect().global_progress_epoch;
    let clock = engine.now_us();
    for tick in 0..1_000_000u64 {
        if engine.is_idle() {
            return Ok(events);
        }
        engine.tick(clock + tick)?;
        events.extend(engine.drain_events());
        if engine.program().backend.is_device() && engine.inspect().inflight_step.is_some() {
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
    }
    Err(Error::invariant(format!(
        "runtime exceeded tick limit; initial progress epoch {start}"
    )))
}
#[cfg(any(target_os = "macos", feature = "test-backends"))]
pub fn results(
    engine: &Engine<SelectedBackend>,
    ids: &[RequestId],
) -> Result<Vec<infer_runtime::CompletedRequest>> {
    ids.iter()
        .map(|id| {
            engine
                .request(*id)?
                .completed
                .clone()
                .ok_or_else(|| Error::invariant("request has no completion"))
        })
        .collect()
}
#[cfg(feature = "test-backends")]
pub fn outputs(
    engine: &Engine<SelectedBackend>,
    ids: &[RequestId],
) -> Result<Vec<Option<WorkloadOutput>>> {
    Ok(results(engine, ids)?
        .into_iter()
        .map(|r| r.output)
        .collect())
}
#[cfg(any(target_os = "macos", feature = "test-backends"))]
pub fn run_requests(
    engine: &mut Engine<SelectedBackend>,
    requests: Vec<CanonicalRequest>,
) -> Result<Vec<infer_core::event::SemanticEvent>> {
    for request in requests {
        engine.submit(request)?;
    }
    run_to_idle(engine)
}
