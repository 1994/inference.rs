//! Execution CLI support.
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
use crate::backend::SelectedBackend;
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
use infer_core::{Error, RequestId, Result};
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
use infer_ir::CanonicalRequest;
#[cfg(feature = "test-backends")]
use infer_ir::WorkloadOutput;
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
use infer_runtime::Engine;

#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
pub fn run_to_idle(
    engine: &mut Engine<SelectedBackend>,
) -> Result<Vec<infer_core::event::SemanticEvent>> {
    run_to_idle_started(engine, engine.now_us(), std::time::Instant::now())
}

#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
fn run_to_idle_started(
    engine: &mut Engine<SelectedBackend>,
    clock: u64,
    started: std::time::Instant,
) -> Result<Vec<infer_core::event::SemanticEvent>> {
    let mut events = engine.drain_events();
    let start = engine.inspect().global_progress_epoch;
    for tick in 0..crate::constants::MAX_IDLE_TICKS {
        if engine.is_idle() {
            return Ok(events);
        }
        let elapsed = if engine.program().backend.is_device() {
            u64::try_from(started.elapsed().as_micros())
                .map_err(|_| Error::invalid("CLI clock overflow"))?
        } else {
            tick
        };
        engine.tick(
            clock
                .checked_add(elapsed)
                .ok_or_else(|| Error::invalid("CLI clock overflow"))?,
        )?;
        events.extend(engine.drain_events());
        if engine.program().backend.is_device() && engine.inspect().inflight_step.is_some() {
            std::thread::sleep(crate::constants::DEVICE_POLL_INTERVAL);
        }
    }
    Err(Error::invariant(format!(
        "runtime exceeded tick limit; initial progress epoch {start}"
    )))
}
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
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
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
pub fn run_requests(
    engine: &mut Engine<SelectedBackend>,
    requests: Vec<CanonicalRequest>,
) -> Result<Vec<infer_core::event::SemanticEvent>> {
    let clock = engine.now_us();
    let started = std::time::Instant::now();
    for request in requests {
        engine.submit(request)?;
    }
    run_to_idle_started(engine, clock, started)
}
