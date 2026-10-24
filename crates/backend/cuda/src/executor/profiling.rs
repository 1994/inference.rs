//! Opt-in host-side execution evidence; no token contents are recorded.
use super::CudaBackend;
use crate::device::CudaDevice;
use crate::loading::PooledServingQuote;
use infer_ir::{ExecutionInput, ExecutionTask};
use std::collections::BTreeMap;
use std::env::var_os;
use std::fs::OpenOptions;
use std::{
    cell::RefCell,
    io::Write,
    path::{Path, PathBuf},
    sync::OnceLock,
    time::{Duration, Instant},
};

/// Set before process startup to append task grouping and slot occupancy as JSONL.
const PROFILE_ENV: &str = "INFER_CUDA_EXECUTION_PROFILE";
static OUTPUT: OnceLock<Option<PathBuf>> = OnceLock::new();

thread_local! {
    /// Host wall time accumulated per named phase while a step is in flight. Only written
    /// when the execution profile is enabled, so the steady path pays one `Option` check.
    static PHASES: RefCell<Option<Vec<(&'static str, Duration)>>> = const { RefCell::new(None) };
    /// Previous allocatable byte count within one unpublished pool capture.
    static PREVIOUS_ALLOCATABLE: RefCell<Option<u64>> = const { RefCell::new(None) };
}

/// Scratch phase accumulator bound to the calling thread for one step.
pub(super) struct PhaseScope;

impl PhaseScope {
    /// Install the accumulator; the caller must hold the guard for the whole step.
    pub(super) fn install(enabled: bool) -> Option<Self> {
        if !enabled {
            return None;
        }
        PHASES.with(|phases| *phases.borrow_mut() = Some(Vec::new()));
        Some(Self)
    }

    /// Take the accumulated phases, ending the scope.
    #[expect(
        clippy::unused_self,
        reason = "the guard marks scope lifetime; taking the accumulator is a side effect"
    )]
    pub(super) fn finish(self) -> Vec<(&'static str, Duration)> {
        PHASES.with(|phases| phases.borrow_mut().take().unwrap_or_default())
    }
}

/// Time one host-side phase. A no-op unless a [`PhaseScope`] is installed.
///
/// Phases must not nest: the accumulator is a `RefCell` borrowed for the duration of the
/// body, so an inner `phase` re-enters the same thread-local and panics. Time the finest
/// granularity needed and leave the enclosing region unmeasured.
pub(super) fn phase<T>(name: &'static str, body: impl FnOnce() -> T) -> T {
    PHASES.with(|phases| {
        let mut phases = phases.borrow_mut();
        let Some(recorded) = phases.as_mut() else {
            drop(phases);
            return body();
        };
        let started = Instant::now();
        let value = body();
        recorded.push((name, started.elapsed()));
        value
    })
}

pub(super) struct StepProfile {
    path: &'static PathBuf,
    started: Instant,
    inputs: Vec<serde_json::Value>,
}

impl StepProfile {
    pub(super) fn start(backend: &CudaBackend, tasks: &[ExecutionTask]) -> Option<Self> {
        let path = output_path()?;
        let inputs = tasks
            .iter()
            .map(|task| {
                let state = backend.states.get(&task.state);
                let position = state.map_or(0, |state| state.history.len());
                let kind = match &task.tokens {
                    ExecutionInput::Decode { .. } => "decode",
                    ExecutionInput::Prefill { .. } => "prefill",
                    ExecutionInput::Full(_) => "full",
                };
                serde_json::json!({
                    "request": task.request,
                    "state": task.state,
                    "kind": kind,
                    "position": position,
                    "input_tokens": task.tokens.delta(position).map_or(0, <[u32]>::len),
                    "capacity": state.map(|state| state.capacity),
                    "slot_before": state.and_then(|state| state.slot.as_ref().map(|lease| lease.slot)),
                })
            })
            .collect();
        Some(Self {
            path,
            started: Instant::now(),
            inputs,
        })
    }

    pub(super) fn finish(
        self,
        backend: &CudaBackend,
        tasks: &[ExecutionTask],
        outputs: Option<&[infer_ir::TaskOutput]>,
        phases: Vec<(&'static str, Duration)>,
    ) {
        let elapsed_us = self.started.elapsed().as_micros();
        let mut by_phase: BTreeMap<&'static str, u64> = BTreeMap::new();
        for (name, elapsed) in phases {
            *by_phase.entry(name).or_default() += u64::try_from(elapsed.as_micros()).unwrap_or(0);
        }
        let slots: Vec<_> = tasks
            .iter()
            .map(|task| {
                backend
                    .states
                    .get(&task.state)
                    .and_then(|state| state.slot.as_ref().map(|lease| lease.slot))
            })
            .collect();
        let line = serde_json::json!({
            "execution_wall_us": elapsed_us,
            "host_phases_us": by_phase,
            "tasks": self.inputs,
            "slots_after": slots,
            "pool_width": backend.slots.as_ref().map(super::SlotPool::width),
            "worker_prompt_width": backend.worker.as_ref().map(|worker| worker.program.prefill_width()),
            "pool_disabled": backend.slots_disabled,
            "draft_pool_width": backend.draft_slots.as_ref().map(super::SlotPool::width),
            "successful": outputs.is_some(),
            "accepted_tokens": outputs.map(|rows| rows.iter().map(|row| row.output.tokens.len()).collect::<Vec<_>>()),
        });
        let result = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.path)
            .and_then(|mut file| writeln!(file, "{line}"));
        if let Err(error) = result {
            eprintln!("CUDA execution profile write failed: {error}");
        }
    }
}

/// Keep fallback causes alongside grouping evidence instead of silently losing batching.
pub(super) fn pool_failure(kind: &str, width: usize, capacity: usize, error: &infer_core::Error) {
    let Some(path) = output_path() else {
        return;
    };
    let line = serde_json::json!({"pool_kind": kind, "pool_creation_failed": error.to_string(), "width": width, "capacity": capacity});
    if let Err(error) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut file| writeln!(file, "{line}"))
    {
        eprintln!("CUDA execution profile write failed: {error}");
    }
}

fn output_path() -> Option<&'static PathBuf> {
    OUTPUT
        .get_or_init(|| {
            var_os(PROFILE_ENV)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        })
        .as_ref()
}

fn append_allocation_record(path: &Path, line: &serde_json::Value) {
    if let Err(error) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut file| writeln!(file, "{line}"))
    {
        eprintln!("CUDA allocation profile write failed: {error}");
    }
}

/// Preserve the exact quote before its unpublished capture can be changed or discarded.
pub(super) fn plan(attempt: usize, envelope: u64, quote: &PooledServingQuote) {
    let Some(path) = output_path() else {
        return;
    };
    append_allocation_record(
        path,
        &serde_json::json!({
            "capture_attempt": attempt,
            "pool_plan_quote": quote,
            "envelope_bytes": envelope,
        }),
    );
}

fn memory_increment(stage: &str, previous: Option<u64>, current: u64) -> u64 {
    if stage == "before_pool" {
        0
    } else {
        previous.map_or(0, |bytes| bytes.saturating_sub(current))
    }
}

/// Startup allocation evidence shares the opt-in host profile and never synchronizes the GPU.
pub(super) fn memory(attempt: usize, stage: &str, width: usize, device: &CudaDevice) {
    let Some(path) = output_path() else {
        return;
    };
    let (Ok((free, total)), Ok(reusable)) = (device.memory_info(), device.pool_reclaimable_bytes())
    else {
        return;
    };
    let allocatable = free.saturating_add(reusable).min(total);
    let incremental = PREVIOUS_ALLOCATABLE.with(|previous| {
        let incremental = memory_increment(stage, *previous.borrow(), allocatable);
        *previous.borrow_mut() = Some(allocatable);
        incremental
    });
    append_allocation_record(
        path,
        &serde_json::json!({
            "capture_attempt": attempt,
            "memory_stage": stage,
            "width": width,
            "driver_free_bytes": free,
            "pool_unused_bytes": reusable,
            "allocatable_bytes": allocatable,
            "incremental_allocatable_decrease_bytes": incremental,
        }),
    );
}

fn capture_outcome_record(
    attempt: usize,
    outcome: &str,
    quoted: u64,
    allocated: Option<u64>,
    envelope: u64,
    next_funding: Option<u64>,
    error: Option<&str>,
) -> serde_json::Value {
    serde_json::json!({
        "capture_attempt": attempt,
        "pool_capture_outcome": outcome,
        "quoted_bytes": quoted,
        "allocated_bytes": allocated,
        "unquoted_bytes": allocated.map(|bytes| bytes.saturating_sub(quoted)),
        "allocated_over_envelope_bytes": allocated.map(|bytes| bytes.saturating_sub(envelope)),
        "envelope_bytes": envelope,
        "next_quote_budget": next_funding,
        "error": error,
    })
}

/// Join accepted and rejected captures to their quote and subsequent replan funding.
pub(super) fn capture_outcome(
    attempt: usize,
    outcome: &str,
    quoted: u64,
    allocated: Option<u64>,
    envelope: u64,
    next_funding: Option<u64>,
    error: Option<&str>,
) {
    let Some(path) = output_path() else {
        return;
    };
    append_allocation_record(
        path,
        &capture_outcome_record(
            attempt,
            outcome,
            quoted,
            allocated,
            envelope,
            next_funding,
            error,
        ),
    );
}

#[cfg(test)]
#[path = "../../tests/unit/executor_profiling.rs"]
mod tests;
