//! Opt-in host-side execution evidence; no token contents are recorded.
use super::CudaBackend;
use infer_ir::{ExecutionInput, ExecutionTask};
use std::{io::Write, path::PathBuf, sync::OnceLock, time::Instant};

/// Set before process startup to append task grouping and slot occupancy as JSONL.
const PROFILE_ENV: &str = "INFER_CUDA_EXECUTION_PROFILE";
static OUTPUT: OnceLock<Option<PathBuf>> = OnceLock::new();

pub(super) struct StepProfile {
    path: &'static PathBuf,
    started: Instant,
    inputs: Vec<serde_json::Value>,
}

impl StepProfile {
    pub(super) fn start(backend: &CudaBackend, tasks: &[ExecutionTask]) -> Option<Self> {
        let path = OUTPUT
            .get_or_init(|| {
                std::env::var_os(PROFILE_ENV)
                    .filter(|v| !v.is_empty())
                    .map(PathBuf::from)
            })
            .as_ref()?;
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
    ) {
        let elapsed_us = self.started.elapsed().as_micros();
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
            "tasks": self.inputs,
            "slots_after": slots,
            "pool_width": backend.slots.as_ref().map(super::SlotPool::width),
            "pool_disabled": backend.slots_disabled,
            "draft_pool_width": backend.draft_slots.as_ref().map(super::SlotPool::width),
            "successful": outputs.is_some(),
            "accepted_tokens": outputs.map(|rows| rows.iter().map(|row| row.output.tokens.len()).collect::<Vec<_>>()),
        });
        let result = std::fs::OpenOptions::new()
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
    let Some(path) = OUTPUT.get().and_then(Option::as_ref) else {
        return;
    };
    let line = serde_json::json!({"pool_kind": kind, "pool_creation_failed": error.to_string(), "width": width, "capacity": capacity});
    if let Err(error) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut file| writeln!(file, "{line}"))
    {
        eprintln!("CUDA execution profile write failed: {error}");
    }
}
