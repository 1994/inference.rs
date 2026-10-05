//! Capture responsibilities.
use super::{Engine, ReplayAction, RuntimeSnapshot};
use infer_core::{Error, ErrorCode, Result};
use infer_spi::{BackendProvider, SchedulingPolicy, WorkloadProvider};

impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    pub(crate) fn snapshot_for_diagnostic(&self) -> RuntimeSnapshot {
        RuntimeSnapshot {
            fault: self.fault.clone(),
            schema_version: 6,
            backend_identity: self.backend.identity().into(),
            policy_identity: self.policy.identity().into(),
            workload_identity: self.workloads.identity().into(),
            checkpoint_supported: self.backend.supports_control_checkpoint(),
            model: self.model.clone(),
            program: self.program.clone(),
            config: self.config.clone(),
            state: self.state.clone(),
            requests: self.host.requests.snapshot(),
            queues: self.host.queues.clone(),
            resource_epoch: self.resource_epoch,
            cost_epoch: self.host.cost_epoch,
            seen_requests: self.seen_requests.clone(),
            retired_request_floor: self.retired_request_floor,
            tenants: self.tenants.clone(),
            ids: self.ids.clone(),
            global_progress_epoch: self.global_progress_epoch,
            now_us: self.now_us,
            guard: self.guard.clone(),
            decisions: self.decisions.clone(),
            actions: self.actions.clone(),
            dropped_actions: self.dropped_actions,
            inflight: self.inflight.as_ref().map(|s| (*s.step).clone()),
            execution_state: None,
            cost_provider_identity: self.costs.identity().into(),
            admission_provider_identity: self.admission_policy.identity().into(),
            cost_state: None,
            pending_cost_observations: self.pending_cost_observations.clone(),
            preemption_focus: self.preemption_focus,
        }
    }
    /// Device tickets are not serializable. Capture at a submission boundary.
    ///
    /// # Errors
    /// Returns a conflict error while execution is in flight, or a state/cost-provider error while capturing a checkpoint.
    pub fn snapshot(&self) -> Result<RuntimeSnapshot> {
        if !self.resources_pending.is_empty() || !self.output_pending.is_empty() {
            return Err(Error::new(
                ErrorCode::Conflict,
                "resource acknowledgements must settle before checkpoint",
            ));
        }

        if self.fault.is_some() {
            return Err(Error::new(
                ErrorCode::Conflict,
                "quarantined engine snapshots are diagnostic evidence only",
            ));
        }
        if self.inflight.is_some() {
            return Err(Error::new(
                ErrorCode::Conflict,
                "checkpoint requires a quiescent submission boundary",
            ));
        }
        if !self.backend.supports_control_checkpoint() {
            return Err(Error::unsupported(
                "backend cannot reconstruct execution state from a control checkpoint",
            ));
        }
        self.check_invariants()?;
        self.validate_backend_state()?;
        let mut snapshot = self.snapshot_for_diagnostic();
        snapshot.execution_state = self.backend.capture_execution_state()?;
        snapshot.cost_state = self.costs.capture_state()?;
        Ok(snapshot)
    }
    ///
    /// # Errors
    /// Returns a serialization error if recorded replay actions cannot be encoded and decoded.
    pub fn journal(&self) -> Result<Vec<ReplayAction>> {
        if self.dropped_actions != 0 {
            return Err(Error::new(
                ErrorCode::Capacity,
                "action history was truncated; restore a checkpoint instead",
            ));
        }
        Ok(self.actions.iter().cloned().collect())
    }
    ///
    /// # Errors
    /// Returns validation, planning, backend, or invariant errors if an action cannot be applied to this engine.
    pub fn replay(&mut self, actions: &[ReplayAction]) -> Result<()> {
        self.replaying = true;
        let result = (|| {
            for action in actions {
                match action {
                    ReplayAction::Submit(request) => self.submit((**request).clone())?,
                    ReplayAction::Tick { now_us } => {
                        self.tick(*now_us)?;
                    }
                    ReplayAction::Poll { now_us } => {
                        self.poll_completed(*now_us)?;
                    }
                    ReplayAction::Quiesce { now_us } => {
                        self.quiesce(*now_us)?;
                    }
                    ReplayAction::Cancel(id) => {
                        self.cancel(*id)?;
                    }
                    ReplayAction::Drain(id) => {
                        self.take_completed(*id)?;
                    }
                    ReplayAction::CostFeedback(observation) => {
                        self.enqueue_cost_observation(observation.clone(), true)?;
                    }
                }
            }
            self.check_invariants()
        })();
        self.replaying = false;
        result
    }
}
