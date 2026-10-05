//! Invariants responsibilities.
use super::Engine;
use infer_core::{Error, RequestId, RequestStatus, Result, event::EventKind, event::ObjectKind};
use infer_spi::{BackendProvider, SchedulingPolicy};
use std::collections::BTreeSet;

impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    pub(crate) fn progress(&mut self, id: RequestId) -> Result<()> {
        self.global_progress_epoch = self.global_progress_epoch.saturating_add(1);
        self.host
            .requests
            .get_mut(id)
            .ok_or_else(|| Error::invariant("registered request"))?
            .progress_epoch = self.global_progress_epoch;
        self.event(
            EventKind::Progress,
            ObjectKind::Request,
            id.get(),
            0,
            self.global_progress_epoch,
            0,
        );
        Ok(())
    }
    ///
    /// # Errors
    /// Returns an invariant error for inconsistent identities, ownership, capacity, or execution cursors.
    pub fn check_invariants(&self) -> Result<()> {
        self.check_history_storage()?;
        self.check_tenant_accounting()?;
        self.check_resource_ownership()?;
        self.check_output_ownership()?;
        self.state.check_invariants()?;
        self.host.queues.check_invariants()?;
        if self.host.queues.inspect().capacity != self.config.max_requests {
            return Err(Error::invariant("queue capacity mismatch"));
        }
        if self.seen_requests.len() > self.config.history_capacity
            || self
                .seen_requests
                .iter()
                .any(|id| id.get() <= self.retired_request_floor)
            || self
                .host
                .requests
                .keys()
                .any(|id| id.get() > self.retired_request_floor && !self.seen_requests.contains(id))
        {
            return Err(Error::invariant(
                "invalid request identity retention window",
            ));
        }
        let running: BTreeSet<_> = self
            .inflight
            .as_ref()
            .map(|s| s.step.work.iter().map(|w| w.request).collect())
            .unwrap_or_default();
        let mut active = 0;
        for (id, r) in self.host.requests.iter() {
            let queued = self.host.queues.state(*id);
            let matches = match (&r.status, queued) {
                (RequestStatus::Runnable, Some(infer_scheduler::QueueState::Ready))
                | (RequestStatus::Waiting(_), Some(infer_scheduler::QueueState::Blocked { .. })) => {
                    true
                }
                (
                    RequestStatus::Running { step },
                    Some(infer_scheduler::QueueState::Running { step: owner }),
                ) => *step == owner && r.pending_finish.is_none(),
                (
                    RequestStatus::Running { step },
                    Some(infer_scheduler::QueueState::CancelPending { step: owner }),
                ) => *step == owner && r.pending_finish.is_some(),
                (RequestStatus::Waiting(_), None)
                    if (self.resources_pending.contains_key(id)
                        || self.output_owners.contains_key(id))
                        && r.pending_finish.is_some() =>
                {
                    true
                }
                (status, None) => status.terminal(),
                _ => false,
            };
            if !matches {
                return Err(Error::invariant(
                    "runtime request status differs from queue ownership",
                ));
            }
            if *id != r.request.id {
                return Err(Error::invariant("request identity mismatch"));
            }
            if r.status.terminal() {
                if r.state.is_some() || r.completed.is_none() || running.contains(id) {
                    return Err(Error::invariant("terminal request leaks state/step"));
                }
            } else {
                active += 1;
                let s = self.state.get(
                    r.state
                        .ok_or_else(|| Error::invariant("accepted request missing state"))?,
                )?;
                if s.owner != *id {
                    return Err(Error::invariant("state owned by different request"));
                }
                if matches!(r.status, RequestStatus::Running { .. }) != running.contains(id) {
                    return Err(Error::invariant(
                        "running requests do not equal submitted step",
                    ));
                }
            }
        }
        if active != self.state.snapshot().sequence_count {
            return Err(Error::invariant("orphaned sequence state"));
        }
        Ok(())
    }
    pub(crate) fn validate_backend_state(&self) -> Result<()> {
        let states = self.checkpoint_states()?;
        self.backend.validate_state_ownership(&states)
    }
}
