use super::{BlockedOn, QueueState, QueueTiming, ReadyKey, RequestQueue};
use infer_core::{Error, RequestId, Result, StepId};

impl RequestQueue {
    /// Validate all members before changing any ownership.
    /// # Errors
    /// Rejects duplicate/unknown identities or non-ready work without a partial transition.
    pub fn dispatch(&mut self, step: StepId, requests: &[RequestId]) -> Result<()> {
        if requests.is_empty()
            || self
                .flights
                .from((step, RequestId::ONE))
                .next()
                .is_some_and(|key| key.0 == step)
            || requests.iter().enumerate().any(|(at, id)| {
                requests[..at].contains(id) || self.state(*id) != Some(QueueState::Ready)
            })
        {
            return Err(Error::invariant("invalid queue batch dispatch"));
        }
        for id in requests {
            let entry = self
                .entries
                .get(id)
                .ok_or_else(|| Error::invariant("dispatch member missing"))?;
            let key = ReadyKey::from(entry);
            self.wait_expiry
                .remove(&(entry.request.wait_deadline_us, *id));
            self.index(key, false);
            if let Some(entry) = self.entries.get_mut(id) {
                entry.state = QueueState::Running { step };
            }
        }
        for id in requests {
            self.flights.insert((step, *id));
        }
        self.running += requests.len();
        Ok(())
    }
    /// # Errors
    /// Rejects stale/wrong-step/duplicate fences. Cancelled members remain owned until this call.
    pub fn complete(&mut self, id: RequestId, step: StepId) -> Result<bool> {
        let cancelled = self.acknowledge_flight(id, step)?;
        let entry = self
            .entries
            .get_mut(&id)
            .ok_or_else(|| Error::invariant("completion owner missing"))?;
        entry.state = QueueState::Ready;
        let key = ReadyKey::from(&*entry);
        self.wait_expiry
            .insert((entry.request.wait_deadline_us, id));
        self.index(key, true);
        Ok(cancelled)
    }
    fn acknowledge_flight(&mut self, id: RequestId, step: StepId) -> Result<bool> {
        let state = self
            .state(id)
            .ok_or_else(|| Error::invalid("unknown queue completion"))?;
        let cancelled = state == QueueState::CancelPending { step };
        if !cancelled && state != (QueueState::Running { step }) {
            return Err(Error::invariant("stale queue completion"));
        }
        if !self.flights.remove(&(step, id)) {
            return Err(Error::invariant("queue flight member missing"));
        }
        if cancelled {
            self.cancelling -= 1;
        } else {
            self.running -= 1;
        }
        Ok(cancelled)
    }
    /// Complete directly into the CPU/resource wait lane, without transient ready-index churn.
    /// # Errors
    /// Rejects stale fences; all flight ownership checks precede the new wait publication.
    pub fn complete_blocked(
        &mut self,
        id: RequestId,
        step: StepId,
        reason: BlockedOn,
        epoch: u64,
    ) -> Result<bool> {
        let cancelled = self.acknowledge_flight(id, step)?;
        let entry = self
            .entries
            .get_mut(&id)
            .ok_or_else(|| Error::invariant("completion owner missing"))?;
        entry.state = QueueState::Blocked { reason, epoch };
        self.wait_expiry
            .insert((entry.request.wait_deadline_us, id));
        self.waiters.insert((reason, epoch, id));
        Ok(cancelled)
    }
    /// Commit the final phase cursor and ready priority in one transition after the exact fence.
    /// # Errors
    /// Rejects invalid phases or stale fences without publishing intermediate priorities.
    pub fn complete_ready(
        &mut self,
        id: RequestId,
        step: StepId,
        timing: QueueTiming,
    ) -> Result<bool> {
        if timing.phase == infer_ir::ExecutionRole::Mixed {
            return Err(Error::invalid("invalid ready continuation"));
        }
        let cancelled = self.acknowledge_flight(id, step)?;
        let entry = self
            .entries
            .get_mut(&id)
            .ok_or_else(|| Error::invariant("completion owner missing"))?;
        entry.state = QueueState::Ready;
        entry.request.phase = timing.phase;
        entry.request.last_service_us = timing.last_service_us;
        entry.request.deadline_us = timing.deadline_us;
        entry.request.wait_deadline_us = timing.wait_deadline_us;
        let key = ReadyKey::from(&*entry);
        self.wait_expiry.insert((timing.wait_deadline_us, id));
        self.index(key, true);
        Ok(cancelled)
    }
    /// # Errors
    /// Rejects unknown/non-ready requests; removes blocked work from every ready priority index.
    pub fn block(&mut self, id: RequestId, reason: BlockedOn, epoch: u64) -> Result<()> {
        let entry = self
            .entries
            .get(&id)
            .ok_or_else(|| Error::invalid("unknown blocked request"))?;
        if entry.state != QueueState::Ready {
            return Err(Error::invariant("only ready requests can block"));
        }
        let key = ReadyKey::from(entry);
        self.index(key, false);
        self.waiters.insert((reason, epoch, id));
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.state = QueueState::Blocked { reason, epoch };
        }
        Ok(())
    }
    /// # Errors
    /// Rejects duplicate acknowledgements and missing waiter ownership.
    pub fn unblock(&mut self, id: RequestId) -> Result<()> {
        let entry = self
            .entries
            .get(&id)
            .ok_or_else(|| Error::invalid("unknown waiter"))?;
        let QueueState::Blocked { reason, epoch } = entry.state else {
            return Err(Error::invariant("request is not blocked"));
        };
        let key = ReadyKey::from(entry);
        if !self.waiters.remove(&(reason, epoch, id)) {
            return Err(Error::invariant("waiter acknowledgement lost ownership"));
        }
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.state = QueueState::Ready;
        }
        self.index(key, true);
        Ok(())
    }
    /// # Errors
    /// Running cancellation keeps flight ownership. Repeated cancellation is idempotent.
    pub fn cancel(&mut self, id: RequestId) -> Result<()> {
        let entry = self
            .entries
            .get_mut(&id)
            .ok_or_else(|| Error::invalid("unknown queue cancellation"))?;
        if let QueueState::Running { step } = entry.state {
            entry.state = QueueState::CancelPending { step };
            self.running -= 1;
            self.cancelling += 1;
            if let Some(deadline) = entry.request.hard_deadline_us {
                self.hard_expiry.remove(&(deadline, id));
            }
            return Ok(());
        }
        if matches!(entry.state, QueueState::CancelPending { .. }) {
            return Ok(());
        }
        self.remove(id)
    }
    /// # Errors
    /// Rejects removal before the exact device fence.
    pub fn remove(&mut self, id: RequestId) -> Result<()> {
        let Some(entry) = self.entries.get(&id) else {
            return Ok(());
        };
        if matches!(
            entry.state,
            QueueState::Running { .. } | QueueState::CancelPending { .. }
        ) {
            return Err(Error::invariant("queue removal before completion fence"));
        }
        let key = ReadyKey::from(entry);
        let state = entry.state;
        self.wait_expiry
            .remove(&(entry.request.wait_deadline_us, id));
        if let Some(deadline) = entry.request.hard_deadline_us {
            self.hard_expiry.remove(&(deadline, id));
        }
        if let QueueState::Blocked { reason, epoch } = state {
            self.waiters.remove(&(reason, epoch, id));
        }
        if state == QueueState::Ready {
            self.index(key, false);
        }
        let entry = self
            .entries
            .remove(&id)
            .ok_or_else(|| Error::invariant("removed owner missing"))?;
        let tenant = self
            .tenants
            .get_mut(&entry.request.tenant)
            .ok_or_else(|| Error::invariant("tenant owner missing"))?;
        tenant.owners -= 1;
        if tenant.owners == 0 {
            self.tenant_service.remove(&(tenant.finish, entry.tenant));
            self.names[entry.tenant] = None;
            self.free_tenants.push(entry.tenant);
            self.tenants.remove(&entry.request.tenant);
        }
        Ok(())
    }
}
