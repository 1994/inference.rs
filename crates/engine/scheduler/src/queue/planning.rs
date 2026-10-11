use super::{BlockedOn, FairKey, QueueState, RequestQueue};
use infer_core::{Error, RequestId, Result, StepId};
use std::{
    cmp::Reverse,
    ops::Bound::{Excluded, Unbounded},
};

impl RequestQueue {
    fn next_tenant(&self, head: FairKey) -> Option<FairKey> {
        self.tenant_ready
            .range((Excluded((head.3, head.1, head.2)), Unbounded))
            .next()
            .filter(|key| key.0 == head.3)
            .map(|(_, time, id)| (head.0, *time, *id, head.3))
    }
    /// Capture at most K candidates: aging, latest-start urgency, then hierarchical WFQ.
    /// An unsuccessful window continues by request identity on the next attempt, so a
    /// resource-heavy priority prefix cannot hide all executable work. Scratch never grows.
    /// # Errors
    /// Rejects insufficient scratch or corrupt lifecycle ownership.
    pub fn candidates_into(
        &mut self,
        output: &mut Vec<RequestId>,
        limit: usize,
        now: u64,
        max_wait: u64,
        urgency: u64,
    ) -> Result<()> {
        output.clear();
        let limit = limit.min(self.capacity);
        if output.capacity() < limit {
            return Err(Error::new(
                infer_core::ErrorCode::Capacity,
                "candidate scratch too small",
            ));
        }
        if limit == 0 {
            return Ok(());
        }
        if let Some(after) = self.scan_after {
            output.extend(
                self.ready
                    .range((Excluded(after), Unbounded))
                    .chain(self.ready.range(..=after))
                    .take(limit)
                    .copied(),
            );
            return Ok(());
        }
        self.scratch.begin();
        if let Some(threshold) = now.checked_sub(max_wait) {
            // Aging is an identity-unique index, so no duplicate lookups are needed here.
            for (_, id) in self
                .aging
                .iter()
                .take_while(|key| key.0 <= threshold)
                .take(limit)
            {
                let slot = self
                    .entries
                    .get(id)
                    .ok_or_else(|| Error::invariant("aging owner missing"))?
                    .slot;
                self.scratch.visit(slot);
                output.push(*id);
            }
        }
        let threshold = i128::from(now) + i128::from(urgency);
        for (_, id) in self
            .deadlines
            .iter()
            .take_while(|key| key.0 <= threshold)
            .take(limit)
        {
            if output.len() == limit {
                break;
            }
            let slot = self
                .entries
                .get(id)
                .ok_or_else(|| Error::invariant("urgency owner missing"))?
                .slot;
            if self.scratch.visit(slot) {
                output.push(*id);
            }
        }
        // The tree cursor is held separately from the mutable generation marks and merge heap.
        let mut outer = self.fair.iter().copied().peekable();
        for _ in 0..limit.saturating_mul(2) {
            if output.len() == limit {
                break;
            }
            let staged = self.scratch.frontier.peek().map(|Reverse(key)| *key);
            let head = match (outer.peek().copied(), staged) {
                (Some(head), Some(next)) if head <= next => outer.next(),
                (None | Some(_), Some(_)) => self.scratch.frontier.pop().map(|Reverse(key)| key),
                (Some(_), None) => outer.next(),
                (None, None) => None,
            };
            let Some(head) = head else {
                break;
            };
            let slot = self
                .entries
                .get(&head.2)
                .ok_or_else(|| Error::invariant("fair owner missing"))?
                .slot;
            if self.scratch.visit(slot) {
                output.push(head.2);
            }
            if let Some(next) = self.next_tenant(head) {
                self.scratch.frontier.push(Reverse(next));
            }
        }
        self.scratch.frontier.clear();
        Ok(())
    }
    /// Continue an unproductive captured window; dispatch resets normal priority selection.
    pub const fn advance_candidate_window(&mut self, last: RequestId) {
        self.scan_after = Some(last);
    }
    /// Update urgency after a cost epoch without changing blocked or flight ownership.
    /// # Errors
    /// Rejects missing request identities.
    pub fn update_priority(&mut self, id: RequestId, latest_start_us: Option<i128>) -> Result<()> {
        let entry = self
            .entries
            .get_mut(&id)
            .ok_or_else(|| Error::invalid("unknown priority owner"))?;
        if entry.state == QueueState::Ready {
            if let Some(previous) = entry.request.latest_start_us {
                self.deadlines.remove(&(previous, id));
            }
            if let Some(next) = latest_start_us {
                self.deadlines.insert((next, id));
            }
        }
        entry.request.latest_start_us = latest_start_us;
        Ok(())
    }
    /// Wake a bounded number of resource waiters into reusable scratch.
    /// # Errors
    /// Rejects insufficient scratch or corrupt waiter ownership.
    pub fn wake_into(
        &mut self,
        reason: BlockedOn,
        epoch: u64,
        limit: usize,
        output: &mut Vec<RequestId>,
    ) -> Result<()> {
        output.clear();
        if output.capacity() < limit.min(self.waiters.len()) {
            return Err(Error::new(
                infer_core::ErrorCode::Capacity,
                "waiter scratch too small",
            ));
        }
        output.extend(
            self.waiters
                .range((reason, 0, RequestId::ONE)..)
                .take_while(|key| key.0 == reason && key.1 < epoch)
                .take(limit)
                .map(|key| key.2),
        );
        for id in output.iter().copied() {
            self.unblock(id)?;
        }
        Ok(())
    }
    /// Compatibility inspection; the owner uses `wake_into` with fixed scratch.
    pub fn wake(&mut self, reason: BlockedOn, epoch: u64) -> Vec<RequestId> {
        let mut output = Vec::with_capacity(self.waiters.len());
        if self
            .wake_into(reason, epoch, self.capacity, &mut output)
            .is_err()
        {
            output.clear();
        }
        output
    }
    /// Capture bounded expiry work without quadratic duplicate searches. Flight wait timeouts pause.
    pub fn expired_into(&self, now: u64, limit: usize, output: &mut Vec<RequestId>) {
        output.clear();
        if limit == 0 {
            return;
        }
        output.extend(
            self.hard_expiry
                .iter()
                .take_while(|key| key.0 <= now)
                .take(limit)
                .map(|key| key.1),
        );
        let remaining = limit.saturating_sub(output.len());
        output.extend(
            self.wait_expiry
                .iter()
                .take_while(|key| key.0 <= now)
                .filter(|key| {
                    !self.entries.get(&key.1).is_some_and(|entry| {
                        entry
                            .request
                            .hard_deadline_us
                            .is_some_and(|deadline| deadline <= now)
                    })
                })
                .take(remaining)
                .map(|key| key.1),
        );
        output.sort_unstable();
    }
    #[must_use]
    pub fn expired_ids(&self, now: u64) -> Vec<RequestId> {
        let mut output = Vec::with_capacity(self.entries.len());
        self.expired_into(now, self.capacity, &mut output);
        output
    }
    pub fn ready_ids(&self) -> impl Iterator<Item = RequestId> + '_ {
        self.ready.iter().copied()
    }
    /// Identity-ordered cursor for bounded cost-epoch maintenance.
    pub fn ready_ids_after(
        &self,
        after: Option<RequestId>,
    ) -> impl Iterator<Item = RequestId> + '_ {
        self.ready
            .range((after.map_or(Unbounded, Excluded), Unbounded))
            .copied()
    }
    /// # Errors
    /// Independently rebuilds cold indexes and checks exact membership/counters and flight ownership.
    pub fn check_invariants(&self) -> Result<()> {
        self.scratch.validate(self.capacity)?;
        if self.names.len() != self.capacity
            || self.entries.capacity() != self.capacity
            || self.tenants.capacity() != self.capacity
        {
            return Err(Error::invariant("queue lifecycle capacity corrupt"));
        }
        let mut expected = Self::new(self.capacity)?;
        let mut flights = std::collections::BTreeMap::<StepId, Vec<RequestId>>::new();
        // Preserve interned slots; HashMap iteration never determines scheduling order.
        for (slot, name) in self.names.iter().enumerate() {
            if let Some(name) = name {
                let tenant = self
                    .tenants
                    .get(name)
                    .ok_or_else(|| Error::invariant("tenant name missing"))?;
                if tenant.slot != slot || tenant.owners == 0 {
                    return Err(Error::invariant("tenant slot mismatch"));
                }
                expected.names[slot] = Some(name.clone());
                expected.tenant_service.insert((tenant.finish, slot));
                expected.tenants.insert(
                    name.clone(),
                    super::Tenant {
                        slot,
                        finish: tenant.finish,
                        owners: 0,
                    },
                )?;
            }
        }
        let mut free = vec![false; self.capacity];
        for slot in &self.free_tenants {
            if *slot >= self.capacity || free[*slot] || self.names[*slot].is_some() {
                return Err(Error::invariant("tenant free slots corrupt"));
            }
            free[*slot] = true;
        }
        if self
            .names
            .iter()
            .enumerate()
            .any(|(slot, name)| name.is_none() != free[slot])
        {
            return Err(Error::invariant("tenant free slots incomplete"));
        }
        expected.free_tenants.clone_from(&self.free_tenants);
        let mut slots = vec![false; self.capacity];
        for slot in &self.free_entries {
            if *slot >= self.capacity || std::mem::replace(&mut slots[*slot], true) {
                return Err(Error::invariant("request free slots corrupt"));
            }
        }
        for (id, entry) in self.entries.iter() {
            if entry.slot >= self.capacity
                || std::mem::replace(&mut slots[entry.slot], true)
                || self.names.get(entry.tenant).and_then(Option::as_ref)
                    != Some(&entry.request.tenant)
            {
                return Err(Error::invariant("request slot or tenant ownership corrupt"));
            }
            if *id != entry.request.request {
                return Err(Error::invariant("queue entry identity mismatch"));
            }
            expected.enqueue(entry.request.clone())?;
            expected
                .entries
                .get_mut(id)
                .ok_or_else(|| Error::invariant("reference queue owner missing"))?
                .slot = entry.slot;
            if let QueueState::Blocked { reason, epoch } = entry.state {
                expected.block(*id, reason, epoch)?;
            }
            if let QueueState::Running { step } | QueueState::CancelPending { step } = entry.state {
                flights.entry(step).or_default().push(*id);
            }
        }
        for (step, ids) in flights {
            expected.dispatch(step, &ids)?;
            for id in ids {
                if matches!(self.state(id), Some(QueueState::CancelPending { .. })) {
                    expected.cancel(id)?;
                }
            }
        }
        if slots.contains(&false) {
            return Err(Error::invariant("request slots incomplete"));
        }
        expected.free_entries.clone_from(&self.free_entries);
        expected.scan_after = self.scan_after;
        if *self != expected {
            return Err(Error::invariant(
                "queue indexes disagree with lifecycle entries",
            ));
        }
        Ok(())
    }
}
