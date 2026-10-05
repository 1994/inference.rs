use super::{BlockedOn, FairKey, QueueState, RequestQueue, TenantKey, index::Index};
use infer_core::{Error, RequestId, Result, StepId};

impl RequestQueue {
    fn next_tenant(
        index: &Index<TenantKey>,
        head: FairKey,
        position: usize,
    ) -> Option<(FairKey, usize)> {
        index
            .next_at(position)
            .filter(|(_, key)| key.0 == head.3)
            .map(|(position, (_, time, id))| ((head.0, time, id, head.3), position))
    }
    /// Priority visits are bounded by K: aging, near deadlines, then hierarchical WFQ.
    /// Reuses caller scratch and fixed frontier nodes; never walks the request table.
    /// # Errors
    /// Rejects insufficient caller scratch or corrupt tenant frontier capacity without growing storage.
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
        if let Some(threshold) = now.checked_sub(max_wait) {
            for (_, id) in self
                .aging
                .iter()
                .take_while(|key| key.0 <= threshold)
                .take(limit)
            {
                output.push(id);
            }
        }
        for (_, id) in self
            .deadlines
            .iter()
            .take_while(|key| key.0 <= now.saturating_add(urgency))
            .take(limit)
        {
            if output.len() == limit {
                break;
            }
            if !output.contains(&id) {
                output.push(id);
            }
        }
        let mut outer = self.fair.iter().peekable();
        // At most K already visited items can precede the next fair item.
        for _ in 0..limit.saturating_mul(2) {
            if output.len() == limit {
                break;
            }
            let staged = self.frontier.first();
            let entry = match (outer.peek().copied(), staged) {
                (Some(head), Some((next, _))) if head <= next => {
                    outer.next();
                    self.tenant_ready
                        .position((head.3, head.1, head.2))
                        .map(|slot| (head, slot))
                }
                (None | Some(_), Some(next)) => {
                    self.frontier.pop();
                    Some(next)
                }
                (Some(head), None) => {
                    outer.next();
                    self.tenant_ready
                        .position((head.3, head.1, head.2))
                        .map(|slot| (head, slot))
                }
                (None, None) => None,
            };
            let Some((head, position)) = entry else {
                break;
            };
            if !output.contains(&head.2) {
                output.push(head.2);
            }
            if let Some(next) = Self::next_tenant(&self.tenant_ready, head, position)
                && let Err(error) = self.frontier.insert(next)
            {
                self.frontier.clear();
                output.clear();
                return Err(error);
            }
        }
        self.frontier.clear();
        Ok(())
    }
    /// Wake a bounded number of resource waiters into reusable scratch.
    /// # Errors
    /// Rejects corrupt waiter ownership rather than dropping an acknowledgement.
    pub fn wake_into(
        &mut self,
        reason: BlockedOn,
        epoch: u64,
        limit: usize,
        output: &mut Vec<RequestId>,
    ) -> Result<()> {
        output.clear();
        output.extend(
            self.waiters
                .from((reason, 0, RequestId::ONE))
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
    /// Populate caller scratch with bounded deadline work. Running wait deadlines are suspended.
    pub fn expired_into(&self, now: u64, limit: usize, output: &mut Vec<RequestId>) {
        output.clear();
        for (_, id) in self
            .hard_expiry
            .iter()
            .take_while(|key| key.0 <= now)
            .take(limit)
            .chain(
                self.wait_expiry
                    .iter()
                    .take_while(|key| key.0 <= now)
                    .take(limit),
            )
        {
            if !output.contains(&id) {
                output.push(id);
            }
            if output.len() == limit {
                break;
            }
        }
        output.sort_unstable();
    }
    #[must_use]
    pub fn expired_ids(&self, now: u64) -> Vec<RequestId> {
        let mut output = Vec::with_capacity(self.entries.len());
        self.expired_into(now, self.capacity, &mut output);
        output
    }
    pub fn ready_ids(&self) -> impl Iterator<Item = RequestId> + '_ {
        self.aging.iter().map(|key| key.1)
    }
    /// # Errors
    /// Independently rebuilds cold indexes and checks exact membership/counters and flight ownership.
    pub fn check_invariants(&self) -> Result<()> {
        self.tenant_service.validate()?;
        self.flights.validate()?;
        self.tenant_ready.validate()?;
        self.fair.validate()?;
        self.frontier.validate()?;
        self.aging.validate()?;
        self.deadlines.validate()?;
        self.hard_expiry.validate()?;
        self.wait_expiry.validate()?;
        self.waiters.validate()?;
        if !self.frontier.is_empty() || self.names.len() != self.capacity {
            return Err(Error::invariant("queue scratch or capacity corrupt"));
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
        for (id, entry) in self.entries.iter() {
            if *id != entry.request.request {
                return Err(Error::invariant("queue entry identity mismatch"));
            }
            expected.enqueue(entry.request.clone())?;
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
        if *self != expected {
            return Err(Error::invariant(
                "queue indexes disagree with lifecycle entries",
            ));
        }
        Ok(())
    }
}
