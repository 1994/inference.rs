use super::{CudaBackend, Sequence};
use infer_core::{Error, ErrorCode, Result, StateId};
use infer_ir::OutputReadout;
impl CudaBackend {
    pub(super) fn available(&self) -> Result<u64> {
        let used = self.active_budget()?;
        // Idle programs and prefix snapshots are reclaimable. Omitting them makes
        // repeated requests fail admission even when their exact state is already cached.
        let reclaimable = self
            .pool
            .reclaimable_bytes
            .saturating_add(self.prefix.bytes());
        Ok(self
            .budget
            .saturating_sub(used)
            .min(self.physical_available()?.saturating_add(reclaimable)))
    }
    pub(super) fn physical_available(&self) -> Result<u64> {
        Ok(self
            .loaded
            .device()
            .memory_info()?
            .0
            .saturating_sub(self.loaded.profile().device_headroom_bytes()))
    }
    /// Prefill width the state's captured program will use.
    ///
    /// A pooled state must keep its wide prefill path: silently downgrading chunked prefill to
    /// the verification width costs several times the prefill latency on every reused request.
    /// # Errors
    /// Rejects unknown state ids.
    pub fn prefill_width_for(&self, state: StateId) -> Result<usize> {
        self.states
            .get(&state)
            .map(|sequence| sequence.program.prefill_width())
            .ok_or_else(|| Error::invalid("unknown CUDA state"))
    }

    /// Whether a sequence must carry admission for its own verification graph. A sequence
    /// that fits the slot pool verifies through the pool's shared graph instead — but only
    /// up to the pool's width: past it, a resident sequence can lose the slot race and fall
    /// back to serial execution, which lazily recaptures the private graph and must have
    /// been charged for it.
    pub(super) fn private_verification_for(&self, capacity: usize, readout: OutputReadout) -> bool {
        let Some(pool) = &self.slots else {
            return true;
        };
        if self.loaded.mtp_depth() == 0
            || readout != OutputReadout::Logits
            || capacity > pool.capacity()
            || pool.verify().is_none()
        {
            return true;
        }
        let pooled = self
            .states
            .values()
            .filter(|state| {
                state.speculation.is_some()
                    && state.readout == OutputReadout::Logits
                    && state.capacity <= pool.capacity()
            })
            .count();
        pooled >= pool.width()
    }

    pub(super) fn reserve(
        &mut self,
        id: StateId,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        self.idle()?;
        if self.states.contains_key(&id) {
            return Err(Error::new(
                ErrorCode::Conflict,
                "CUDA state already reserved",
            ));
        }
        let private_verification = self.private_verification_for(capacity, readout);
        let budget = self
            .loaded
            .sequence_budget(capacity, readout, private_verification)?;
        if self.states.len() >= self.maximum_states
            || self
                .active_budget()?
                .checked_add(budget)
                .is_none_or(|n| n > self.budget)
        {
            return Err(Error::new(
                ErrorCode::Capacity,
                format!(
                    "CUDA resident state admission budget exhausted: needed={budget}, active={}, budget={}, states={}, maximum={}",
                    self.active_budget()?,
                    self.budget,
                    self.states.len(),
                    self.maximum_states
                ),
            ));
        }
        if let Some(state) = self.pool.take(capacity, readout) {
            return self.reuse_pooled(id, state, capacity, budget, private_verification);
        }
        self.make_room(budget)?;
        let program = match if private_verification {
            self.loaded.sequence(capacity)
        } else {
            self.loaded.sequence_pooled(capacity)
        } {
            Ok(program) => program,
            Err(error) => {
                if let Err(drain_error) = self.loaded.device().reclaim_barrier() {
                    self.fatal = Some(drain_error);
                }
                return Err(error);
            }
        };
        let speculation = match self.loaded.draft(capacity) {
            Ok(draft) => {
                draft.map(|program| super::Speculation::new(program, self.loaded.mtp_depth()))
            }
            Err(error) => {
                if let Err(drain_error) = self.loaded.device().reclaim_barrier() {
                    self.fatal = Some(drain_error);
                }
                return Err(error);
            }
        };
        self.pool.allocations = self.pool.allocations.saturating_add(1);
        self.states.insert(
            id,
            Sequence {
                program,
                speculation,
                capacity,
                readout,
                budget,
                history: Vec::new(),
                hidden: Vec::new(),
                poisoned: false,
                slot: None,
                verification_priced: private_verification,
            },
        );
        Ok(())
    }

    /// Re-admit a pooled sequence shell under a freshly charged budget: reset program and
    /// speculation state, realign verification pricing with the new charge, and clear the
    /// shell's poison flag. A reset failure poisons the shell back into the pool; when even
    /// the drain after it fails, the backend records the drain as fatal.
    fn reuse_pooled(
        &mut self,
        id: StateId,
        mut state: Sequence,
        capacity: usize,
        budget: u64,
        private_verification: bool,
    ) -> Result<()> {
        state.poisoned = true;
        let reset = state.program.reset().and_then(|()| {
            state
                .speculation
                .as_mut()
                .map_or(Ok(()), super::Speculation::reset)
        });
        if let Err(error) = reset {
            self.pool.retain(state);
            if let Err(drain_error) = self.loaded.device().drain() {
                self.fatal = Some(drain_error);
            }
            return Err(error);
        }
        if state.slot.is_some() {
            return Err(Error::invariant("pooled CUDA state holds a slot lease"));
        }
        state.capacity = capacity;
        // The reused program's verification storage must match the price just charged:
        // a serial-priced program may still hold one (a later slot lease credits its
        // release), while the pooled price never included one.
        if !private_verification {
            state.program.forfeit_verification();
        }
        state.verification_priced = private_verification;
        state.budget = budget;
        state.history.clear();
        state.hidden.clear();
        state.poisoned = false;
        self.pool.reuses = self.pool.reuses.saturating_add(1);
        self.states.insert(id, state);
        Ok(())
    }
    pub(super) fn reset(&mut self, id: StateId) -> Result<()> {
        self.idle()?;
        let restore = self
            .states
            .get(&id)
            .filter(|s| s.slot.is_some() && s.verification_priced)
            .map_or(0, |s| s.program.released_verification_bytes());
        if restore > 0 {
            self.make_room(restore)?;
        }
        let state = self
            .states
            .get_mut(&id)
            .ok_or_else(|| Error::invalid("unknown CUDA state"))?;
        state.poisoned = true;
        state.budget = state
            .budget
            .checked_add(restore)
            .ok_or_else(|| Error::invariant("private verification reservation overflow"))?;
        state.program.reset()?;
        if let Some(speculation) = state.speculation.as_mut() {
            speculation.reset()?;
        }
        // The slot keeps no rollback state: the next bind overwrites its covered prefix.
        if let Some(lease) = state.slot.take()
            && let Some(pool) = self.slots.as_mut()
        {
            pool.release(lease.slot);
        }
        state.history.clear();
        state.hidden.clear();
        state.poisoned = false;
        Ok(())
    }
    pub(super) fn release(&mut self, id: StateId) -> Result<()> {
        self.idle()?;
        self.loaded.device().drain()?;
        let mut state = self
            .states
            .remove(&id)
            .ok_or_else(|| Error::invalid("unknown CUDA state"))?;
        if let Some(lease) = state.slot.take()
            && let Some(pool) = self.slots.as_mut()
        {
            pool.release(lease.slot);
        }
        if !state.poisoned {
            state.history.clear();
            state.hidden.clear();
            self.pool.retain(state);
        }
        Ok(())
    }
    pub(super) fn ownership(&self, states: &[(StateId, usize, usize)]) -> Result<()> {
        if states.len() != self.states.len()
            || states
                .iter()
                .map(|s| s.0)
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != states.len()
            || states.iter().any(|(id, capacity, cursor)| {
                !self.states.get(id).is_some_and(|s| {
                    !s.poisoned
                        && s.capacity == *capacity
                        && s.history.len() == *cursor
                        && self.cursor_matches(s, *cursor)
                })
            })
        {
            return Err(Error::invariant("CUDA state ownership mismatch"));
        }
        Ok(())
    }

    /// Device cursor agreement: slot-leased sequences track the pool cursor, unbound
    /// sequences track their own program's position.
    fn cursor_matches(&self, state: &Sequence, cursor: usize) -> bool {
        state.slot.as_ref().map_or_else(
            || state.program.position() == cursor,
            |lease| {
                lease.position == cursor
                    && self
                        .slots
                        .as_ref()
                        .is_some_and(|pool| pool.cursors().get(lease.slot) == Some(&cursor))
            },
        )
    }
}
