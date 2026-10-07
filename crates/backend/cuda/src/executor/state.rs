use super::{CudaBackend, Sequence};
use infer_core::{Error, ErrorCode, Result, StateId};
use infer_ir::OutputReadout;
impl CudaBackend {
    pub(super) fn available(&self) -> Result<u64> {
        let used = self.active_budget()?;
        // Preserve a GiB for external display allocations, modules and driver bookkeeping.
        Ok(self.budget.saturating_sub(used).min(
            self.loaded
                .device()
                .memory_info()?
                .0
                .saturating_sub(self.loaded.profile().device_headroom_bytes()),
        ))
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
        let budget = self.loaded.sequence_budget(capacity, readout)?;
        if self.states.len() >= self.maximum_states
            || self
                .active_budget()?
                .checked_add(budget)
                .is_none_or(|n| n > self.budget)
        {
            return Err(Error::new(
                ErrorCode::Capacity,
                "CUDA resident state admission budget exhausted",
            ));
        }
        if let Some(mut state) = self.pool.take(capacity, readout) {
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
            state.history.clear();
            state.hidden.clear();
            state.poisoned = false;
            self.pool.reuses = self.pool.reuses.saturating_add(1);
            self.states.insert(id, state);
            return Ok(());
        }
        self.make_room(budget)?;
        let program = match self.loaded.sequence(capacity) {
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
            },
        );
        Ok(())
    }
    pub(super) fn reset(&mut self, id: StateId) -> Result<()> {
        self.idle()?;
        let state = self
            .states
            .get_mut(&id)
            .ok_or_else(|| Error::invalid("unknown CUDA state"))?;
        state.poisoned = true;
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
