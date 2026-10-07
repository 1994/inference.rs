//! Bounded idle graph/state retention. Live request IDs never enter this cache.
use super::{CudaBackend, Sequence};
use infer_core::{Error, Result};
use infer_ir::OutputReadout;
use std::collections::VecDeque;

#[derive(Default)]
pub(super) struct StatePool {
    entries: VecDeque<Sequence>,
    bytes: u64,
    pub(super) reclaimable_bytes: u64,
    pub allocations: u64,
    pub reuses: u64,
    evictions: u64,
}

#[derive(Debug, serde::Serialize)]
pub struct PoolInspection {
    pub active_sequences: usize,
    pub cached_sequences: usize,
    pub cached_admission_bytes: u64,
    pub sequence_allocations: u64,
    pub sequence_reuses: u64,
    pub sequence_evictions: u64,
}

impl Sequence {
    fn reclaimable_bytes(&self) -> u64 {
        self.program.reclaimable_bytes
            + self
                .speculation
                .as_ref()
                .map_or(0, |s| s.program.reclaimable_bytes)
    }
}

impl StatePool {
    pub fn take(&mut self, capacity: usize, readout: OutputReadout) -> Option<Sequence> {
        let index = self.entries.iter().position(|s| {
            !s.poisoned
                && s.capacity.next_power_of_two() == capacity.next_power_of_two()
                && s.readout == readout
        })?;
        let sequence = self.entries.remove(index)?;
        self.bytes -= sequence.budget;
        self.reclaimable_bytes -= sequence.reclaimable_bytes();
        Some(sequence)
    }
    pub fn retain(&mut self, sequence: Sequence) {
        self.bytes += sequence.budget;
        self.reclaimable_bytes += sequence.reclaimable_bytes();
        self.entries.push_back(sequence);
    }
    pub fn evict(&mut self) -> bool {
        let Some(sequence) = self.entries.pop_front() else {
            return false;
        };
        self.bytes -= sequence.budget;
        self.reclaimable_bytes -= sequence.reclaimable_bytes();
        self.evictions = self.evictions.saturating_add(1);
        drop(sequence);
        true
    }
}

impl CudaBackend {
    #[must_use]
    pub fn pool_inspection(&self) -> PoolInspection {
        PoolInspection {
            active_sequences: self.states.len(),
            cached_sequences: self.pool.entries.len(),
            cached_admission_bytes: self.pool.bytes,
            sequence_allocations: self.pool.allocations,
            sequence_reuses: self.pool.reuses,
            sequence_evictions: self.pool.evictions,
        }
    }
    pub(super) fn active_budget(&self) -> Result<u64> {
        self.states.values().try_fold(0_u64, |sum, s| {
            sum.checked_add(s.budget)
                .ok_or_else(|| Error::invariant("CUDA reservation accounting overflow"))
        })
    }
    pub(super) fn make_room(&mut self, needed: u64) -> Result<()> {
        let active = self.active_budget()?;
        while active
            .checked_add(self.pool.bytes)
            .and_then(|n| n.checked_add(self.prefix.bytes()))
            .and_then(|n| n.checked_add(needed))
            .is_none_or(|n| n > self.budget)
            || self.states.len() + self.pool.entries.len() >= self.maximum_states
            || needed > self.physical_available()?
        {
            if !self.pool.evict() && !self.prefix.evict() {
                return Err(Error::new(
                    infer_core::ErrorCode::Capacity,
                    format!(
                        "CUDA resident state admission budget exhausted: needed={needed}, active={active}, cached={}, prefix={}, physical={}, budget={}",
                        self.pool.bytes,
                        self.prefix.bytes(),
                        self.physical_available()?,
                        self.budget
                    ),
                ));
            }
            self.loaded.device().reclaim_barrier()?;
        }
        Ok(())
    }
    /// Release only cached graphs. Live states and completion tickets remain owned.
    /// # Errors
    /// Rejects an in-flight completion or a failed device drain.
    pub fn trim_state_pool(&mut self) -> Result<()> {
        self.idle()?;
        self.loaded.device().drain()?;
        while self.pool.evict() {}
        self.loaded.device().reclaim_barrier()
    }
}
