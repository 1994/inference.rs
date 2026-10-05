//! Reusable owned decision buffers; immutable readers keep their own bounded storage.
use infer_core::{Error, ErrorCode, Result};
use infer_ir::{DeferredWork, PlannedWork, SchedulingDecision, SelectionEvidence};

#[derive(Default)]
pub struct DecisionStorage {
    pub work: Vec<PlannedWork>,
    pub selected: Vec<SelectionEvidence>,
    pub deferred: Vec<DeferredWork>,
}
impl DecisionStorage {
    /// # Errors
    /// Reports inability to reserve the startup candidate and batch limits.
    pub fn new(candidates: usize, batch: usize) -> Result<Self> {
        fn buffer<T>(limit: usize) -> Result<Vec<T>> {
            let mut storage = Vec::new();
            storage
                .try_reserve_exact(limit)
                .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
            Ok(storage)
        }
        Ok(Self {
            work: buffer(batch)?,
            selected: buffer(batch)?,
            deferred: buffer(candidates)?,
        })
    }
    /// Reclaim only after publication/dispatch has finished borrowing this decision.
    pub fn reclaim(&mut self, mut decision: SchedulingDecision) {
        if let Some(step) = decision.step.as_mut() {
            step.work.clear();
            if step.work.capacity() >= self.work.capacity() {
                self.work = std::mem::take(&mut step.work);
            }
        }
        decision.selected.clear();
        decision.deferred.clear();
        self.selected = decision.selected;
        self.deferred = decision.deferred;
    }
}
