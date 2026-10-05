//! Archive buffers retain capacities across empty decisions and first publication.
use infer_core::{DecisionId, Error, ProgramId, Result, StepId};
use infer_ir::{CostEstimate, ExecutionRole, SchedulingDecision, StepPlan};

pub struct HistoryStorage {
    spare: Vec<SchedulingDecision>,
    steps: Vec<StepPlan>,
}
impl HistoryStorage {
    pub fn new(
        capacity: usize,
        candidates: usize,
        batch: usize,
        program: ProgramId,
    ) -> Result<Self> {
        let mut spare = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            let storage = infer_spi::DecisionStorage::new(candidates, batch)?;
            spare.push(SchedulingDecision {
                window: None,
                id: DecisionId::ONE,
                deferred: storage.deferred,
                selected: storage.selected,
                step: Some(StepPlan {
                    id: StepId::ONE,
                    decision: DecisionId::ONE,
                    program,
                    role: ExecutionRole::Prefill,
                    work: storage.work,
                    cost: CostEstimate::default(),
                    graph: None,
                    quantum_overrun: false,
                }),
            });
        }
        Ok(Self {
            spare,
            steps: Vec::with_capacity(capacity),
        })
    }
    pub fn record(
        &mut self,
        old: Option<SchedulingDecision>,
        source: &SchedulingDecision,
    ) -> Result<SchedulingDecision> {
        let mut record = old
            .or_else(|| self.spare.pop())
            .ok_or_else(|| Error::invariant("archive decision credit missing"))?;
        if source.step.is_none()
            && let Some(step) = record.step.take()
        {
            self.steps.push(step);
        }
        if source.step.is_some() && record.step.is_none() {
            record.step = Some(
                self.steps
                    .pop()
                    .ok_or_else(|| Error::invariant("archive step credit missing"))?,
            );
        }
        record.clone_from(source);
        Ok(record)
    }
}

impl HistoryStorage {
    pub fn prepare_archive(
        &mut self,
        archive: &mut std::collections::VecDeque<SchedulingDecision>,
        capacity: usize,
    ) -> Result<()> {
        archive.reserve(capacity.saturating_sub(archive.len()));
        for record in archive {
            *record = self.record(None, record)?;
        }
        Ok(())
    }
}
