use crate::{Engine, ReplayAction};
use infer_core::{Error, ErrorCode, Result};
use infer_ir::{CostEstimate, CostObservation, CostQuery, ExecutionRole};
use infer_spi::{BackendProvider, CostModelProvider, SchedulingPolicy};

impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    /// Install a planning provider before any admission.
    /// # Errors
    /// Rejects provider changes after request or replay history exists.
    pub fn with_cost_model(
        mut self,
        costs: impl CostModelProvider + Send + Sync + 'static,
    ) -> Result<Self> {
        if !self.host.requests.is_empty() || !self.actions.is_empty() {
            return Err(Error::new(
                ErrorCode::Conflict,
                "install cost provider before admission",
            ));
        }
        self.costs = Box::new(costs);
        Ok(self)
    }
    pub(crate) fn enqueue_cost_observation(
        &mut self,
        observation: CostObservation,
        record: bool,
    ) -> Result<()> {
        if observation.work.is_empty()
            || observation.work.len() > self.config.max_num_seqs
            || observation.work.iter().any(|q| {
                q.program != self.program.id
                    || q.backend != self.program.backend
                    || q.tokens == 0
                    || q.tokens > self.model.max_sequence
                    || q.context_tokens == 0
                    || q.context_tokens > self.model.max_sequence
                    || q.fallback_per_token_us != self.config.cost_per_token_us
            })
            || observation.timing.elapsed_us == 0
            || !observation.timing.matches_backend(self.program.backend)
        {
            return Err(Error::invalid(
                "cost feedback does not match loaded program/budgets",
            ));
        }
        if self.pending_cost_observations.len() >= self.config.history_capacity {
            return Err(Error::new(ErrorCode::Capacity, "cost feedback queue full"));
        }
        if record {
            self.record_action(ReplayAction::CostFeedback(observation.clone()));
        }
        self.pending_cost_observations.push_back(observation);
        Ok(())
    }
    pub(crate) fn apply_cost_feedback(&mut self) -> Result<()> {
        for _ in 0..self.config.cpu.maintenance_items {
            let Some(observation) = self.pending_cost_observations.front() else {
                break;
            };
            self.costs.observe(observation)?;
            self.host.cost_epoch = self
                .host
                .cost_epoch
                .checked_add(1)
                .ok_or_else(|| Error::invariant("cost epoch exhausted"))?;
            self.pending_cost_observations.pop_front();
        }
        Ok(())
    }
    pub(crate) fn base_query(
        &self,
        role: ExecutionRole,
        tokens: usize,
        context: usize,
    ) -> CostQuery {
        CostQuery::from_unit(
            self.program.id,
            self.program.backend,
            role,
            tokens,
            context,
            CostEstimate {
                gpu_us: self.config.cost_per_token_us,
                workspace_bytes: self.program.workspace_bytes,
                ..Default::default()
            },
        )
    }
}
