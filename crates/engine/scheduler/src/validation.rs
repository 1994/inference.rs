//! Validate scheduler decisions against immutable resource and cost snapshots.
use infer_core::{Error, Result};
use infer_ir::{
    CostQuery, DeferReason, ExecutionRole, ReadyWork, ResourceSnapshot, SchedulingDecision,
    StepPlan,
};
use infer_spi::CostModelProvider;

/// Caller-owned bounded validation storage. Startup allocation is separate from owner polling.
pub struct ValidationScratch {
    queries: Vec<CostQuery>,
    evidence: Vec<u8>,
    ready: Vec<(infer_core::RequestId, usize)>,
    work_index: Vec<usize>,
}
impl ValidationScratch {
    /// # Errors
    /// Returns capacity errors if fixed validation storage cannot be allocated.
    pub fn new(candidates: usize, batch: usize) -> Result<Self> {
        let mut queries = Vec::new();
        let mut evidence = Vec::new();
        queries
            .try_reserve_exact(batch)
            .map_err(|e| Error::new(infer_core::ErrorCode::Capacity, e.to_string()))?;
        evidence
            .try_reserve_exact(candidates)
            .map_err(|e| Error::new(infer_core::ErrorCode::Capacity, e.to_string()))?;
        let mut ready = Vec::new();
        let mut work_index = Vec::new();
        ready
            .try_reserve_exact(candidates)
            .map_err(|e| Error::new(infer_core::ErrorCode::Capacity, e.to_string()))?;
        work_index
            .try_reserve_exact(candidates)
            .map_err(|e| Error::new(infer_core::ErrorCode::Capacity, e.to_string()))?;
        Ok(Self {
            queries,
            evidence,
            ready,
            work_index,
        })
    }
    fn prepare(&mut self, ready: &[ReadyWork]) -> Result<()> {
        if ready.len() > self.ready.capacity() {
            return Err(Error::invariant("validation candidate capacity exceeded"));
        }
        self.ready.clear();
        self.ready.extend(
            ready
                .iter()
                .enumerate()
                .map(|(index, row)| (row.request, index)),
        );
        self.ready.sort_unstable();
        if self.ready.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(Error::invariant("duplicate validation candidate"));
        }
        self.work_index.clear();
        self.work_index.resize(ready.len(), usize::MAX);
        self.evidence.clear();
        self.evidence.resize(ready.len(), 0);
        self.queries.clear();
        Ok(())
    }
    fn position(&self, request: infer_core::RequestId) -> Result<usize> {
        self.ready
            .binary_search_by_key(&request, |pair| pair.0)
            .map(|index| self.ready[index].1)
            .map_err(|_| Error::invariant("scheduler evidence references unknown ready request"))
    }
}
/// Fill preallocated cost queries without allocating on the scheduling owner.
/// # Errors
/// Rejects unknown ready work or insufficient caller storage.
pub fn step_cost_queries_into(
    step: &StepPlan,
    ready: &[ReadyWork],
    queries: &mut Vec<CostQuery>,
) -> Result<()> {
    queries.clear();
    if step.work.len() > queries.capacity() {
        return Err(Error::invariant("cost query scratch capacity exceeded"));
    }
    for work in &step.work {
        let item = ready
            .iter()
            .find(|r| r.request == work.request)
            .ok_or_else(|| Error::invariant("non-runnable selected"))?;
        queries.push(work_query(item, work.token_count));
    }
    Ok(())
}
fn work_query(item: &ReadyWork, tokens: usize) -> CostQuery {
    let mut query = item.cost_query;
    query.tokens = tokens;
    if item.role != ExecutionRole::Decode {
        query.context_tokens = query
            .context_tokens
            .saturating_add(tokens.saturating_sub(1));
    }
    query
}

/// # Errors
/// Rejects selection of an unknown ready request.
pub fn step_cost_queries(step: &StepPlan, ready: &[ReadyWork]) -> Result<Vec<CostQuery>> {
    let mut queries = Vec::with_capacity(step.work.len());
    step_cost_queries_into(step, ready, &mut queries)?;
    Ok(queries)
}

/// # Errors
/// Returns cost-provider errors.
pub fn any_feasible(
    ready: &[ReadyWork],
    r: &ResourceSnapshot,
    costs: &dyn CostModelProvider,
) -> Result<bool> {
    for item in ready {
        let c = costs.estimate(&[item.cost_query])?;
        if c.gpu_us <= r.scheduler.max_singleton_gpu_us
            && c.workspace_bytes <= r.workspace_bytes
            && c.num_gpu_blocks <= r.free_state_pages
            && c.logical_pages <= r.free_logical_pages
            && c.state_bytes <= r.free_state_bytes
            && c.transfer_us <= r.transfer_budget_us
            && c.encoder_us <= r.encoder_budget_us
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// # Errors
/// Rejects invalid identities, cursors, resource budgets or predicted costs.
pub fn validate_step(
    step: &StepPlan,
    ready: &[ReadyWork],
    r: &ResourceSnapshot,
    program: infer_core::ProgramId,
    costs: &dyn CostModelProvider,
) -> Result<()> {
    let mut scratch = ValidationScratch::new(ready.len(), step.work.len())?;
    validate_step_reusing(step, ready, r, program, costs, &mut scratch)
}
/// # Errors
/// Rejects invalid work and forecasts using fixed caller storage.
pub fn validate_step_reusing(
    step: &StepPlan,
    ready: &[ReadyWork],
    r: &ResourceSnapshot,
    program: infer_core::ProgramId,
    costs: &dyn CostModelProvider,
    scratch: &mut ValidationScratch,
) -> Result<()> {
    scratch.prepare(ready)?;
    validate_prepared_step(step, ready, r, program, costs, scratch)
}
fn validate_prepared_step(
    step: &StepPlan,
    ready: &[ReadyWork],
    r: &ResourceSnapshot,
    program: infer_core::ProgramId,
    costs: &dyn CostModelProvider,
    scratch: &mut ValidationScratch,
) -> Result<()> {
    if step.work.is_empty()
        || step.work.len() > scratch.queries.capacity()
        || step.work.len() > r.max_num_seqs
        || step.program != program
        || step.graph.is_some()
    {
        return Err(Error::invariant("invalid scheduler step/program/graph"));
    }
    let mut tokens = 0usize;
    for (index, work) in step.work.iter().enumerate() {
        let position = scratch.position(work.request)?;
        let item = &ready[position];
        let cap = match item.role {
            ExecutionRole::Decode => 1,
            ExecutionRole::Prefill => r.scheduler.prefill_chunk_tokens,
            ExecutionRole::Forward => r.scheduler.forward_chunk_tokens,
            ExecutionRole::Mixed => 0,
        };
        if scratch.work_index[position] != usize::MAX
            || work.state != item.state
            || work.role != item.role
            || work.token_count == 0
            || work.token_count > item.remaining_tokens.min(cap)
        {
            return Err(Error::invariant("invalid scheduler work/cursor/role/chunk"));
        }
        scratch.work_index[position] = index;
        scratch.queries.push(work_query(item, work.token_count));
        tokens = tokens
            .checked_add(work.token_count)
            .ok_or_else(|| Error::invariant("scheduler token overflow"))?;
    }
    let role = step.work[0].role;
    let expected_role = if step.work.iter().all(|w| w.role == role) {
        role
    } else {
        ExecutionRole::Mixed
    };
    let predicted = costs.estimate(&scratch.queries)?;
    let allowed_overrun = step.quantum_overrun
        && step.work.len() == 1
        && tokens == 1
        && predicted.gpu_us <= r.scheduler.max_singleton_gpu_us
        && predicted.gpu_us > r.gpu_budget_us;
    if tokens > r.max_num_batched_tokens
        || step.role != expected_role
        || step.cost != predicted
        || (step.quantum_overrun && !allowed_overrun)
        || (predicted.gpu_us > r.gpu_budget_us && !allowed_overrun)
        || predicted.workspace_bytes > r.workspace_bytes
        || predicted.num_gpu_blocks > r.free_state_pages
        || predicted.logical_pages > r.free_logical_pages
        || predicted.state_bytes > r.free_state_bytes
        || predicted.transfer_us > r.transfer_budget_us
        || predicted.encoder_us > r.encoder_budget_us
    {
        return Err(Error::invariant(
            "scheduler forecast or resource budget mismatch",
        ));
    }
    Ok(())
}

/// # Errors
/// Rejects incomplete, forged or overlapping decision evidence.
pub fn validate_decision(
    decision: &SchedulingDecision,
    ready: &[ReadyWork],
    resources: &ResourceSnapshot,
    program: infer_core::ProgramId,
    costs: &dyn CostModelProvider,
) -> Result<()> {
    let mut scratch = ValidationScratch::new(ready.len(), resources.max_num_seqs)?;
    validate_decision_reusing(decision, ready, resources, program, costs, &mut scratch)
}
/// # Errors
/// Rejects incomplete, forged or overlapping evidence without temporary trees or vectors.
pub fn validate_decision_reusing(
    decision: &SchedulingDecision,
    ready: &[ReadyWork],
    resources: &ResourceSnapshot,
    program: infer_core::ProgramId,
    costs: &dyn CostModelProvider,
    scratch: &mut ValidationScratch,
) -> Result<()> {
    if ready.len() > scratch.evidence.capacity() {
        return Err(Error::invariant(
            "decision evidence scratch capacity exceeded",
        ));
    }
    scratch.prepare(ready)?;
    if let Some(step) = &decision.step {
        validate_prepared_step(step, ready, resources, program, costs, scratch)?;
    }
    for d in &decision.deferred {
        let index = scratch.position(d.request)?;
        let item = &ready[index];
        if matches!(d.reason, DeferReason::PreemptionFocus { .. })
            || scratch.evidence[index] != 0
            || scratch.work_index[index] != usize::MAX
            || (d.reason == DeferReason::AtomicCostLimit
                && costs.estimate(&[item.cost_query])?.gpu_us
                    <= resources.scheduler.max_singleton_gpu_us)
        {
            return Err(Error::invariant(
                "scheduler forged rejection or selected/deferred overlap",
            ));
        }
        scratch.evidence[index] = 1;
    }
    for selected in &decision.selected {
        let index = scratch.position(selected.request)?;
        if scratch.evidence[index] != 0
            || !decision
                .step
                .as_ref()
                .and_then(|step| step.work.get(scratch.work_index[index]))
                .is_some_and(|work| {
                    let tokens_match = work.token_count == selected.tokens;
                    work.role == selected.role && tokens_match
                })
        {
            return Err(Error::invariant(
                "selection evidence differs from planned work",
            ));
        }
        scratch.evidence[index] = 2;
    }
    if decision.step.as_ref().map_or(0, |step| step.work.len()) != decision.selected.len()
        || scratch.evidence.contains(&0)
    {
        return Err(Error::invariant(
            "scheduler must provide one selection or deferral for every runnable request",
        ));
    }
    Ok(())
}
