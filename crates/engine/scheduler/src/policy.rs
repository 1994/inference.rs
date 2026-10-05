//! One batch may mix incremental prefill/decode/forward for a compatible program.
use crate::FallbackCosts;
use infer_core::{DecisionId, Error, RequestId, Result, StepId};
use infer_ir::{
    CostEstimate, CostQuery, DeferReason, ExecutionRole, ReadyWork, ResourceSnapshot,
    SchedulerConfig, SchedulingDecision, SelectionReason,
};
use infer_spi::{CostModelProvider, SchedulingPolicy};

mod packing;
pub use packing::PackingWorkspace;
#[derive(Debug, Clone, Default)]
pub struct CostAwarePolicy;
///
/// # Errors
/// Returns an invalid-input error for zero or inconsistent scheduler limits and invalid urgency budgets.
pub fn validate_scheduler(c: &SchedulerConfig) -> Result<()> {
    if c.prefill_chunk_tokens == 0
        || c.forward_chunk_tokens == 0
        || c.fair_quantum_tokens == 0
        || c.fair_quantum_tokens > 128
        || c.max_wait_us == 0
        || c.urgent_budget_percent > 100
        || c.max_singleton_gpu_us == 0
        || c.max_planning_probes == 0
        || c.max_planning_probes > 65536
    {
        return Err(Error::invalid("invalid micro scheduler configuration"));
    }
    Ok(())
}
const fn hard_limit(cost: CostEstimate, r: &ResourceSnapshot) -> Option<(DeferReason, u64, u64)> {
    if cost.workspace_bytes > r.workspace_bytes {
        Some((
            DeferReason::Workspace,
            cost.workspace_bytes,
            r.workspace_bytes,
        ))
    } else if cost.logical_pages > r.free_logical_pages {
        Some((
            DeferReason::StateCapacity,
            cost.logical_pages as u64,
            r.free_logical_pages as u64,
        ))
    } else if cost.state_pages > r.free_state_pages {
        Some((
            DeferReason::StateCapacity,
            cost.state_pages as u64,
            r.free_state_pages as u64,
        ))
    } else if cost.state_bytes > r.free_state_bytes {
        Some((
            DeferReason::StateCapacity,
            cost.state_bytes,
            r.free_state_bytes,
        ))
    } else if cost.transfer_us > r.transfer_budget_us {
        Some((
            DeferReason::TransferBudget,
            cost.transfer_us,
            r.transfer_budget_us,
        ))
    } else if cost.encoder_us > r.encoder_budget_us {
        Some((
            DeferReason::EncoderBudget,
            cost.encoder_us,
            r.encoder_budget_us,
        ))
    } else {
        None
    }
}
fn query(item: &ReadyWork, count: usize) -> CostQuery {
    let mut q = item.cost_query;
    q.tokens = count;
    if q.role != ExecutionRole::Decode {
        q.context_tokens = q.context_tokens.saturating_add(count - 1);
    }
    q
}
fn target(item: &ReadyWork) -> Option<u64> {
    [
        (item.deadline_us, item.remaining_completion_us),
        (item.latency_deadline_us, item.remaining_latency_us),
    ]
    .into_iter()
    .filter_map(|(t, c)| t.map(|t| (t, c)))
    .min_by_key(|(t, c)| i128::from(*t) - i128::from(*c))
    .map(|(t, _)| t)
}
fn slack(item: &ReadyWork, now: u64) -> Option<i64> {
    [
        (item.deadline_us, item.remaining_completion_us),
        (item.latency_deadline_us, item.remaining_latency_us),
    ]
    .into_iter()
    .filter_map(|(t, c)| t.map(|t| i128::from(t) - i128::from(now) - i128::from(c)))
    .min()
    .map(|s| i64::try_from(s).unwrap_or(if s < 0 { i64::MIN } else { i64::MAX }))
}
#[must_use]
pub fn service_charge(us: u64, weight: u32) -> u64 {
    if weight == 0 {
        return u64::MAX;
    }
    u64::try_from((u128::from(us) * 1_000_000).div_ceil(u128::from(weight))).unwrap_or(u64::MAX)
}
#[derive(Clone, Copy)]
struct Priority {
    class: u8,
    slack: i64,
    finish: u64,
    role_bias: u8,
    request: RequestId,
    reason: SelectionReason,
}
impl Priority {
    const fn key(self) -> (u8, i64, u64, u8, RequestId) {
        (
            self.class,
            self.slack,
            self.finish,
            self.role_bias,
            self.request,
        )
    }
}
pub fn preemption_indices(
    ready: &[ReadyWork],
    resources: &ResourceSnapshot,
    now: u64,
    ordered: &mut Vec<usize>,
) {
    ordered.clear();
    ordered.extend(0..ready.len());
    ordered.sort_unstable_by_key(|index| {
        let item = &ready[*index];
        priority(
            item,
            item.virtual_finish,
            item.cost_per_token.gpu_us,
            now,
            resources,
            0,
        )
        .key()
    });
}
/// Use the same urgency and fairness ordering for memory-pressure decisions.
/// The runtime supplies only quiescent runnable requests; this function changes no state.
#[must_use]
pub fn preemption_order(
    ready: &[ReadyWork],
    resources: &ResourceSnapshot,
    now_us: u64,
) -> Vec<RequestId> {
    let mut ordered: Vec<_> = ready
        .iter()
        .map(|item| {
            (
                priority(
                    item,
                    item.virtual_finish,
                    item.cost_per_token.gpu_us,
                    now_us,
                    resources,
                    0,
                )
                .key(),
                item.request,
            )
        })
        .collect();
    ordered.sort_unstable_by_key(|(key, _)| *key);
    ordered.into_iter().map(|(_, request)| request).collect()
}
fn priority(
    item: &ReadyWork,
    tag: u64,
    cost: u64,
    now: u64,
    r: &ResourceSnapshot,
    urgent_used: u64,
) -> Priority {
    let wait = now.saturating_sub(item.last_service_us);
    let slack = slack(item, now);
    let urgent = slack.is_some_and(|s| s <= i64::try_from(r.gpu_budget_us).unwrap_or(i64::MAX))
        && (u128::from(urgent_used) * 100)
            < u128::from(r.gpu_budget_us) * u128::from(r.scheduler.urgent_budget_percent);
    let (class, reason) = if wait >= r.scheduler.max_wait_us {
        (0, SelectionReason::Aging)
    } else if urgent {
        (1, SelectionReason::UrgentSlo)
    } else if item.role == ExecutionRole::Decode {
        (2, SelectionReason::DecodeLatency)
    } else {
        (2, SelectionReason::WeightedFair)
    };
    Priority {
        class,
        slack: if urgent { slack.unwrap_or(0) } else { 0 },
        finish: tag.saturating_add(service_charge(cost, item.weight)),
        role_bias: u8::from(item.role != ExecutionRole::Decode),
        request: item.request,
        reason,
    }
}
impl SchedulingPolicy for CostAwarePolicy {
    type Workspace = PackingWorkspace;
    fn reserve_workspace(
        &self,
        workspace: &mut PackingWorkspace,
        requests: usize,
        batch: usize,
    ) -> Result<()> {
        workspace.reserve(requests, batch)
    }
    fn plan_reusing(
        &self,
        context: infer_spi::PlanningContext<'_>,
        workspace: &mut PackingWorkspace,
    ) -> Result<SchedulingDecision> {
        packing::BatchBuilder::new(
            context.ready,
            context.resources,
            context.now_us,
            context.costs,
            workspace,
        )?
        .pack(context.decision, context.step)
    }
    fn plan_into(
        &self,
        context: infer_spi::PlanningContext<'_>,
        workspace: &mut PackingWorkspace,
        output: &mut infer_spi::DecisionStorage,
    ) -> Result<SchedulingDecision> {
        packing::BatchBuilder::new(
            context.ready,
            context.resources,
            context.now_us,
            context.costs,
            workspace,
        )?
        .pack_into(context.decision, context.step, output)
    }
    fn identity(&self) -> &'static str {
        "slack-wfq-mixed-v3"
    }
    fn plan(
        &self,
        ready: &[ReadyWork],
        r: &ResourceSnapshot,
        now: u64,
        id: DecisionId,
        step: StepId,
    ) -> Result<SchedulingDecision> {
        self.plan_with_cost(ready, r, now, id, step, &FallbackCosts)
    }

    fn plan_with_cost(
        &self,
        ready: &[ReadyWork],
        r: &ResourceSnapshot,
        now: u64,
        id: DecisionId,
        step: StepId,
        costs: &dyn CostModelProvider,
    ) -> Result<SchedulingDecision> {
        self.plan_reusing(
            infer_spi::PlanningContext {
                ready,
                resources: r,
                now_us: now,
                decision: id,
                step,
                costs,
            },
            &mut PackingWorkspace::default(),
        )
    }
}
