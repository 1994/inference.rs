//! Pure recompute selection; the runtime performs resets only after inspecting this plan.
use crate::any_feasible;
use infer_core::{Error, RequestId, Result};
use infer_ir::{ExecutionRole, ReadyWork, ResourceSnapshot};
use infer_spi::CostModelProvider;
use std::collections::BTreeSet;

pub struct RecomputePlan {
    pub focus: RequestId,
    pub victims: Vec<RequestId>,
}
/// # Errors
/// Rejects a stale focus or a failed cost estimate. No runtime/device state is mutated.
pub fn recompute_plan(
    ready: &[ReadyWork],
    resources: &ResourceSnapshot,
    now: u64,
    existing: Option<RequestId>,
    reclaimable: &BTreeSet<RequestId>,
    costs: &dyn CostModelProvider,
) -> Result<Option<RecomputePlan>> {
    let mut workspace = PreemptionWorkspace::new(ready.len())?;
    let focus = recompute_plan_into(
        ready,
        resources,
        now,
        existing,
        |id| reclaimable.contains(&id),
        costs,
        &mut workspace,
    )?;
    Ok(focus.map(|focus| RecomputePlan {
        focus,
        victims: workspace.victims,
    }))
}
/// Fixed candidate order and victim storage shared by all pressure decisions.
pub struct PreemptionWorkspace {
    ordered: Vec<usize>,
    pub victims: Vec<RequestId>,
}
impl PreemptionWorkspace {
    /// # Errors
    /// Reports failure to reserve the candidate limit on startup.
    pub fn new(capacity: usize) -> Result<Self> {
        let (mut ordered, mut victims) = (Vec::new(), Vec::new());
        ordered
            .try_reserve_exact(capacity)
            .map_err(|error| Error::invalid(error.to_string()))?;
        victims
            .try_reserve_exact(capacity)
            .map_err(|error| Error::invalid(error.to_string()))?;
        Ok(Self { ordered, victims })
    }
}
/// # Errors
/// Rejects stale focus, insufficient prepared scratch or failed estimates; changes no runtime state.
pub fn recompute_plan_into(
    ready: &[ReadyWork],
    resources: &ResourceSnapshot,
    now: u64,
    existing: Option<RequestId>,
    reclaimable: impl Fn(RequestId) -> bool,
    costs: &dyn CostModelProvider,
    workspace: &mut PreemptionWorkspace,
) -> Result<Option<RequestId>> {
    workspace.victims.clear();
    if ready.len() > workspace.ordered.capacity() || ready.len() > workspace.victims.capacity() {
        return Err(Error::invariant("preemption candidate capacity exceeded"));
    }
    crate::policy::preemption_indices(ready, resources, now, &mut workspace.ordered);
    let Some(focus) =
        existing.or_else(|| workspace.ordered.first().map(|index| ready[*index].request))
    else {
        return Ok(None);
    };
    let item = ready
        .iter()
        .find(|item| item.request == focus)
        .ok_or_else(|| Error::invariant("preemption focus is not ready"))?;
    let cost = costs.estimate(&[item.cost_query])?;
    let memory_blocked = cost.state_pages > resources.free_state_pages
        || cost.logical_pages > resources.free_logical_pages
        || cost.state_bytes > resources.free_state_bytes;
    let execution_blocked = [
        (cost.gpu_us, resources.scheduler.max_singleton_gpu_us),
        (cost.workspace_bytes, resources.workspace_bytes),
        (cost.transfer_us, resources.transfer_budget_us),
        (cost.encoder_us, resources.encoder_budget_us),
    ]
    .into_iter()
    .any(|(required, available)| required > available);
    if !memory_blocked || execution_blocked {
        return Ok(None);
    }
    let aged = now.saturating_sub(item.last_service_us) >= resources.scheduler.max_wait_us;
    let urgent = [item.deadline_us, item.latency_deadline_us]
        .into_iter()
        .flatten()
        .any(|deadline| deadline <= now.saturating_add(resources.gpu_budget_us));
    if existing.is_none()
        && !aged
        && !urgent
        && item.role != ExecutionRole::Decode
        && any_feasible(ready, resources, costs)?
    {
        return Ok(None);
    }
    workspace.victims.extend(
        workspace
            .ordered
            .iter()
            .rev()
            .map(|index| ready[*index].request)
            .filter(|id| *id != focus && reclaimable(*id)),
    );
    Ok(Some(focus))
}
