//! Scheduling extension contract.
use super::{CostModelProvider, DecisionStorage};
use infer_core::{Result, StepId};
use infer_ir::{
    AdmissionDecision, AdmissionInput, ReadyWork, ResourceSnapshot, SchedulingDecision,
};

/// Immutable planning inputs; reusable storage belongs to the engine rather than the policy.
pub struct PlanningContext<'a> {
    pub ready: &'a [ReadyWork],
    pub resources: &'a ResourceSnapshot,
    pub now_us: u64,
    pub decision: infer_core::DecisionId,
    pub step: StepId,
    pub costs: &'a dyn CostModelProvider,
}
pub trait SchedulingPolicy {
    type Workspace: Default;
    /// Reserve engine-owned planner storage before request admission.
    /// # Errors
    /// Rejects invalid limits or allocation failure; custom policies may retain empty storage.
    fn reserve_workspace(
        &self,
        _workspace: &mut Self::Workspace,
        _requests: usize,
        _batch: usize,
    ) -> Result<()> {
        Ok(())
    }
    /// Plan using persistent, engine-owned scratch storage.
    /// # Errors
    /// Returns incompatible work, resource or cost-estimation errors.
    fn plan_reusing(
        &self,
        context: PlanningContext<'_>,
        _workspace: &mut Self::Workspace,
    ) -> Result<SchedulingDecision> {
        self.plan_with_cost(
            context.ready,
            context.resources,
            context.now_us,
            context.decision,
            context.step,
            context.costs,
        )
    }
    /// Fill reusable output storage. The compatibility adapter may allocate owned decisions.
    /// # Errors
    /// Returns planning, resource or provider errors without publishing device work.
    fn plan_into(
        &self,
        context: PlanningContext<'_>,
        workspace: &mut Self::Workspace,
        _output: &mut DecisionStorage,
    ) -> Result<SchedulingDecision> {
        self.plan_reusing(context, workspace)
    }
    fn identity(&self) -> &str {
        std::any::type_name::<Self>()
    }
    ///
    /// # Errors
    /// Returns an invalid-input or unsupported error if the request, model, or backend cannot produce a valid plan.
    fn plan(
        &self,
        ready: &[ReadyWork],
        resources: &ResourceSnapshot,
        now_us: u64,
        decision: infer_core::DecisionId,
        step: StepId,
    ) -> Result<SchedulingDecision>;
    ///
    /// # Errors
    /// Returns a planning or cost-estimation error for incompatible work or resources.
    fn plan_with_cost(
        &self,
        ready: &[ReadyWork],
        resources: &ResourceSnapshot,
        now_us: u64,
        decision: infer_core::DecisionId,
        step: StepId,
        _costs: &dyn CostModelProvider,
    ) -> Result<SchedulingDecision> {
        self.plan(ready, resources, now_us, decision, step)
    }
}
pub trait AdmissionPolicy {
    fn identity(&self) -> &str {
        std::any::type_name::<Self>()
    }
    ///
    /// # Errors
    /// Returns an invalid-input or estimation error if admission inputs cannot be evaluated.
    fn check(&self, input: &AdmissionInput<'_>) -> Result<AdmissionDecision>;
}
