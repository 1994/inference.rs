mod construction;
mod diagnostics;
mod history;
mod inspection;
mod invariants;
use crate::{RequestRecord, RuntimeConfig};
use infer_core::{
    Error, FinishReason, IdAllocator, ModelId, ProgramId, RequestId, StepId, event::EventReader,
    event::EventWriter,
};
use infer_ir::{
    BackendKind, CostModelInspection, CostObservation, CostQuery, ExecutionProgram,
    KvCacheInspection, ModelIr, SchedulingDecision, StepPlan, WorkloadOutput,
};
use infer_quality::{ProgressGuard, RequestMeasurement};
use infer_scheduler::CostAwarePolicy;
use infer_spi::{
    AdmissionPolicy, BackendProvider, CostModelProvider, SchedulingPolicy, WorkloadProvider,
};
use infer_state::{SequenceStateManager, StateSnapshot};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, collections::VecDeque};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletedRequest {
    pub request: RequestId,
    pub reason: FinishReason,
    pub output: Option<WorkloadOutput>,
    pub measurement: RequestMeasurement,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EngineOutput {
    Token {
        request: RequestId,
        token: u32,
        index: usize,
    },
    Finished(CompletedRequest),
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeInspection {
    pub state_recipe: Option<infer_ir::StateRecipe>,
    pub ready: bool,
    pub fault: Option<EngineFault>,
    pub resource_release_pending: bool,
    pub resource_waiters: usize,
    pub output_waiters: usize,
    pub output_batches: usize,
    pub backend_kind: BackendKind,
    pub weight_backed_dataflow: bool,
    pub backend: String,
    pub model: ModelId,
    pub program: ProgramId,
    pub active_requests: usize,
    pub completed_requests: usize,
    pub inflight_step: Option<StepId>,
    pub global_progress_epoch: u64,
    pub state: StateSnapshot,
    pub dropped_events: u64,
    pub dropped_actions: u64,
    pub cost_model: CostModelInspection,
    pub pending_cost_observations: usize,
    pub preemptions: u64,
    pub kv_cache: Option<KvCacheInspection>,
    /// Effective scheduler chunk budgets; a benchmark can prove the logical prompt chunk
    /// instead of inferring it from the captured graph width.
    #[serde(default)]
    pub scheduler: infer_ir::SchedulerConfig,
    /// Effective load-time graph geometry and speculation depth.
    #[serde(default)]
    pub execution_profile: Option<infer_ir::ExecutionProfileInspection>,
    pub queues: infer_scheduler::QueueInspection,
    pub cpu: crate::CpuRuntimeInspection,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineFault {
    pub error: Error,
    pub since_us: u64,
}
pub struct InFlight<T> {
    pub step: std::sync::Arc<StepPlan>,
    pub ticket: T,
    pub submitted_us: u64,
    pub launch_acked: bool,
    pub cost_queries: infer_ir::SharedOutput<CostQuery>,
    pub(crate) output_credit: Option<crate::stages::worker::OutputCredit>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantService {
    pub owners: usize,
    pub active: usize,
    pub tokens: usize,
    pub pages: usize,
    pub weight: u32,
    pub virtual_finish: u64,
}

pub struct Engine<B: BackendProvider, P: SchedulingPolicy = CostAwarePolicy> {
    pub(crate) backend: B,
    pub(crate) policy: P,
    pub(crate) planning_workspace: P::Workspace,
    pub(crate) host: crate::cpu::storage::HostStorage,
    pub(crate) output_worker: Option<crate::stages::worker::OutputWorker>,
    pub(crate) output_pending: crate::pipeline::completion::cpu::OutputFlights,
    pub(crate) output_owners: infer_core::map::BoundedMap<RequestId, StepId>,
    pub(crate) workloads: Box<dyn WorkloadProvider + Send + Sync>,
    pub(crate) costs: Box<dyn CostModelProvider + Send + Sync>,
    pub(crate) admission_policy: Box<dyn AdmissionPolicy + Send + Sync>,
    pub(crate) pending_cost_observations: VecDeque<CostObservation>,
    pub(crate) replaying: bool,
    pub(crate) preemption_focus: Option<RequestId>,
    pub(crate) model: ModelIr,
    pub(crate) program: ExecutionProgram,
    pub(crate) config: RuntimeConfig,
    pub(crate) state: SequenceStateManager,
    pub(crate) resources_pending: crate::resource::ResourceWaiters,
    pub(crate) resource_epoch: u64,
    pub(crate) backend_resource_epoch: u64,
    pub(crate) seen_requests: BTreeSet<RequestId>,
    pub(crate) retired_request_floor: u64,
    pub(crate) tenants: infer_core::map::BoundedMap<std::sync::Arc<str>, TenantService>,
    pub(crate) ids: IdAllocator,
    pub(crate) inflight: Option<InFlight<B::Ticket>>,
    pub(crate) global_progress_epoch: u64,
    pub(crate) now_us: u64,
    pub(crate) guard: ProgressGuard,
    pub(crate) events: EventWriter,
    pub(crate) event_reader: Option<EventReader>,
    pub(crate) decisions: VecDeque<SchedulingDecision>,
    pub(crate) actions: VecDeque<crate::ReplayAction>,
    pub(crate) dropped_actions: u64,
    pub(crate) diagnostic_snapshot: Option<crate::RuntimeSnapshot>,
    pub(crate) fault: Option<EngineFault>,
    pub(crate) observations: crate::observation::Observations,
}
