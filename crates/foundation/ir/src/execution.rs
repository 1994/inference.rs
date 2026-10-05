use crate::{CapabilityRequirements, PrecisionPlan};
use infer_core::{DecisionId, KernelId, ModelId, OpId, ProgramId, RequestId, StateId, StepId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Operation {
    TokenEmbedding,
    RmsNorm,
    Attention,
    LinearAttention,
    Convolution,
    MatMul,
    Silu,
    Residual,
    LmHead,
    Pool,
    Rank,
    Decision,
    Split,
    Rope,
    Sigmoid,
    Multiply,
    GatedNorm,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationIr {
    pub id: OpId,
    pub operation: Operation,
    pub shape: Vec<usize>,
    pub layer: Option<usize>,
    pub requirements: CapabilityRequirements,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionIr {
    pub model: ModelId,
    pub precision: PrecisionPlan,
    pub operations: Vec<OperationIr>,
    pub dataflow: crate::DataflowGraph,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledOp {
    pub op: OperationIr,
    pub kernel: KernelId,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionProgram {
    pub backend: crate::BackendKind,
    pub id: ProgramId,
    pub model: ModelId,
    pub precision: PrecisionPlan,
    pub operations: Vec<CompiledOp>,
    pub workspace_bytes: u64,
    pub dataflow: crate::DataflowGraph,
}
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum ExecutionRole {
    #[default]
    Prefill,
    Decode,
    Forward,
    /// A step contains compatible work from more than one execution phase.
    Mixed,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphVariant {
    pub batch_capacity: usize,
    pub max_tokens: usize,
    #[serde(default)]
    pub program: Option<ProgramId>,
    #[serde(default)]
    pub workspace_bytes: u64,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostEstimate {
    pub gpu_us: u64,
    pub workspace_bytes: u64,
    pub state_pages: usize,
    #[serde(default)]
    pub logical_pages: usize,
    #[serde(default)]
    pub state_bytes: u64,
    pub transfer_us: u64,
    pub encoder_us: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadyWork {
    pub request: RequestId,
    pub program: ProgramId,
    pub state: StateId,
    pub role: ExecutionRole,
    pub remaining_tokens: usize,
    pub tenant: String,
    pub weight: u32,
    pub deadline_us: Option<u64>,
    pub virtual_finish: u64,
    pub cost_per_token: CostEstimate,
    pub cost_query: crate::CostQuery,
    pub latency_deadline_us: Option<u64>,
    pub remaining_latency_us: u64,
    pub remaining_completion_us: u64,
    pub last_service_us: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedWork {
    pub request: RequestId,
    pub state: StateId,
    pub token_count: usize,
    #[serde(default)]
    pub role: ExecutionRole,
}
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepPlan {
    pub id: StepId,
    pub decision: DecisionId,
    pub program: ProgramId,
    pub role: ExecutionRole,
    pub work: Vec<PlannedWork>,
    pub cost: CostEstimate,
    pub graph: Option<GraphVariant>,
    #[serde(default)]
    pub quantum_overrun: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeferReason {
    TokenBudget,
    GpuBudget,
    Workspace,
    BatchLimit,
    IncompatibleProgram,
    Deadline,
    StateCapacity,
    TransferBudget,
    EncoderBudget,
    ChunkLimit,
    AtomicCostLimit,
    PreemptionFocus { request: RequestId },
    Preparation,
    PlanningBudget,
}
impl DeferReason {
    pub const LABELS: [&'static str; 14] = [
        "token_budget",
        "gpu_budget",
        "workspace",
        "batch_limit",
        "incompatible_program",
        "deadline",
        "state_capacity",
        "transfer_budget",
        "encoder_budget",
        "chunk_limit",
        "atomic_cost_limit",
        "preemption_focus",
        "preparation",
        "planning_budget",
    ];
    #[must_use]
    pub const fn code(&self) -> u32 {
        match self {
            Self::TokenBudget => 0,
            Self::GpuBudget => 1,
            Self::Workspace => 2,
            Self::BatchLimit => 3,
            Self::IncompatibleProgram => 4,
            Self::Deadline => 5,
            Self::StateCapacity => 6,
            Self::TransferBudget => 7,
            Self::EncoderBudget => 8,
            Self::ChunkLimit => 9,
            Self::AtomicCostLimit => 10,
            Self::PreemptionFocus { .. } => 11,
            Self::Preparation => 12,
            Self::PlanningBudget => 13,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeferredWork {
    pub request: RequestId,
    pub reason: DeferReason,
    pub required: u64,
    pub available: u64,
}
/// Frozen bounded planning coverage. Unvisited requests have no resource rejection evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionWindow {
    pub epoch: u64,
    pub cost_epoch: u64,
    pub resource_epoch: u64,
    pub inspected: usize,
    pub ready: usize,
}
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulingDecision {
    #[serde(default)]
    pub window: Option<DecisionWindow>,
    pub id: DecisionId,
    pub step: Option<StepPlan>,
    pub deferred: Vec<DeferredWork>,
    #[serde(default)]
    pub selected: Vec<crate::SelectionEvidence>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    pub max_batch: usize,
    pub token_budget: usize,
    pub gpu_budget_us: u64,
    pub workspace_bytes: u64,
    pub free_state_pages: usize,
    pub free_logical_pages: usize,
    pub graphs: Vec<GraphVariant>,
    pub free_state_bytes: u64,
    pub transfer_budget_us: u64,
    pub encoder_budget_us: u64,
    pub scheduler: crate::SchedulerConfig,
}
#[derive(Debug, Clone)]
pub struct ExecutionTask {
    pub request: RequestId,
    pub state: StateId,
    pub tokens: crate::ExecutionInput,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelOutput {
    pub logits: Vec<f32>,
    pub hidden: Vec<Vec<f32>>,
}
#[derive(Debug, Clone)]
pub struct TaskOutput {
    pub request: RequestId,
    pub output: ModelOutput,
}

impl Clone for StepPlan {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            decision: self.decision,
            program: self.program,
            role: self.role,
            work: self.work.clone(),
            cost: self.cost,
            graph: self.graph.clone(),
            quantum_overrun: self.quantum_overrun,
        }
    }
    fn clone_from(&mut self, source: &Self) {
        self.id = source.id;
        self.decision = source.decision;
        self.program = source.program;
        self.role = source.role;
        self.cost = source.cost;
        self.quantum_overrun = source.quantum_overrun;
        self.work.clone_from(&source.work);
        self.graph.clone_from(&source.graph);
    }
}
impl Clone for SchedulingDecision {
    fn clone(&self) -> Self {
        Self {
            window: self.window.clone(),
            id: self.id,
            step: self.step.clone(),
            deferred: self.deferred.clone(),
            selected: self.selected.clone(),
        }
    }
    fn clone_from(&mut self, source: &Self) {
        self.id = source.id;
        self.window.clone_from(&source.window);
        self.step.clone_from(&source.step);
        self.deferred.clone_from(&source.deferred);
        self.selected.clone_from(&source.selected);
    }
}
