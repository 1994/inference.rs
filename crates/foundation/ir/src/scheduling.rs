//! Scheduling inputs, evidence and cold-path cost calibration contracts.
use crate::{BackendKind, CostEstimate, ExecutionRole};
use infer_core::{ProgramId, RequestId, StepId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SchedulerConfig {
    pub prefill_chunk_tokens: usize,
    pub forward_chunk_tokens: usize,
    pub fair_quantum_tokens: usize,
    pub max_wait_us: u64,
    pub urgent_budget_percent: u32,
    /// Execution-time quantum is soft only for an otherwise feasible singleton token.
    pub max_singleton_gpu_us: u64,
    /// Deterministic packing trials per decision; yields a partial batch when exhausted.
    pub max_planning_probes: usize,
}
impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            prefill_chunk_tokens: 32,
            forward_chunk_tokens: 64,
            fair_quantum_tokens: 8,
            max_wait_us: 20_000,
            urgent_budget_percent: 75,
            max_singleton_gpu_us: 1_000_000,
            max_planning_probes: 4096,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostQuery {
    pub program: ProgramId,
    pub backend: BackendKind,
    pub role: ExecutionRole,
    pub tokens: usize,
    pub context_tokens: usize,
    /// GPU execution per token, or CPU execution per token on the Host backend.
    pub fallback_per_token_us: u64,
    pub workspace_bytes: u64,
    /// Additional resources only: already-reserved KV must not be charged twice.
    pub state_pages: usize,
    pub state_bytes: u64,
    pub transfer_us: u64,
    pub encoder_us: u64,
    #[serde(default)]
    pub page_growth: Option<PageGrowth>,
    #[serde(default)]
    pub logical_growth: Option<PageGrowth>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageGrowth {
    pub page_tokens: usize,
    pub allocated_pages: usize,
    pub bytes_per_page: u64,
    pub cow_tail: bool,
}
impl PageGrowth {
    #[must_use]
    pub fn required_pages(&self, end_tokens: usize) -> Option<usize> {
        (self.page_tokens > 0).then(|| {
            end_tokens
                .div_ceil(self.page_tokens)
                .saturating_sub(self.allocated_pages)
                .saturating_add(usize::from(self.cow_tail))
        })
    }
}
impl CostQuery {
    #[must_use]
    pub const fn from_unit(
        program: ProgramId,
        backend: BackendKind,
        role: ExecutionRole,
        tokens: usize,
        context_tokens: usize,
        unit: CostEstimate,
    ) -> Self {
        Self {
            program,
            backend,
            role,
            tokens,
            context_tokens,
            fallback_per_token_us: unit.gpu_us,
            workspace_bytes: unit.workspace_bytes,
            state_pages: unit.state_pages,
            state_bytes: unit.state_bytes,
            transfer_us: unit.transfer_us,
            encoder_us: unit.encoder_us,
            page_growth: None,
            logical_growth: None,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimingSource {
    CpuWall,
    MetalGpu,
    CudaGpu,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionTiming {
    pub elapsed_us: u64,
    pub source: TimingSource,
}
impl ExecutionTiming {
    #[must_use]
    pub const fn matches_backend(&self, backend: BackendKind) -> bool {
        match (backend, self.source) {
            (BackendKind::Cuda, TimingSource::CudaGpu)
            | (BackendKind::Metal, TimingSource::MetalGpu) => true,
            #[cfg(feature = "test-backends")]
            (BackendKind::TestCpu, TimingSource::CpuWall) => true,
            _ => false,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostObservation {
    pub step: StepId,
    pub work: crate::SharedOutput<CostQuery>,
    pub timing: ExecutionTiming,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CostModelConfig {
    pub adaptive: bool,
    pub max_profiles: usize,
    pub ewma_alpha_percent: u32,
    pub safety_margin_percent: u32,
}
impl Default for CostModelConfig {
    fn default() -> Self {
        Self {
            adaptive: true,
            max_profiles: 512,
            ewma_alpha_percent: 25,
            safety_margin_percent: 20,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostModelInspection {
    pub provider: String,
    pub adaptive: bool,
    pub profiles: usize,
    pub observations: u64,
    pub evictions: u64,
    pub last_source: Option<TimingSource>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionReason {
    UrgentSlo,
    Aging,
    DecodeLatency,
    WeightedFair,
    MinimumQuantum,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectionEvidence {
    pub request: RequestId,
    pub role: ExecutionRole,
    pub tokens: usize,
    pub target_us: Option<u64>,
    pub slack_us: Option<i64>,
    pub waiting_us: u64,
    pub virtual_finish: u64,
    pub predicted_us: u64,
    pub reason: SelectionReason,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TenantQuota {
    pub max_active_requests: usize,
    pub max_reserved_tokens: usize,
    pub max_state_pages: usize,
    pub weight: Option<u32>,
}
impl Default for TenantQuota {
    fn default() -> Self {
        Self {
            max_active_requests: 256,
            max_reserved_tokens: 65536,
            max_state_pages: 4096,
            weight: None,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AdmissionConfig {
    pub default_tenant: TenantQuota,
    pub tenants: BTreeMap<String, TenantQuota>,
    /// Reject predicted impossible latency targets; this is not an SLO guarantee.
    pub reject_infeasible_slo: bool,
}
#[derive(Debug, Clone)]
pub struct AdmissionInput<'a> {
    pub tenant: &'a str,
    pub weight: u32,
    pub reserved_tokens: usize,
    pub required_pages: usize,
    pub initial_pages: usize,
    pub required_bytes: Option<u64>,
    pub tenant_active: usize,
    pub tenant_tokens: usize,
    pub tenant_pages: usize,
    pub free_pages: usize,
    pub free_bytes: Option<u64>,
    pub now_us: u64,
    pub target_us: Option<u64>,
    pub predicted_latency_us: u64,
    pub minimum_execution_us: u64,
    pub max_atomic_us: u64,
    pub resources_ready: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdmissionReason {
    Capacity,
    TenantRequests,
    TenantTokens,
    TenantPages,
    TenantWeight,
    StateBytes,
    MediaNotReady,
    SloInfeasible,
    AtomicExecution,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissionDecision {
    pub rejection: Option<AdmissionReason>,
    pub required: u64,
    pub available: u64,
    pub predicted_latency_us: u64,
    pub target_us: Option<u64>,
    pub slo_at_risk: bool,
}
