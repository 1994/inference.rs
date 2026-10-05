//! Validated runtime capacity and planning limits.
use infer_core::{Error, Result};
use infer_ir::{AdmissionConfig, CostModelConfig, SchedulerConfig};
use infer_scheduler::{CalibratedCosts, ResourceAdmission};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    pub max_requests: usize,
    pub cpu: crate::CpuRuntimeConfig,
    pub candidate_limit: usize,
    pub max_request_units: usize,
    pub max_input_tokens: usize,
    pub state_pages: usize,
    pub page_tokens: usize,
    pub max_batch: usize,
    pub token_budget: usize,
    pub gpu_budget_us: u64,
    pub workspace_bytes: u64,
    pub cost_per_token_us: u64,
    pub progress_limit: usize,
    pub submission_timeout_us: u64,
    pub resource_timeout_us: u64,
    pub output_timeout_us: u64,
    pub max_queue_wait_us: u64,
    pub history_capacity: usize,
    pub max_history_tokens: usize,
    pub max_history_bytes: usize,
    pub event_capacity: usize,
    pub scheduler: SchedulerConfig,
    pub cost_model: CostModelConfig,
    pub admission: AdmissionConfig,
    pub transfer_budget_us: u64,
    pub encoder_budget_us: u64,
}
impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_requests: 256,
            cpu: crate::CpuRuntimeConfig::default(),
            candidate_limit: 256,
            max_request_units: 256,
            max_input_tokens: 65536,
            state_pages: 4096,
            page_tokens: 16,
            max_batch: 16,
            token_budget: 64,
            gpu_budget_us: 10000,
            workspace_bytes: 1 << 20,
            cost_per_token_us: 1,
            progress_limit: 3,
            submission_timeout_us: 30_000_000,
            resource_timeout_us: 250_000,
            output_timeout_us: 5_000_000,
            max_queue_wait_us: 30_000_000,
            history_capacity: 4096,
            max_history_tokens: 1 << 20,
            max_history_bytes: 16 << 20,
            event_capacity: 8192,
            scheduler: SchedulerConfig::default(),
            cost_model: CostModelConfig::default(),
            admission: AdmissionConfig::default(),
            transfer_budget_us: 10_000,
            encoder_budget_us: 10_000,
        }
    }
}
impl RuntimeConfig {
    ///
    /// # Errors
    /// Returns an invalid-input or invariant error if dimensions, identities, ranges, or ownership are inconsistent.
    pub fn validate(&self) -> Result<()> {
        if self.event_capacity > 1_048_576 || self.history_capacity > 1_048_576 {
            return Err(Error::invalid(
                "event/history capacities must not exceed 1048576 records",
            ));
        }
        if [
            self.max_requests,
            self.candidate_limit,
            self.max_request_units,
            self.max_input_tokens,
            self.state_pages,
            self.page_tokens,
            self.max_batch,
            self.token_budget,
            self.progress_limit,
            self.history_capacity,
            self.max_history_tokens,
            self.max_history_bytes,
            self.event_capacity,
        ]
        .contains(&0)
            || self.gpu_budget_us == 0
            || self.cost_per_token_us == 0
            || self.max_queue_wait_us == 0
            || self.submission_timeout_us == 0
            || self.resource_timeout_us == 0
            || self.output_timeout_us == 0
        {
            return Err(Error::invalid(
                "runtime capacities/timeouts must be positive",
            ));
        }
        if self.max_batch > 64 {
            return Err(Error::invalid("runtime batch exceeds 64 completion cells"));
        }
        if self.cost_per_token_us > self.scheduler.max_singleton_gpu_us {
            return Err(Error::invalid("atomic dispatch limit cannot fit one token"));
        }
        self.cpu.fixed_bytes(self)?;
        infer_scheduler::validate_scheduler(&self.scheduler)?;
        ResourceAdmission::new(self.admission.clone())?;
        CalibratedCosts::new("validation".into(), self.cost_model.clone())?;
        Ok(())
    }
}
