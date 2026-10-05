//! Frozen owner budgets. Backend drivers share this CPU contract without selecting a device kind.
mod history;
pub mod plans;
pub mod storage;
use crate::RuntimeConfig;
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CpuRuntimeConfig {
    pub placement: OwnerPlacement,
    pub maintenance_items: usize,
    pub host_tokens: usize,
    pub host_bytes: usize,
    pub urgency_window_us: u64,
    pub poll_min_us: u64,
    pub poll_max_us: u64,
}
impl Default for CpuRuntimeConfig {
    fn default() -> Self {
        Self {
            placement: OwnerPlacement::default(),
            maintenance_items: 64,
            host_tokens: 1 << 22,
            host_bytes: 256 << 20,
            urgency_window_us: 10_000,
            poll_min_us: 25,
            poll_max_us: 1000,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuRuntimeInspection {
    pub config: CpuRuntimeConfig,
    pub configured_storage_bytes: usize,
    pub retained_host_tokens: usize,
    pub retained_host_bytes: usize,
    pub candidate_capacity: usize,
    pub logical_cpus: usize,
    pub placement: infer_core::placement::PlacementReport,
    pub prepared_next: bool,
}
impl CpuRuntimeConfig {
    /// Include all three synchronous/asynchronous sampling workspaces in the frozen owner budget.
    /// # Errors
    /// Rejects overflow or a vocabulary whose persistent worker storage exceeds the host limit.
    pub fn fixed_bytes_for(&self, runtime: &RuntimeConfig, vocabulary: usize) -> Result<usize> {
        let fixed = self.fixed_bytes(runtime)?;
        let total = vocabulary
            .checked_mul(3 * (size_of::<(usize, f32)>() + size_of::<f64>()))
            .and_then(|sampling| fixed.checked_add(sampling))
            .ok_or_else(|| Error::invalid("CPU sampling storage overflow"))?;
        if self
            .host_tokens
            .checked_mul(4)
            .and_then(|tokens| total.checked_add(tokens))
            .is_none_or(|bytes| bytes > self.host_bytes)
        {
            return Err(Error::invalid(
                "CPU sampling pools exceed host memory budget",
            ));
        }
        Ok(total)
    }
    /// Conservative application storage budget; driver/device allocations have a separate backend pool.
    /// # Errors
    /// Rejects overflowing capacities, invalid poll/drain limits or a host budget too small for its pools.
    pub fn fixed_bytes(&self, runtime: &RuntimeConfig) -> Result<usize> {
        if self.maintenance_items == 0
            || self.host_tokens == 0
            || self.host_bytes == 0
            || self.poll_min_us == 0
            || self.poll_max_us < self.poll_min_us
            || self.poll_max_us > 1_000_000
        {
            return Err(Error::invalid("invalid CPU owner budgets"));
        }
        let batch_record = size_of::<infer_ir::PlannedWork>()
            + size_of::<infer_ir::SelectionEvidence>()
            + 3 * size_of::<infer_ir::CostQuery>();
        let history_record = runtime
            .max_batch
            .checked_mul(batch_record)
            .and_then(|bytes| {
                runtime
                    .candidate_limit
                    .min(runtime.max_requests)
                    .checked_mul(size_of::<infer_ir::DeferredWork>())
                    .and_then(|deferred| bytes.checked_add(deferred))
            })
            .and_then(|bytes| bytes.checked_add(512))
            .ok_or_else(|| Error::invalid("CPU history record size overflow"))?;
        let mut fixed = 0usize;
        for (count, bytes) in [
            (runtime.max_requests, 8192),
            (runtime.max_batch, 4096),
            (runtime.event_capacity, 128),
            (runtime.history_capacity, history_record),
            (1, runtime.max_history_bytes),
            (
                runtime.candidate_limit.min(runtime.max_requests),
                size_of::<infer_ir::ReadyWork>() + 256,
            ),
            (runtime.state_pages, 256),
            (runtime.cost_model.max_profiles, 256),
        ] {
            fixed = count
                .checked_mul(bytes)
                .and_then(|bytes| fixed.checked_add(bytes))
                .ok_or_else(|| Error::invalid("CPU storage budget overflow"))?;
        }
        let total = self
            .host_tokens
            .checked_mul(size_of::<u32>())
            .and_then(|tokens| fixed.checked_add(tokens))
            .ok_or_else(|| Error::invalid("CPU token budget overflow"))?;
        if total > self.host_bytes {
            return Err(Error::invalid(
                "CPU pools exceed configured host memory budget",
            ));
        }
        Ok(fixed)
    }
}

impl<B: infer_spi::BackendProvider, P: infer_spi::SchedulingPolicy> crate::Engine<B, P> {
    pub(crate) fn check_history_storage(&self) -> Result<()> {
        let (tokens, bytes) =
            self.actions
                .iter()
                .fold((0usize, 0usize), |(tokens, bytes), action| {
                    (
                        tokens.saturating_add(action.retained_tokens()),
                        bytes.saturating_add(action.retained_bytes()),
                    )
                });
        if tokens != self.host.history_tokens
            || bytes != self.host.history_bytes
            || tokens > self.config.max_history_tokens
            || bytes > self.config.max_history_bytes
            || self.actions.len() > self.config.history_capacity
            || self.decisions.len() > self.config.history_capacity
        {
            return Err(Error::invariant("CPU history storage accounting mismatch"));
        }
        Ok(())
    }
    pub(crate) fn cpu_stage(
        &mut self,
        stage: infer_core::event::CpuStage,
        id: u64,
        start: std::time::Instant,
        items: usize,
    ) {
        if self.replaying {
            return;
        }
        self.emit_semantic(infer_core::event::SemanticEvent {
            timestamp_us: self.now_us,
            kind: infer_core::event::EventKind::CpuStageTiming,
            object_kind: infer_core::event::ObjectKind::Program,
            reserved: stage as u32,
            object_id: self.program.id.get(),
            correlation_id: id,
            arg0: u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX),
            arg1: items as u64,
        });
    }
}

/// Separate owners can select separate cores/nodes, without putting device architecture in CPU policy.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OwnerPlacement {
    pub scheduler: infer_core::placement::ThreadPlacement,
    pub device: infer_core::placement::ThreadPlacement,
    pub output: infer_core::placement::ThreadPlacement,
}

impl OwnerPlacement {
    /// Resolve separate physical cores on the GPU NUMA node within inherited cpusets.
    /// # Errors
    /// Rejects unknown PCI NUMA locality, unavailable cores/nodes or non-Linux hosts.
    pub fn for_gpu(pci_address: &str) -> Result<Self> {
        let pair = infer_core::placement::PlacementPair::for_gpu(pci_address)?;
        Ok(Self {
            scheduler: pair.scheduler,
            device: pair.device,
            output: infer_core::placement::ThreadPlacement::default(),
        })
    }
}
