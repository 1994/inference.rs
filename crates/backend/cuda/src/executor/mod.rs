//! Engine adapter for shared weights and isolated resident CUDA graphs.
//! Execution currently settles synchronously; tickets preserve the engine ownership protocol.
mod capabilities;
mod drafting;
mod execution;
mod pool;
pub mod prefix;
mod prefix_state;
mod profiling;
mod provider;
pub use pool::PoolInspection;
mod state;
mod validation;
use crate::{loading::LoadedModel, resident::DeviceProgram, resident::slot_batch::SlotPool};
use infer_core::{Error, Result, StateId};
use infer_ir::{DeviceCapabilities, ExecutionProgram, ModelOutput, OutputReadout, TaskOutput};
use std::{collections::BTreeMap, sync::Arc};

pub struct CudaBackend {
    loaded: LoadedModel,
    capabilities: DeviceCapabilities,
    operations: Vec<infer_ir::CompiledOp>,
    workspace_bytes: u64,
    identity: String,
    owner: Arc<()>,
    states: BTreeMap<StateId, Sequence>,
    budget: u64,
    maximum_states: usize,
    pool: pool::StatePool,
    prefix: prefix::PrefixCache,
    /// Continuous-batching decode slots, captured lazily on the first batchable step;
    /// `None` with `slots_disabled` means initialization failed and decode stays serial.
    slots: Option<SlotPool>,
    draft_slots: Option<SlotPool>,
    slots_disabled: bool,
    busy: bool,
    fatal: Option<Error>,
}
struct Sequence {
    program: DeviceProgram,
    speculation: Option<Speculation>,
    capacity: usize,
    readout: OutputReadout,
    history: Vec<u32>,
    hidden: Vec<Vec<f32>>,
    budget: u64,
    poisoned: bool,
    slot: Option<SlotLease>,
}
/// A sequence's lease on one continuous-batching decode slot. `batched` is the sticky
/// mode: a leased sequence always decodes through the shared slot graph and never
/// returns to its private target graph; the pool may also verify draft proposals.
pub(super) struct SlotLease {
    slot: usize,
    position: usize,
    batched: bool,
}
/// Mutable draft state for greedy MTP speculation on one resident sequence.
pub(super) struct Speculation {
    program: DeviceProgram,
    depth: usize,
    /// Index where generated tokens start, captured on the first decode step.
    prompt_len: Option<usize>,
    /// Target hidden of the most recently consumed token; the draft fuses the token before it.
    last_hidden: Vec<f32>,
    scratch: infer_workloads::SamplingWorkspace,
}
impl Speculation {
    pub(super) fn new(program: DeviceProgram, depth: usize) -> Self {
        Self {
            program,
            depth,
            prompt_len: None,
            last_hidden: Vec::new(),
            scratch: infer_workloads::SamplingWorkspace::default(),
        }
    }
    /// # Errors
    /// Rejects failed CUDA state resets.
    pub(super) fn reset(&mut self) -> Result<()> {
        self.program.reset()?;
        self.prompt_len = None;
        self.last_hidden.clear();
        Ok(())
    }
}
/// A completion can be consumed exactly once by its originating backend.
pub struct CudaTicket {
    owner: Arc<()>,
    result: Option<Result<Vec<TaskOutput>>>,
    drain_required: bool,
}
/// Prefix-cache capacity derived from the state budget: default on, a quarter of the budget,
/// with a floor so small devices still cache one prefix and a disabled cache when nothing fits.
const fn prefix_cache_bytes(state_budget: u64) -> u64 {
    const DIVISOR: u64 = 4;
    const FLOOR: u64 = 128 * 1024 * 1024;
    let share = state_budget / DIVISOR;
    if share < FLOOR { 0 } else { share }
}

impl CudaBackend {
    /// Prefix cache occupancy and hit counters, for observability.
    #[must_use]
    pub fn prefix_inspection(&self) -> (u64, u64, u64, usize) {
        (
            self.prefix.hits,
            self.prefix.lookups,
            self.prefix.bytes(),
            self.prefix.len(),
        )
    }

    /// Effective load-time choices, so a benchmark report can prove which graph widths and
    /// speculation depth actually ran instead of restating the request.
    #[must_use]
    pub fn execution_profile(&self) -> infer_ir::ExecutionProfileInspection {
        self.loaded.execution_profile()
    }

    /// # Errors
    /// Rejects an empty state limit/budget or unavailable device attributes.
    pub fn new(loaded: LoadedModel, state_budget: u64, maximum_states: usize) -> Result<Self> {
        if state_budget == 0 || maximum_states == 0 {
            return Err(Error::invalid("CUDA state limits must be nonzero"));
        }
        let mut capabilities = capabilities::query(loaded.device())?;
        // The model provider declares what the device must support; fail at load, not per step.
        capabilities.require(loaded.requirements())?;
        // Greedy-only speculation, and only when a draft was actually loaded.
        capabilities.speculation = infer_ir::SpeculationCapability {
            draft_depth: loaded.mtp_depth(),
            greedy_only: true,
        };
        let mut registry = infer_kernel_api::KernelRegistry::default();
        registry.register(&crate::registry::CudaKernels)?;
        let program = infer_compiler::compile(
            infer_core::ProgramId::ONE,
            infer_compiler::lower(
                loaded.model(),
                loaded.graph().clone(),
                infer_ir::PrecisionPlan::f32(),
            )?,
            &registry,
            &capabilities,
            state_budget,
        )?;
        let identity = format!("cuda-resident/{}", loaded.device().name()?);
        Ok(Self {
            loaded,
            capabilities,
            operations: program.operations,
            workspace_bytes: program.workspace_bytes,
            identity,
            owner: Arc::new(()),
            states: BTreeMap::new(),
            budget: state_budget,
            maximum_states,
            pool: pool::StatePool::default(),
            prefix: prefix::PrefixCache::new(prefix_cache_bytes(state_budget)),
            slots: None,
            draft_slots: None,
            slots_disabled: false,
            busy: false,
            fatal: None,
        })
    }
    #[must_use]
    pub const fn model(&self) -> &infer_ir::ModelIr {
        self.loaded.model()
    }

    /// Compile registrations for the actual loaded graph, including package-bound weights.
    /// # Errors
    /// Rejects invalid lowering, unavailable kernels or insufficient workspace.
    pub fn compile(&self, id: infer_core::ProgramId) -> Result<ExecutionProgram> {
        let mut registry = infer_kernel_api::KernelRegistry::default();
        registry.register(&crate::registry::CudaKernels)?;
        let ir = infer_compiler::lower(
            self.model(),
            self.loaded.graph().clone(),
            infer_ir::PrecisionPlan::f32(),
        )?;
        infer_compiler::compile(id, ir, &registry, &self.capabilities, self.budget)
    }
    fn idle(&self) -> Result<()> {
        if let Some(error) = &self.fatal {
            return Err(error.clone());
        }
        if self.busy {
            return Err(Error::new(
                infer_core::ErrorCode::Conflict,
                "CUDA completion must be consumed first",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod prefix_capacity_tests {
    use super::prefix_cache_bytes;

    const MIB: u64 = 1024 * 1024;

    #[test]
    fn capacity_is_a_quarter_of_the_budget_above_the_floor() {
        assert_eq!(prefix_cache_bytes(4 * 1024 * MIB), 1024 * MIB);
        assert_eq!(prefix_cache_bytes(8 * 1024 * MIB), 2 * 1024 * MIB);
    }

    #[test]
    fn capacity_is_disabled_when_a_quarter_would_be_useless() {
        // Below the floor a prefix entry could not be stored, so the cache stays off.
        assert_eq!(prefix_cache_bytes(0), 0);
        assert_eq!(prefix_cache_bytes(511 * MIB), 0);
        // Exactly at the floor it turns on.
        assert_eq!(prefix_cache_bytes(512 * MIB), 128 * MIB);
    }
}
