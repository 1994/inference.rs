//! Fixtures shared by the conformance self-tests.
#![allow(
    dead_code,
    reason = "each test module mounts the same support module but uses only a subset of it"
)]

use infer_core::{DeviceId, Error, ErrorCode, KernelId, ModelId, OpId, ProgramId, Result, StateId};
use infer_ir::{
    BackboneKind, BackendKind, BackendRequirements, CapabilityRequirements, CompiledOp, DType,
    DataflowGraph, DeviceBackend, DeviceCapabilities, ExecutionProgram, FeedForward, Head,
    MetalCapabilities, MetalRequirements, Mixer, ModelIr, Operation, OperationIr, OutputReadout,
    PositionSpec, PrecisionPlan, SpeculationCapability, StateRecipe,
};
use infer_spi::{BackendProvider, KernelProvider, KernelRegistration, SourceLocation};
use std::collections::BTreeSet;

/// State recipe block size used by the mock.
const BLOCK_SIZE: usize = 4;
/// Bytes per state element used by the mock recipe.
const STATE_BYTES: usize = 4;
/// Bytes per block lease used by the mock recipe; `StateRecipe` rejects a zero lease size.
const LEASE_BYTES: usize = 8;
/// Shape limit declared by the mock manifest.
const SHAPE_LIMIT: usize = 1_024;
/// SIMD width the mock capability sample reports.
const SIMD_WIDTH: u32 = 32;
/// SIMD width a deliberately unselectable registration asks for.
const UNSATISFIED_SIMD: u32 = 64;

/// Capability sample used by every self-test.
#[must_use]
pub fn caps() -> DeviceCapabilities {
    DeviceCapabilities {
        backend: DeviceBackend::Metal(MetalCapabilities {
            simd_width: SIMD_WIDTH,
        }),
        device: DeviceId::ONE,
        compute_dtypes: vec![DType::F32],
        memory_bytes: 1 << 20,
        unified_memory: true,
        profiling: false,
        speculation: SpeculationCapability::default(),
    }
}

/// Small dense model with one attention layer.
#[must_use]
pub fn model() -> ModelIr {
    ModelIr {
        id: ModelId::ONE,
        backbone: BackboneKind::Decoder,
        vocab_size: 8,
        hidden_size: 4,
        max_sequence: 16,
        mixers: vec![Mixer::Attention {
            query_heads: 1,
            kv_heads: 1,
            head_dim: 4,
            sliding_window: None,
            output_gate: false,
            qk_norm: false,
        }],
        feed_forward: FeedForward::Dense { intermediate: 8 },
        position: PositionSpec {
            rope_theta: 10_000.0,
            rotary_fraction: 1.0,
            multimodal_sections: Vec::new(),
            interleaved: false,
        },
        norm_epsilon: 1e-5,
        norm_weight_offset: 0.0,
        heads: vec![Head::LanguageModel],
        modalities: Vec::new(),
        state: Vec::new(),
        tied_embeddings: true,
    }
}

/// Empty dataflow used only for state-recipe sizing and program identity.
#[must_use]
pub fn empty_graph() -> DataflowGraph {
    DataflowGraph {
        tensors: Vec::new(),
        nodes: Vec::new(),
        hidden: None,
        logits: None,
        lifetimes: Vec::new(),
        scratch_elements: 0,
    }
}

/// Program with one compiled operation, so kernel-id rejection is exercised.
#[must_use]
pub fn program(model: &ModelIr, backend: BackendKind) -> ExecutionProgram {
    ExecutionProgram {
        backend,
        id: ProgramId::ONE,
        model: model.id,
        precision: PrecisionPlan::f32(),
        operations: vec![CompiledOp {
            op: OperationIr {
                id: OpId::ONE,
                operation: Operation::MatMul,
                shape: vec![1],
                layer: None,
                requirements: CapabilityRequirements {
                    compute_dtypes: vec![DType::F32],
                    backend: None,
                },
            },
            kernel: KernelId::ONE,
        }],
        workspace_bytes: 0,
        dataflow: empty_graph(),
    }
}

/// Backend that satisfies the Tier-0 checks, with switches for negative cases.
pub struct MockBackend {
    model: ModelIr,
    program: ExecutionProgram,
    recipe: StateRecipe,
    caps: DeviceCapabilities,
    identity: &'static str,
    strict: bool,
    discount: u64,
    reserved: BTreeSet<StateId>,
}

impl MockBackend {
    /// Consistent backend.
    ///
    /// # Errors
    /// Returns an invalid-input error if the mock recipe cannot be compiled.
    pub fn new() -> Result<Self> {
        Self::build(true, "mock", 0)
    }

    /// Backend whose program validation accepts everything.
    ///
    /// # Errors
    /// Returns an invalid-input error if the mock recipe cannot be compiled.
    pub fn loose() -> Result<Self> {
        Self::build(false, "mock", 0)
    }

    /// Backend with an empty identity.
    ///
    /// # Errors
    /// Returns an invalid-input error if the mock recipe cannot be compiled.
    pub fn anonymous() -> Result<Self> {
        Self::build(true, "", 0)
    }

    /// Subtract `bytes` from every state quote, to exercise the quote floor.
    #[must_use]
    pub fn with_discount(mut self, bytes: u64) -> Self {
        self.discount = bytes;
        self
    }

    #[must_use]
    pub fn model(&self) -> &ModelIr {
        &self.model
    }

    #[must_use]
    pub fn program(&self) -> &ExecutionProgram {
        &self.program
    }

    fn build(strict: bool, identity: &'static str, discount: u64) -> Result<Self> {
        let model = model();
        let recipe = StateRecipe::compile(
            &model,
            &empty_graph(),
            BLOCK_SIZE,
            STATE_BYTES,
            false,
            LEASE_BYTES,
        )?;
        let program = program(&model, BackendKind::Metal);
        Ok(Self {
            model,
            program,
            recipe,
            caps: caps(),
            identity,
            strict,
            discount,
            reserved: BTreeSet::new(),
        })
    }

    fn quote(&self, capacity: usize) -> Result<Option<u64>> {
        let layout = self.recipe.layout(capacity, OutputReadout::Full)?;
        let pages = u64::try_from(layout.max_pages).unwrap_or(u64::MAX);
        let kv = layout
            .kv_block_bytes
            .checked_mul(pages)
            .ok_or_else(|| Error::invalid("mock KV quote overflow"))?;
        let total = kv
            .checked_add(layout.private_bytes)
            .and_then(|total| total.checked_add(layout.host_bytes))
            .ok_or_else(|| Error::invalid("mock state quote overflow"))?;
        Ok(Some(total.saturating_sub(self.discount)))
    }
}

impl BackendProvider for MockBackend {
    type Ticket = ();

    fn identity(&self) -> &str {
        self.identity
    }

    fn capabilities(&self) -> DeviceCapabilities {
        self.caps.clone()
    }

    fn state_recipe(&self) -> Option<&StateRecipe> {
        Some(&self.recipe)
    }

    fn state_reservation_bytes(&self, capacity: usize) -> Result<Option<u64>> {
        self.quote(capacity)
    }

    fn state_reservation_bytes_for(
        &self,
        capacity: usize,
        _readout: OutputReadout,
    ) -> Result<Option<u64>> {
        self.quote(capacity)
    }

    fn state_capacity_bytes(&self) -> Result<Option<u64>> {
        Ok(Some(self.caps.memory_bytes))
    }

    fn free_state_bytes(&self) -> Result<Option<u64>> {
        Ok(Some(self.caps.memory_bytes))
    }

    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()> {
        if !self.strict {
            return Ok(());
        }
        if model != &self.model {
            return Err(Error::invalid("mock model mismatch"));
        }
        if program != &self.program {
            return Err(Error::invalid("mock program mismatch"));
        }
        Ok(())
    }

    fn execution_graph(&self, model: &ModelIr) -> Result<DataflowGraph> {
        if model != &self.model {
            return Err(Error::invalid("mock execution graph model mismatch"));
        }
        Ok(self.program.dataflow.clone())
    }

    fn reserve_state(&mut self, state: StateId, capacity: usize) -> Result<()> {
        if capacity == 0 {
            return Err(Error::invalid("mock state capacity"));
        }
        if !self.reserved.insert(state) {
            return Err(Error::new(ErrorCode::Conflict, "mock state exists"));
        }
        Ok(())
    }

    fn release_state(&mut self, state: StateId) -> Result<()> {
        if self.reserved.remove(&state) {
            Ok(())
        } else {
            Err(Error::new(ErrorCode::NotFound, "mock state is unknown"))
        }
    }

    fn reusable_prefix(&self, _tokens: &[u32], _maximum: usize) -> usize {
        0
    }

    fn submit(
        &mut self,
        _program: &ExecutionProgram,
        _step: &infer_ir::StepPlan,
        _tasks: Vec<infer_ir::ExecutionTask>,
    ) -> Result<Self::Ticket> {
        Ok(())
    }

    fn poll(&mut self, _ticket: &mut Self::Ticket) -> Result<Option<Vec<infer_ir::TaskOutput>>> {
        Ok(Some(Vec::new()))
    }
}

/// Kernel table with switches for the negative manifest cases.
pub struct MockKernels {
    backend: BackendKind,
    duplicate: bool,
    requires_simd: bool,
}

impl MockKernels {
    #[must_use]
    pub fn good() -> Self {
        Self {
            backend: BackendKind::Metal,
            duplicate: false,
            requires_simd: false,
        }
    }

    #[must_use]
    pub fn foreign() -> Self {
        Self {
            backend: BackendKind::Cuda,
            duplicate: false,
            requires_simd: false,
        }
    }

    #[must_use]
    pub fn unselectable() -> Self {
        Self {
            backend: BackendKind::Metal,
            duplicate: false,
            requires_simd: true,
        }
    }

    #[must_use]
    pub fn duplicate() -> Self {
        Self {
            backend: BackendKind::Metal,
            duplicate: true,
            requires_simd: false,
        }
    }
}

impl KernelProvider for MockKernels {
    fn kernels(&self) -> Vec<KernelRegistration> {
        let second = if self.duplicate {
            KernelId::ONE
        } else {
            KernelId::new(2).unwrap_or(KernelId::ONE)
        };
        let requirement =
            self.requires_simd
                .then_some(BackendRequirements::Metal(MetalRequirements {
                    min_simd_width: Some(UNSATISFIED_SIMD),
                }));
        vec![
            registration(
                KernelId::ONE,
                Operation::MatMul,
                self.backend,
                requirement.clone(),
            ),
            registration(second, Operation::RmsNorm, self.backend, requirement),
        ]
    }
}

fn registration(
    id: KernelId,
    operation: Operation,
    backend: BackendKind,
    requirement: Option<BackendRequirements>,
) -> KernelRegistration {
    KernelRegistration {
        backend,
        id,
        operation,
        precision: PrecisionPlan::f32(),
        requirements: CapabilityRequirements {
            compute_dtypes: vec![DType::F32],
            backend: requirement,
        },
        max_shape_elements: SHAPE_LIMIT,
        workspace_bytes: 0,
        priority: 0,
        estimated_ns: 1,
        source: SourceLocation {
            crate_name: "infer-backend-conformance-tests".into(),
            file: "support/mod.rs".into(),
            function: "registration".into(),
        },
    }
}
