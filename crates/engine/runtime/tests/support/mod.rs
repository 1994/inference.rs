// Each integration test compiles its own copy of this module, so whatever a given scene does not
// use is dead code in that target.
#![allow(
    dead_code,
    reason = "shared test source, included once per test target"
)]

//! The declared protocol double for runtime scenes.
//!
//! Scenes that exercise admission, completion identity, checkpointing or resource ownership need
//! an engine, not a transformer. This module supplies the three declarations such a scene needs -
//! a model descriptor, the operations it lowers to, and a backend that completes work with fixed
//! outputs - so the scene can stop depending on a full CPU inference executor.
//!
//! Nothing here executes a model. `ProtocolBackend` reports a planning sample rather than a
//! device, hands out fixed outputs, and reports no timing, because a backend that runs nothing
//! has no timing to report.
use infer_core::{DeviceId, Error, KernelId, ModelId, Result, StateId, map::BoundedMap};
use infer_ir::{
    BackboneKind, CapabilityRequirements, DType, DeviceCapabilities, ExecutionProgram,
    ExecutionTask, FeedForward, Head, Mixer, Modality, ModelIr, ModelOutput, Operation,
    OutputReadout, PositionSpec, PrecisionPlan, StateKind, StateRequirement, StepPlan, TaskOutput,
};
use infer_spi::{BackendProvider, KernelProvider, KernelRegistration, SourceLocation};
use std::num::NonZeroU64;

/// A small decoder descriptor: two attention layers, hidden 8, intermediate 16, vocabulary 32.
#[must_use]
pub fn model(id: ModelId) -> ModelIr {
    const HIDDEN: usize = 8;
    const LAYERS: usize = 2;
    const KV_ELEMENTS_PER_HEAD: usize = 2;
    ModelIr {
        id,
        backbone: BackboneKind::Decoder,
        vocab_size: 32,
        hidden_size: HIDDEN,
        max_sequence: 128,
        mixers: vec![
            Mixer::Attention {
                query_heads: 1,
                kv_heads: 1,
                head_dim: HIDDEN,
                sliding_window: None,
                output_gate: false,
                qk_norm: false,
            };
            LAYERS
        ],
        feed_forward: FeedForward::Dense { intermediate: 16 },
        position: PositionSpec {
            rope_theta: 10_000.0,
            rotary_fraction: 1.0,
            multimodal_sections: vec![],
            interleaved: false,
        },
        norm_epsilon: 1e-5,
        norm_weight_offset: 0.0,
        heads: vec![
            Head::LanguageModel,
            Head::Embedding,
            Head::Rank,
            Head::Decision,
        ],
        modalities: vec![Modality::Text],
        state: (0..LAYERS)
            .map(|layer| StateRequirement {
                layer,
                kind: StateKind::AttentionKv,
                dtype: DType::F32,
                elements: HIDDEN * KV_ELEMENTS_PER_HEAD,
                per_token: true,
            })
            .collect(),
        tied_embeddings: false,
    }
}

/// The operations the descriptor above lowers to, declared so the engine can validate its plan.
pub struct DeclaredKernels;

const OPERATIONS: [Operation; 15] = [
    Operation::TokenEmbedding,
    Operation::RmsNorm,
    Operation::Attention,
    Operation::MatMul,
    Operation::Silu,
    Operation::Residual,
    Operation::LmHead,
    Operation::Pool,
    Operation::Rank,
    Operation::Decision,
    Operation::Split,
    Operation::Rope,
    Operation::Sigmoid,
    Operation::Multiply,
    Operation::GatedNorm,
];

impl KernelProvider for DeclaredKernels {
    fn kernels(&self) -> Vec<KernelRegistration> {
        OPERATIONS
            .into_iter()
            .enumerate()
            .map(|(index, operation)| KernelRegistration {
                backend: infer_ir::BackendKind::Cuda,
                id: KernelId::from_nonzero(
                    NonZeroU64::MIN.saturating_add(u64::try_from(index).unwrap_or(0)),
                ),
                operation,
                precision: PrecisionPlan::f32(),
                requirements: CapabilityRequirements {
                    compute_dtypes: vec![DType::F32],
                    ..CapabilityRequirements::default()
                },
                max_shape_elements: usize::MAX,
                workspace_bytes: 0,
                priority: 0,
                estimated_ns: 1,
                // Report the crate and file this module was compiled into, so a suite in another
                // crate that includes it does not claim to be the runtime's.
                source: SourceLocation {
                    crate_name: env!("CARGO_PKG_NAME").into(),
                    file: file!().into(),
                    function: "DeclaredKernels::kernels".into(),
                },
            })
            .collect()
    }
}

struct State {
    history: Vec<u32>,
}

pub struct ProtocolBackend {
    identity: String,
    states: BoundedMap<StateId, State>,
    template: ModelOutput,
    resources: infer_spi::ResourcePool,
    hidden: usize,
    capacity: usize,
}

impl ProtocolBackend {
    /// # Errors
    /// Returns an invalid-input error when the configured request bound cannot be reserved.
    pub fn new(requests: usize, batch: usize, ir: &ModelIr) -> Result<Self> {
        Self::tagged("declared-protocol-double", requests, batch, ir)
    }

    /// The same double with a declared weight tag.
    ///
    /// A snapshot records the backend's `identity` as its weights fingerprint, so a scene that
    /// checks a restore rejects different weights gives each side a different tag. The double
    /// declares that identity instead of hashing tensors it never loads.
    ///
    /// # Errors
    /// Returns an invalid-input error when the configured request bound cannot be reserved.
    pub fn tagged(tag: &str, requests: usize, batch: usize, ir: &ModelIr) -> Result<Self> {
        let _ = batch;
        // A declared ramp rather than a constant: sampling has to pick a token that is not the stop
        // token for scenes that count generated tokens, and the ramp makes that deterministic
        // instead of depending on which index sampling happens to choose.
        let mut logits = Vec::with_capacity(ir.vocab_size);
        let mut value = 0.0_f32;
        for _ in 0..ir.vocab_size {
            logits.push(value);
            value += 1.0;
        }
        Ok(Self {
            identity: tag.to_owned(),
            states: BoundedMap::new(requests)?,
            template: ModelOutput {
                logits,
                hidden: vec![vec![0.5; ir.hidden_size]],
                tokens: Vec::with_capacity(8),
            },
            resources: infer_spi::ResourcePool::new(requests + 4)?,
            hidden: ir.hidden_size,
            capacity: requests,
        })
    }
}

impl BackendProvider for ProtocolBackend {
    type Ticket = Option<Vec<TaskOutput>>;
    fn identity(&self) -> &str {
        &self.identity
    }
    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            backend: infer_ir::DeviceBackend::Cuda(infer_ir::NvidiaCapabilities {
                architecture: infer_ir::NvidiaArchitecture {
                    compute_major: 12,
                    compute_minor: 0,
                    multiprocessors: 170,
                },
                tensor_core_generation: Some(5),
                warp_size: 32,
                graphs: true,
                tma: true,
                clusters: true,
                pinned_transfer: true,
                cuda_ipc: true,
                nvlink: false,
                gpu_direct: false,
            }),
            device: DeviceId::ONE,
            compute_dtypes: vec![DType::F32],
            memory_bytes: 0,
            unified_memory: true,
            profiling: false,
            speculation: infer_ir::SpeculationCapability::default(),
        }
    }
    fn capture_execution_state(&self) -> Result<Option<Vec<u8>>> {
        // The double's whole device state is the token history it has been given per state. That
        // history has to survive a checkpoint: the engine resumes from the cursor it committed, and
        // a restored double with an empty history rejects the next decode as a stale cursor.
        let history: Vec<(&StateId, &Vec<u32>)> = self
            .states
            .iter()
            .map(|(id, state)| (id, &state.history))
            .collect();
        serde_json::to_vec(&history)
            .map(Some)
            .map_err(|error| Error::invalid(error.to_string()))
    }
    fn restore_execution_state(&mut self, state: Option<&[u8]>) -> Result<()> {
        let history: Vec<(StateId, Vec<u32>)> = state.map_or_else(
            || Ok(Vec::new()),
            |bytes| {
                serde_json::from_slice(bytes).map_err(|error| Error::invalid(error.to_string()))
            },
        )?;
        if history.len() > self.capacity {
            return Err(Error::invalid(
                "checkpoint holds more states than the double granted",
            ));
        }
        self.states.clear();
        for (id, tokens) in history {
            self.states.insert(id, State { history: tokens })?;
        }
        Ok(())
    }
    fn supports_recompute_preemption(&self) -> bool {
        // A scene that reclaims pages by recomputing needs the backend to allow it; the double has
        // no device state to lose, so it always can.
        true
    }
    fn supports_control_checkpoint(&self) -> bool {
        // The double holds fixed outputs and no device state, so a control checkpoint reconstructs
        // it exactly; scenes that checkpoint need this to be true.
        true
    }
    fn execution_graph(&self, ir: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        infer_model_recipes::decoder::lower(ir)
    }
    fn validate_program(&self, _: &ModelIr, _: &ExecutionProgram) -> Result<()> {
        Ok(())
    }
    fn begin_resource(
        &mut self,
        command: infer_spi::ResourceCommand,
    ) -> Result<infer_spi::ResourceTicket> {
        let (ticket, reply) = self.resources.channel()?;
        let _ = reply.send(infer_spi::execute_resource(self, command));
        Ok(ticket)
    }
    fn reserve_state_for(
        &mut self,
        state: StateId,
        _tokens: usize,
        _readout: OutputReadout,
    ) -> Result<()> {
        self.states.insert(
            state,
            State {
                history: Vec::new(),
            },
        )?;
        Ok(())
    }
    fn reset_state(&mut self, state: StateId) -> Result<()> {
        // A reset starts the sequence over, so the token cursor the engine sends next must start
        // from zero again; keeping the old history rejects the next prefill as a stale cursor.
        self.states
            .get_mut(&state)
            .ok_or_else(|| Error::invalid("unknown state"))?
            .history
            .clear();
        Ok(())
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        self.states
            .remove(&state)
            .ok_or_else(|| Error::invariant("release state missing"))?;
        Ok(())
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<Self::Ticket> {
        self.submit_borrowed(program, step, &tasks)
    }
    fn submit_borrowed(
        &mut self,
        _: &ExecutionProgram,
        _: &StepPlan,
        tasks: &[ExecutionTask],
    ) -> Result<Self::Ticket> {
        // A declared output per task. The double does not lease buffers: the engine may submit the
        // next unit before it returns the previous output, which a buffer pool would have to
        // tolerate anyway.
        let mut batch = Vec::with_capacity(tasks.len());
        for task in tasks {
            let state = self
                .states
                .get_mut(&task.state)
                .ok_or_else(|| Error::invariant("submit state missing"))?;
            // The output must carry one hidden row per token in the sequence, because the engine
            // projects a hidden output only when the rows match the context it planned for.
            task.tokens.commit(&mut state.history)?;
            let rows = state.history.len();
            let readout = task.tokens.readout();
            let mut output = self.template.clone();
            if readout == OutputReadout::None {
                output.logits.clear();
            }
            output.hidden.resize(
                if readout == OutputReadout::Full {
                    rows
                } else {
                    0
                },
                vec![0.5; self.hidden],
            );
            batch.push(TaskOutput {
                request: task.request,
                output,
            });
        }
        Ok(Some(batch))
    }
    fn recycle_output(&mut self, _state: StateId, _output: ModelOutput) -> Result<()> {
        // The double allocates its declared outputs, so a returned buffer needs no bookkeeping.
        Ok(())
    }
    fn recycle_batch(&mut self, _outputs: Vec<TaskOutput>) -> Result<()> {
        Ok(())
    }
    fn poll(&mut self, ticket: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>> {
        Ok(ticket.take())
    }
}
