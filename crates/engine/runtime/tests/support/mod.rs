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
                source: SourceLocation {
                    crate_name: "infer-runtime".into(),
                    file: "crates/engine/runtime/tests/support/mod.rs".into(),
                    function: "DeclaredKernels::kernels".into(),
                },
            })
            .collect()
    }
}

struct State {
    output: Option<ModelOutput>,
    history: Vec<u32>,
}

pub struct ProtocolBackend {
    states: BoundedMap<StateId, State>,
    outputs: Vec<ModelOutput>,
    batches: Vec<Vec<TaskOutput>>,
    resources: infer_spi::ResourcePool,
    vocabulary: usize,
    hidden: usize,
    capacity: usize,
}

impl ProtocolBackend {
    /// # Errors
    /// Returns an invalid-input error when the configured request bound cannot be reserved.
    pub fn new(requests: usize, batch: usize, ir: &ModelIr) -> Result<Self> {
        Ok(Self {
            states: BoundedMap::new(requests)?,
            outputs: (0..requests)
                .map(|_| ModelOutput {
                    logits: vec![1.0; ir.vocab_size],
                    hidden: vec![vec![0.5; ir.hidden_size]],
                    tokens: Vec::with_capacity(8),
                })
                .collect(),
            batches: (0..4).map(|_| Vec::with_capacity(batch)).collect(),
            resources: infer_spi::ResourcePool::new(requests + 4)?,
            vocabulary: ir.vocab_size,
            hidden: ir.hidden_size,
            capacity: requests,
        })
    }
}

impl BackendProvider for ProtocolBackend {
    type Ticket = Option<Vec<TaskOutput>>;
    fn identity(&self) -> &'static str {
        "declared-protocol-double"
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
        // The double's whole device state is which states it has granted; it holds nothing else,
        // so a checkpoint is that list and no tensor data.
        let states: Vec<StateId> = self.states.keys().copied().collect();
        serde_json::to_vec(&states)
            .map(Some)
            .map_err(|error| Error::invalid(error.to_string()))
    }
    fn restore_execution_state(&mut self, state: Option<&[u8]>) -> Result<()> {
        let states: Vec<StateId> = state.map_or_else(
            || Ok(Vec::new()),
            |bytes| {
                serde_json::from_slice(bytes).map_err(|error| Error::invalid(error.to_string()))
            },
        )?;
        if states.len() > self.capacity {
            return Err(Error::invalid(
                "checkpoint holds more states than the double granted",
            ));
        }
        self.states.clear();
        for id in states {
            let output = self
                .outputs
                .pop()
                .ok_or_else(|| Error::invariant("readback credit missing"))?;
            self.states.insert(
                id,
                State {
                    output: Some(output),
                    history: Vec::new(),
                },
            )?;
        }
        Ok(())
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
        let output = self
            .outputs
            .pop()
            .ok_or_else(|| Error::invariant("readback credit missing"))?;
        self.states.insert(
            state,
            State {
                output: Some(output),
                history: Vec::new(),
            },
        )?;
        Ok(())
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        let state = self
            .states
            .remove(&state)
            .ok_or_else(|| Error::invariant("release state missing"))?;
        if let Some(output) = state.output {
            self.outputs.push(output);
        }
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
        let mut batch = self
            .batches
            .pop()
            .ok_or_else(|| Error::invariant("batch credit missing"))?;
        for task in tasks {
            let state = self
                .states
                .get_mut(&task.state)
                .ok_or_else(|| Error::invariant("submit state missing"))?;
            let mut output = state
                .output
                .take()
                .ok_or_else(|| Error::invariant("readback still leased"))?;
            // The output must carry one hidden row per token in the sequence, because the engine
            // projects a hidden output only when the rows match the context it planned for.
            task.tokens.commit(&mut state.history)?;
            let rows = state.history.len();
            let readout = task.tokens.readout();
            output.logits.resize(
                if readout == OutputReadout::None {
                    0
                } else {
                    self.vocabulary
                },
                1.0,
            );
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
    fn recycle_output(&mut self, state: StateId, output: ModelOutput) -> Result<()> {
        self.states
            .get_mut(&state)
            .ok_or_else(|| Error::invariant("recycle owner missing"))?
            .output = Some(output);
        Ok(())
    }
    fn recycle_batch(&mut self, mut outputs: Vec<TaskOutput>) -> Result<()> {
        outputs.clear();
        self.batches.push(outputs);
        Ok(())
    }
    fn poll(&mut self, ticket: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>> {
        Ok(ticket.take())
    }
}
