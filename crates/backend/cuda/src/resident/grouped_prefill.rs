//! Node-major short prefill over existing independently leased slot states.
use super::{
    ActivationArena, ProgramWeights,
    batch::{BatchBuilder, BatchGraph, BatchOutput, States, take_result},
    fp8_cache::Fp8Caches,
};
use crate::device::{CudaDevice, ReadbackSource, device_error};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_ir::DataflowGraph;
use std::collections::BTreeMap;
const LANES: usize = crate::constants::MID_PREFILL_LANES;
const MAX_SLOTS: usize = 4;
/// Hidden dimension of the measured workload eligible for this capture.
pub(super) const MEASURED_HIDDEN: usize = 5120;

pub(super) struct Bindings<'a> {
    pub states: &'a mut [States],
    pub fp8: &'a mut [Fp8Caches],
    pub tables: &'a [Tensor<i32>],
    pub external: &'a Tensor<f32>,
    pub capacity: usize,
}

struct Context {
    nvfp4: super::nvfp4_gemm::Workspace,
    delegated: crate::device::cublaslt::Support,
    attention: super::attention_decode::Workspace,
    external: Vec<Tensor<f32>>,
    fusion: Vec<Option<super::program::FusionWorkspace>>,
}
impl Context {
    fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        capacity: usize,
        hidden: usize,
    ) -> Result<Self> {
        let mut delegated = crate::device::cublaslt::Support::new(device);
        super::program::warm_delegated(&mut delegated, weights);
        let mut external = Vec::with_capacity(LANES);
        for _ in 0..LANES {
            external.push(
                api::zeros::<f32>(&[hidden])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
            );
        }
        Ok(Self {
            nvfp4: super::nvfp4_gemm::Workspace::program(
                device,
                graph,
                weights,
                super::ProgramGeometry {
                    verification: 0,
                    prompts: [LANES, 0],
                },
            )?,
            delegated,
            attention: super::attention_decode::Workspace::new(device, graph, capacity)?,
            external,
            fusion: (0..LANES).map(|_| None).collect(),
        })
    }
}
struct Frame {
    arena: ActivationArena,
    metadata: Vec<Tensor<i32>>,
    info: Tensor<i32>,
    partials: BTreeMap<usize, Tensor<f32>>,
    recurrent: crate::resident::recurrent_prefill::Workspace,
    hidden: Option<Arc<Tensor<f32>>>,
    logits: Option<Arc<Tensor<f32>>>,
    width: usize,
    capacity: usize,
}

impl Frame {
    fn new(builder: &mut BatchBuilder<'_>) -> Result<Self> {
        builder.attention.prepare_prefill(
            builder.device,
            builder.graph,
            builder.weights,
            builder.width,
        );
        let arena = ActivationArena::new_batched(
            builder.device,
            builder.graph,
            builder.width,
            usize::try_from(builder.device.profile()?.arena_budget_bytes())
                .map_err(device_error)?,
        )?;
        let mut metadata = Vec::with_capacity(builder.width);
        for _ in 0..builder.width {
            metadata.push(
                api::zeros::<i32>(&[crate::constants::METADATA_FIELDS])
                    .sync_on(&builder.device.stream)
                    .map_err(device_error)?,
            );
        }
        Ok(Self {
            arena,
            metadata,
            info: api::zeros::<i32>(&[crate::constants::METADATA_FIELDS])
                .sync_on(&builder.device.stream)
                .map_err(device_error)?,
            partials: builder.allocate_prompt_partials()?,
            recurrent: crate::resident::recurrent_prefill::Workspace::new(
                builder.device,
                builder.graph,
                builder.weights,
                builder.width,
            )?,
            hidden: None,
            logits: None,
            width: builder.width,
            capacity: builder.capacity,
        })
    }
}

pub(super) struct SlotPrefill {
    graphs: BTreeMap<u32, CudaGraph<()>>,
    readbacks: crate::device::Readbacks,
    frames: Vec<Frame>,
    _contexts: Vec<Context>,
}
impl SlotPrefill {
    pub(super) fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        b: Bindings<'_>,
        hidden: usize,
    ) -> Result<Self> {
        let count = b.states.len();
        if !(2..=MAX_SLOTS).contains(&count)
            || b.fp8.len() != count
            || b.tables.len() != count
            || weights.fusion.is_some()
            || hidden != MEASURED_HIDDEN
            || device.profile()?.architecture != "sm_120"
        {
            return Err(Error::unsupported("grouped prefill geometry"));
        }
        let mut contexts = (0..count)
            .map(|_| Context::new(device, graph, weights, b.capacity, hidden))
            .collect::<Result<Vec<_>>>()?;
        let mut builders = Vec::with_capacity(count);
        for (i, ((state, fp8), ctx)) in b
            .states
            .iter_mut()
            .zip(b.fp8)
            .zip(&mut contexts)
            .enumerate()
        {
            builders.push(BatchBuilder {
                device,
                graph,
                weights,
                nvfp4: &mut ctx.nvfp4,
                delegated: Some(&mut ctx.delegated),
                attention: &mut ctx.attention,
                states: state,
                fp8_states: fp8,
                external: b.external,
                lane_external: &mut ctx.external,
                lane_fusion: &mut ctx.fusion,
                tables: &b.tables[i..=i],
                capacity: b.capacity,
                width: LANES,
            });
        }
        let mut frames = builders
            .iter_mut()
            .map(Frame::new)
            .collect::<Result<Vec<_>>>()?;
        let mut graphs = BTreeMap::new();
        for mask in 1u32..(1u32 << count) {
            if mask.count_ones() < 2 {
                continue;
            }
            graphs.insert(
                mask,
                capture(device, graph, weights, &mut builders, &mut frames, mask)?,
            );
        }
        for f in &mut frames {
            f.hidden = Some(take_result(&mut f.arena, graph.hidden)?);
            f.logits = Some(take_result(&mut f.arena, graph.logits)?);
        }
        drop(builders);
        Ok(Self {
            graphs,
            readbacks: crate::device::Readbacks::default(),
            frames,
            _contexts: contexts,
        })
    }
    pub(super) fn run(
        &mut self,
        device: &CudaDevice,
        inputs: &[(usize, &[u32])],
    ) -> Result<BatchOutput> {
        let mut mask = 0u32;
        for (slot, tokens) in inputs {
            if *slot >= self.frames.len()
                || tokens.is_empty()
                || tokens.len() > LANES
                || mask & (1u32 << slot) != 0
            {
                return Err(Error::invalid("grouped prefill slot or tokens"));
            }
            mask |= 1u32 << slot;
        }
        let graph = self
            .graphs
            .get(&mask)
            .ok_or_else(|| Error::unsupported("grouped prefill needs two slots"))?;
        let mut sources = Vec::with_capacity(inputs.len());
        for (slot, tokens) in inputs {
            let f = &mut self.frames[*slot];
            BatchGraph::update_metadata(graph, &mut f.metadata, LANES, f.capacity, tokens, 0, 0)?;
            super::metadata::update(
                graph,
                &mut f.info,
                [0, i32::try_from(tokens.len()).map_err(device_error)?, 0, 0],
            )?;
            let logits = f
                .logits
                .as_ref()
                .ok_or_else(|| Error::invariant("grouped prefill logits"))?;
            let vocabulary = logits.size() / f.width;
            sources.push(Some(ReadbackSource {
                tensor: Arc::clone(logits),
                skip: (tokens.len() - 1) * vocabulary,
                len: vocabulary,
            }));
        }
        self.readbacks.run(device, graph, &sources).map(|rows| {
            rows.into_iter()
                .map(|logits| (Vec::new(), logits))
                .collect()
        })
    }
}
fn capture(
    device: &CudaDevice,
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    builders: &mut [BatchBuilder<'_>],
    frames: &mut [Frame],
    mask: u32,
) -> Result<CudaGraph<()>> {
    let plan = super::nvfp4_gemm::QuantPairPlan::new(graph, weights);
    for (i, b) in builders.iter_mut().enumerate() {
        if mask & (1u32 << i) != 0 {
            b.nvfp4.begin_quant_capture();
        }
    }
    let result = CudaGraph::scope(&device.stream, |scope| {
        for (index, node) in graph.nodes.iter().enumerate() {
            for (i, (b, f)) in builders.iter_mut().zip(frames.iter_mut()).enumerate() {
                if mask & (1u32 << i) != 0 && !b.skip_quant(&plan, index, false) {
                    b.record_prompt_node(
                        scope,
                        node,
                        &mut f.arena,
                        (&f.metadata, &f.info),
                        (&mut f.partials, &mut f.recurrent),
                    )?;
                }
            }
        }
        Ok(())
    });
    for (i, b) in builders.iter_mut().enumerate() {
        if mask & (1u32 << i) != 0 {
            b.nvfp4.end_quant_capture();
        }
    }
    result.map_err(device_error)
}
