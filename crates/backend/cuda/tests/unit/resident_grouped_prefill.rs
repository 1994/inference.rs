//! Test harness for node-major scheduling; every original kernel and private state is reused.
use super::{BatchBuilder, BatchGraph, BatchOutput, take_result};
use crate::device::{CudaDevice, ReadbackSource, device_error};
use crate::resident::{ActivationArena, DeviceProgram, ProgramWeights};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_ir::DataflowGraph;
use std::collections::BTreeMap;

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

pub struct Grouped {
    graph: CudaGraph<()>,
    sequential: CudaGraph<()>,
    readbacks: crate::device::Readbacks,
    frames: Vec<Frame>,
}

impl Grouped {
    pub(crate) fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        programs: &mut [DeviceProgram],
        width: usize,
    ) -> Result<Self> {
        if !(2..=4).contains(&programs.len())
            || ![64, 256].contains(&width)
            || weights.fusion.is_some()
        {
            return Err(Error::unsupported("node-major group geometry"));
        }
        let mut builders: Vec<_> = programs
            .iter_mut()
            .map(|p| p.grouped_test_builder(graph, weights, width))
            .collect();
        let mut frames = builders
            .iter_mut()
            .map(Frame::new)
            .collect::<Result<Vec<_>>>()?;
        let sequential = capture_order(device, graph, weights, &mut builders, &mut frames, false)?;
        let captured = capture_order(device, graph, weights, &mut builders, &mut frames, true)?;
        for frame in &mut frames {
            frame.hidden = Some(take_result(&mut frame.arena, graph.hidden)?);
            frame.logits = Some(take_result(&mut frame.arena, graph.logits)?);
        }
        Ok(Self {
            graph: captured,
            sequential,
            readbacks: crate::device::Readbacks::default(),
            frames,
        })
    }

    pub(crate) fn run(
        &mut self,
        device: &CudaDevice,
        inputs: &[(&[u32], usize)],
        read_hidden: bool,
    ) -> Result<Vec<BatchOutput>> {
        self.run_order(device, inputs, read_hidden, false)
    }
    pub(crate) fn run_sequential(
        &mut self,
        device: &CudaDevice,
        inputs: &[(&[u32], usize)],
        read_hidden: bool,
    ) -> Result<Vec<BatchOutput>> {
        self.run_order(device, inputs, read_hidden, true)
    }
    fn run_order(
        &mut self,
        device: &CudaDevice,
        inputs: &[(&[u32], usize)],
        read_hidden: bool,
        sequential: bool,
    ) -> Result<Vec<BatchOutput>> {
        let graph = if sequential {
            &self.sequential
        } else {
            &self.graph
        };
        if inputs.len() != self.frames.len() {
            return Err(Error::invalid("group input count"));
        }
        let mut sources = Vec::with_capacity(inputs.len() * 2);
        for (frame, (tokens, position)) in self.frames.iter_mut().zip(inputs) {
            if tokens.is_empty()
                || tokens.len() > frame.width
                || position
                    .checked_add(tokens.len())
                    .is_none_or(|end| end > frame.capacity)
            {
                return Err(Error::invalid("group prompt extent"));
            }
            BatchGraph::update_metadata(
                graph,
                &mut frame.metadata,
                frame.width,
                frame.capacity,
                tokens,
                *position,
                0,
            )?;
            crate::resident::metadata::update(
                graph,
                &mut frame.info,
                [
                    i32::try_from(*position).map_err(device_error)?,
                    i32::try_from(tokens.len()).map_err(device_error)?,
                    0,
                    0,
                ],
            )?;
            let hidden = frame
                .hidden
                .as_ref()
                .ok_or_else(|| Error::invariant("group hidden"))?;
            let logits = frame
                .logits
                .as_ref()
                .ok_or_else(|| Error::invariant("group logits"))?;
            let hidden_dim = hidden.size() / frame.width;
            let vocabulary = logits.size() / frame.width;
            sources.push(read_hidden.then(|| ReadbackSource {
                tensor: Arc::clone(hidden),
                skip: 0,
                len: tokens.len() * hidden_dim,
            }));
            sources.push(Some(ReadbackSource {
                tensor: Arc::clone(logits),
                skip: (tokens.len() - 1) * vocabulary,
                len: vocabulary,
            }));
        }
        let mut readouts = self.readbacks.run(device, graph, &sources)?.into_iter();
        let mut outputs = Vec::with_capacity(inputs.len());
        for (frame, (tokens, _)) in self.frames.iter().zip(inputs) {
            let hidden = readouts
                .next()
                .ok_or_else(|| Error::invariant("group hidden readback"))?;
            let mut logits = readouts
                .next()
                .ok_or_else(|| Error::invariant("group logits readback"))?;
            let hidden_dim = frame
                .hidden
                .as_ref()
                .ok_or_else(|| Error::invariant("group hidden"))?
                .size()
                / frame.width;
            let rows = (0..tokens.len())
                .map(|lane| {
                    let h = if read_hidden {
                        hidden[lane * hidden_dim..(lane + 1) * hidden_dim].to_vec()
                    } else {
                        Vec::new()
                    };
                    let l = if lane + 1 == tokens.len() {
                        std::mem::take(&mut logits)
                    } else {
                        Vec::new()
                    };
                    (h, l)
                })
                .collect();
            outputs.push(rows);
        }
        Ok(outputs)
    }
}

fn capture_order(
    device: &CudaDevice,
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    builders: &mut [BatchBuilder<'_>],
    frames: &mut [Frame],
    node_major: bool,
) -> Result<CudaGraph<()>> {
    let plan = crate::resident::nvfp4_gemm::QuantPairPlan::new(graph, weights);
    for builder in builders.iter_mut() {
        builder.nvfp4.begin_quant_capture();
    }
    let result = CudaGraph::scope(&device.stream, |scope| {
        if node_major {
            for (index, node) in graph.nodes.iter().enumerate() {
                for (builder, frame) in builders.iter_mut().zip(frames.iter_mut()) {
                    if !builder.skip_quant(&plan, index, false) {
                        builder.record_prompt_node(
                            scope,
                            node,
                            &mut frame.arena,
                            (&frame.metadata, &frame.info),
                            (&mut frame.partials, &mut frame.recurrent),
                        )?;
                    }
                }
            }
        } else {
            for (builder, frame) in builders.iter_mut().zip(frames.iter_mut()) {
                for (index, node) in graph.nodes.iter().enumerate() {
                    if !builder.skip_quant(&plan, index, false) {
                        builder.record_prompt_node(
                            scope,
                            node,
                            &mut frame.arena,
                            (&frame.metadata, &frame.info),
                            (&mut frame.partials, &mut frame.recurrent),
                        )?;
                    }
                }
            }
        }
        Ok(())
    });
    for builder in builders.iter_mut() {
        builder.nvfp4.end_quant_capture();
    }
    result.map_err(device_error)
}
