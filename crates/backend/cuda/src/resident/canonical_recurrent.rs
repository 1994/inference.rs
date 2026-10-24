//! Exact target recurrence through the established single-step kernel.
//! One scratch state per geometry is reused across nodes/slots; persistent states stay untouched.
use super::{
    ActivationArena, ProgramWeights,
    batch::States,
    capture::error,
    capture_state::value_block,
    recurrent::recurrent,
    slot_recurrent::{LANES, SlotLanes},
    slot_verify_attention::attention_output_rows,
};
use crate::{
    constants::F32_BYTES,
    device::{CudaDevice, device_error},
};
use cutile::prelude::*;
use infer_core::{Error, Result, TensorId};
use infer_ir::{DataflowGraph, TensorNode, TensorOp};
use std::{
    collections::{BTreeMap, BTreeSet},
    result::Result as StdResult,
};
/// Delta binds qkv, beta, alpha, decay weights and bias.
const DELTA_INPUTS: usize = 5;
type Shape = (usize, usize, usize);
struct Scratch {
    state: Tensor<f32>,
    output: Tensor<f32>,
}
#[derive(Default)]
pub(super) struct Workspace {
    buffers: BTreeMap<Shape, Scratch>,
}
const fn shape(node: &TensorNode) -> Option<Shape> {
    if let TensorOp::Delta {
        value_heads,
        key_dim,
        value_dim,
        ..
    } = node.op
    {
        Some((value_heads, key_dim, value_dim))
    } else {
        None
    }
}
pub(super) fn required_bytes(graph: &DataflowGraph) -> Result<u64> {
    let shapes: BTreeSet<_> = graph.nodes.iter().filter_map(shape).collect();
    shapes
        .into_iter()
        .try_fold(0_u64, |sum, (heads, key, value)| {
            let elements = heads
                .checked_mul(value)
                .and_then(|n| key.checked_add(1).and_then(|k| n.checked_mul(k)))
                .and_then(|n| n.checked_mul(F32_BYTES))
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| Error::invalid("canonical recurrence scratch overflow"))?;
            sum.checked_add(elements)
                .ok_or_else(|| Error::invalid("canonical recurrence scratch overflow"))
        })
}
fn input<'a>(
    weights: &'a ProgramWeights,
    arena: &'a ActivationArena,
    id: TensorId,
    lane: usize,
) -> StdResult<TensorView<'a, f32>, DeviceError> {
    weights.constants.get(&id).map_or_else(
        || arena.row(id, lane).map_err(error),
        |value| value.view(&[value.size()]).map_err(error),
    )
}
fn delta(
    scope: &Scope,
    node: &TensorNode,
    output: &mut Tensor<f32>,
    state: &mut Tensor<f32>,
    inputs: &[TensorView<'_, f32>; DELTA_INPUTS],
    metadata: &Tensor<i32>,
) -> StdResult<(), DeviceError> {
    let TensorOp::Delta {
        key_heads,
        value_heads,
        key_dim,
        value_dim,
    } = node.op
    else {
        return Err(error("canonical Delta node"));
    };
    let block = value_block(value_dim)?;
    scope.record(
        recurrent::delta(
            output.partition([1, 1, block]),
            state.partition([1, key_dim, block]),
            &inputs[0],
            &inputs[1],
            &inputs[2],
            &inputs[3],
            &inputs[4],
            metadata,
        )
        .generics(vec![
            key_heads.to_string(),
            value_heads.to_string(),
            key_dim.to_string(),
            block.to_string(),
        ]),
    )?;
    Ok(())
}
impl Workspace {
    pub(super) fn new(device: &CudaDevice, graph: &DataflowGraph, enabled: bool) -> Result<Self> {
        let mut result = Self::default();
        if !enabled {
            return Ok(result);
        }
        for geometry in graph.nodes.iter().filter_map(shape) {
            if result.buffers.contains_key(&geometry) {
                continue;
            }
            let (heads, key, value) = geometry;
            let state = api::zeros::<f32>(&[heads, key, value])
                .sync_on(&device.stream)
                .map_err(device_error)?;
            let output = api::zeros::<f32>(&[heads, 1, value])
                .sync_on(&device.stream)
                .map_err(device_error)?;
            result.buffers.insert(geometry, Scratch { state, output });
        }
        Ok(result)
    }
    pub(super) fn enabled(&self) -> bool {
        !self.buffers.is_empty()
    }
    pub(super) fn record_verify(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        arena: &mut ActivationArena,
        states: &[States],
        metadata: &[Tensor<i32>],
        weights: &ProgramWeights,
    ) -> StdResult<bool, DeviceError> {
        let Some(geometry) = shape(node) else {
            return Ok(false);
        };
        let Some(scratch) = self.buffers.get_mut(&geometry) else {
            return Ok(false);
        };
        let (heads, _, value) = geometry;
        let output_slot = arena.slot(node.outputs[0]).map_err(error)?;
        let output = arena.buffers[output_slot]
            .take()
            .ok_or_else(|| error("canonical verify output"))?;
        let (rows, owner) = attention_output_rows(output, metadata.len(), heads, value)?;
        for (lane, row) in rows.into_iter().enumerate() {
            if lane % LANES == 0 {
                let source = &states[lane / LANES][&node.states[0]][0];
                scope.record(api::memcpy(&mut scratch.state, source))?;
            }
            let inputs = [
                input(weights, arena, node.inputs[0], lane)?,
                input(weights, arena, node.inputs[1], lane)?,
                input(weights, arena, node.inputs[2], lane)?,
                input(weights, arena, node.inputs[3], lane)?,
                input(weights, arena, node.inputs[4], lane)?,
            ];
            let mut row = row.reshape(&[heads, 1, value])?;
            delta(
                scope,
                node,
                &mut row,
                &mut scratch.state,
                &inputs,
                &metadata[lane],
            )?;
        }
        arena.buffers[output_slot] = Some(owner);
        Ok(true)
    }
    pub(super) fn record_commit(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        (arena, weights): (&ActivationArena, &ProgramWeights),
        states: &mut States,
        slot: &SlotLanes<'_>,
        (staged, count): (&[[Tensor<f32>; LANES]], usize),
    ) -> StdResult<bool, DeviceError> {
        let Some(geometry) = shape(node) else {
            return Ok(false);
        };
        let Some(scratch) = self.buffers.get_mut(&geometry) else {
            return Ok(false);
        };
        let [qkv, beta, alpha] = staged else {
            return Err(error("canonical commit inputs"));
        };
        let state = states
            .get_mut(&node.states[0])
            .ok_or_else(|| error("canonical commit state"))?;
        let [state] = state.as_mut_slice() else {
            return Err(error("canonical commit state shape"));
        };
        for lane in 0..count {
            let inputs = [
                qkv[lane].view(&[qkv[lane].size()]).map_err(error)?,
                beta[lane].view(&[beta[lane].size()]).map_err(error)?,
                alpha[lane].view(&[alpha[lane].size()]).map_err(error)?,
                input(weights, arena, node.inputs[3], slot.lanes[lane])?,
                input(weights, arena, node.inputs[4], slot.lanes[lane])?,
            ];
            delta(
                scope,
                node,
                &mut scratch.output,
                state,
                &inputs,
                slot.metadata[lane],
            )?;
        }
        Ok(true)
    }
}
