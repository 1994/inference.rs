//! Shared head-major scratch for token-chunk recurrent prefill.
use super::super::capture::error;
use super::kernels;
use crate::{
    device::{CudaDevice, device_error},
    resident::{ActivationArena, ProgramWeights},
};
use cutile::prelude::*;
use infer_core::Result;
use infer_ir::{DataflowGraph, TensorNode, TensorOp};
use std::collections::BTreeMap;

#[derive(Default)]
pub struct Workspace {
    buffers: BTreeMap<(usize, usize), Tensor<f32>>,
    lanes: usize,
}
impl Workspace {
    pub(crate) fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        lanes: usize,
    ) -> Result<Self> {
        // A chunk rotates the recurrent state through one kernel launch instead of once per
        // token, which is worth 24% of long-prompt TTFT on the 27B. The two implementations are
        // separate kernels and lower slightly differently: with ONE lane, byte-identical inputs
        // and the same expression tree, 4634 of 6144 outputs and 59412 of 786432 state elements
        // differ in the last bits (worst 2.98e-8 state, 7.45e-9 output, measured by
        // `chunk_and_per_lane_delta_gap_map`). It is not the lane loop, not the load shape
        // (flattening q/k/v/beta to the per-lane kernel's `[D]`/`[1]` partitions leaves the figure
        // identical), not the masked-lane store, and not the capture binding. So it is instruction
        // contraction inside cuTile's lowering, and exact agreement needs a compiler-level knob
        // rather than a kernel edit.
        //
        // 64 layers plus activation quantization amplify one ULP into a different greedy
        // continuation, so the established recurrence stays the default and a quantized checkpoint
        // only drafts in chunks when a caller asks for it. On the 27B the opt-in measures long
        // TTFT -24%, long wall -10%, long TPOT -4%, and 10-24% better TTFT everywhere, while
        // short and hot_long give back 8-10% of wall and TPOT.
        let opted_in = std::env::var_os("INFER_CUDA_CHUNKED_RECURRENT").is_some();
        if !opted_in && (!weights.input_scales.is_empty() || !weights.fp8_inputs.is_empty()) {
            return Ok(Self {
                lanes,
                ..Self::default()
            });
        }
        let mut buffers = BTreeMap::new();
        for node in &graph.nodes {
            if let TensorOp::Delta {
                value_heads,
                value_dim,
                ..
            } = node.op
                && !buffers.contains_key(&(value_heads, value_dim))
            {
                buffers.insert(
                    (value_heads, value_dim),
                    api::zeros::<f32>(&[value_heads, lanes, value_dim])
                        .sync_on(&device.stream)
                        .map_err(device_error)?,
                );
            }
        }
        Ok(Self { buffers, lanes })
    }
    pub(crate) fn record(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        arena: &mut ActivationArena,
        weights: &ProgramWeights,
        states: &mut super::super::batch::States,
        metadata: &Tensor<i32>,
    ) -> std::result::Result<bool, DeviceError> {
        #[cfg(test)]
        if super::LEGACY_CAPTURE.get() {
            return Ok(false);
        }
        if super::super::conv_prefill::record(
            scope, node, arena, weights, states, metadata, self.lanes,
        )? {
            return Ok(true);
        }
        if self.buffers.is_empty() {
            return Ok(false);
        }
        let TensorOp::Delta {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        } = node.op
        else {
            return Ok(false);
        };
        if key_heads == 0
            || !value_heads.is_multiple_of(key_heads)
            || key_dim != value_dim
            || !key_dim.is_power_of_two()
        {
            return Err(error("chunk delta dimensions"));
        }
        if node.inputs[..crate::constants::GDN_PROJECTED_INPUTS]
            .iter()
            .any(|id| weights.constants.contains_key(id))
        {
            return Ok(false);
        }
        let state = states
            .get_mut(&node.states[0])
            .and_then(|s| s.first_mut())
            .ok_or_else(|| error("chunk delta state"))?;
        let scratch = self
            .buffers
            .get_mut(&(value_heads, value_dim))
            .ok_or_else(|| error("chunk delta scratch"))?;
        let slot = arena.slot(node.outputs[0]).map_err(error)?;
        let output = arena.buffers[slot]
            .take()
            .ok_or_else(|| error("chunk delta output"))?;
        let original: Vec<usize> = output
            .shape()
            .iter()
            .map(|&x| usize::try_from(x).map_err(error))
            .collect::<std::result::Result<_, _>>()?;
        let mut output = output.reshape(&[self.lanes, value_heads, value_dim])?;
        scope.record(
            kernels::delta(
                state.partition([1, key_dim, value_dim]),
                (&mut *scratch).partition([1, self.lanes, value_dim]),
                &arena.get(node.inputs[0]).map_err(error)?.view(&[
                    self.lanes,
                    2 * key_heads + value_heads,
                    key_dim,
                ])?,
                &arena
                    .get(node.inputs[1])
                    .map_err(error)?
                    .view(&[self.lanes, value_heads])?,
                &arena
                    .get(node.inputs[2])
                    .map_err(error)?
                    .view(&[self.lanes, value_heads])?,
                weights
                    .constants
                    .get(&node.inputs[3])
                    .ok_or_else(|| error("chunk delta decay weight"))?,
                weights
                    .constants
                    .get(&node.inputs[4])
                    .ok_or_else(|| error("chunk delta bias"))?,
                metadata,
            )
            .generics(vec![
                key_heads.to_string(),
                value_heads.to_string(),
                key_dim.to_string(),
                self.lanes.to_string(),
            ]),
        )?;
        scope.record(
            kernels::transpose((&mut output).partition([1, 1, value_dim]), &*scratch)
                .generics(vec![value_dim.to_string()]),
        )?;
        arena.buffers[slot] = Some(output.reshape(&original)?);
        Ok(true)
    }
}
