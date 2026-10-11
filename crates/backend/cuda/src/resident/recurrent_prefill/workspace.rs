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
        // Keeping state in one kernel reduces long-prompt TTFT in historical opt-in tests,
        // but the scalar and chunk kernels do not evaluate floats bitwise identically.
        // Even with one lane and identical inputs, the isolated gap map found differences
        // in state/output last bits. Neither instruction contraction nor reduction ordering
        // has been isolated as the cause; a compiler knob is not a demonstrated requirement.
        //
        // The checkpoint model check explicitly enables chunked Delta while keeping convolution
        // and GEMM policy identical. On the mixed FP8/NVFP4 27B, small early state differences
        // grow into substantial hidden/logit drift. Keep scalar Delta as the quantized default
        // until the full-model numeric gate passes, rather than accepting a kernel-only tolerance.
        let opted_in =
            weights.chunked_recurrent || std::env::var_os("INFER_CUDA_CHUNKED_RECURRENT").is_some();
        if !opted_in && (!weights.input_scales.is_empty() || !weights.fp8_inputs.is_empty()) {
            return Ok(Self {
                lanes,
                ..Self::default()
            });
        }
        // MoE parity at head width 128 found invalid generated scratch atomics in
        // the chunk delta kernel under Compute Sanitizer. Keep its per-row twin.
        if !weights.expert_banks.is_empty() {
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
        #[cfg(test)]
        if super::LEGACY_DELTA_CAPTURE.get() {
            return Ok(false);
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
        // The value dimension may be split so the grid fills the device and each block moves a
        // quarter of the state per lane. Off by default until the full matrix shows a gain.
        let block = super::super::capture_state::value_block(value_dim)
            .map_err(|_| error("chunk delta value split"))?;
        scope.record(
            kernels::delta(
                state.partition([1, key_dim, block]),
                (&mut *scratch).partition([1, self.lanes, block]),
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
                block.to_string(),
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
