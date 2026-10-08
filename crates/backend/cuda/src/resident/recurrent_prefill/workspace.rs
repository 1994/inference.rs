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
        // Persistent-state reduction changes FP32 rounding slightly. Activation
        // quantization can amplify that difference at later block-scale thresholds;
        // keep the established recurrence until the whole-model gate passes there.
        // Measured on the 27B NVFP4 with `checkpoint_chunk_recurrence_matches_legacy_logits`
        // (`INFER_TEST_MODEL`/`INFER_TEST_TOKENS`, width 128): the register-resident recurrence
        // drifts `hidden relative_l2 = 0.0436, max_abs = 0.499` against the per-lane one. That is
        // a semantic divergence, not the FP32 re-association it was assumed to be, so this is a
        // correctness gate and not a precision trade. Forcing the convolution back to per-lane
        // leaves the drift byte-identical, which puts the defect in the chunked Delta itself even
        // though `chunk_delta_matches_independent_recurrence` passes the same geometry at 2e-5.
        if !weights.input_scales.is_empty() || !weights.fp8_inputs.is_empty() {
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
