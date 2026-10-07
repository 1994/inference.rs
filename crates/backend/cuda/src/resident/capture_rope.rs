use super::{
    capture::{Capture, CaptureMode, error},
    kernels::aux,
};
use cutile::prelude::*;
use infer_ir::{TensorNode, TensorOp};

impl Capture<'_> {
    pub(super) fn record_rope(
        &self,
        node: &TensorNode,
        mut output: Tensor<f32>,
    ) -> Result<Tensor<f32>, DeviceError> {
        let TensorOp::Rope {
            heads,
            head_dim,
            rotary_dim,
            ..
        } = &node.op
        else {
            return Err(error("expected rotary operation"));
        };
        let half = rotary_dim / 2;
        if !half.is_power_of_two() || !head_dim.is_multiple_of(half) {
            return Err(error("unsupported rotary dimensions"));
        }
        let rows = if self.mode == CaptureMode::Prefill {
            self.arena.lanes() * heads
        } else {
            *heads
        };
        output = output.reshape(&[rows, *head_dim])?;
        let frequency = self
            .weights
            .rope_frequencies
            .get(&node.id)
            .ok_or_else(|| error("missing rotary frequencies"))?;
        if self.mode == CaptureMode::Prefill {
            self.scope.record(
                aux::prefill_rope(
                    (&mut output).partition([1, half]),
                    &self.input(node.inputs[0])?.view(&[rows, *head_dim])?,
                    frequency,
                    self.metadata,
                )
                .generics(vec![
                    head_dim.to_string(),
                    half.to_string(),
                    heads.to_string(),
                ]),
            )?;
        } else if self.metadata.size() == crate::constants::METADATA_FIELDS * 2
            && let Some(axes) = self.weights.rope_axes.get(&node.id)
        {
            self.scope.record(
                aux::mrope(
                    (&mut output).partition([1, half]),
                    &self.input(node.inputs[0])?.view(&[*heads, *head_dim])?,
                    frequency,
                    axes,
                    self.metadata,
                )
                .generics(vec![head_dim.to_string(), half.to_string()]),
            )?;
        } else {
            self.scope.record(
                aux::rope(
                    (&mut output).partition([1, half]),
                    &self.input(node.inputs[0])?.view(&[*heads, *head_dim])?,
                    frequency,
                    self.metadata,
                )
                .generics(vec![head_dim.to_string(), half.to_string()]),
            )?;
        }
        Ok(output)
    }
}
