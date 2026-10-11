//! State writes and their rollback snapshots share the original per-lane launch geometry.
use super::{
    capture::{Capture, CaptureMode, error},
    capture_state::value_block,
    recurrent_checkpoint::checkpoint,
};
use cutile::prelude::*;
use infer_ir::{TensorNode, TensorOp};

impl Capture<'_> {
    /// Record only a conv/delta row that has a rollback snapshot. Other row captures
    /// keep their original entry point and kernels.
    pub(super) fn record_row_checkpoint(
        &mut self,
        node: &TensorNode,
        lane: usize,
        snapshot: &mut [Tensor<f32>],
    ) -> Result<(), DeviceError> {
        if self.mode != CaptureMode::Row(lane) || node.outputs.len() != 1 || node.states.len() != 1
        {
            return Err(error("checkpoint row capture bindings"));
        }
        let index = self.arena.slot(node.outputs[0]).map_err(error)?;
        let slot = self.arena.buffers[index]
            .take()
            .ok_or_else(|| error("missing checkpoint output buffer"))?;
        let (row, shared) =
            super::arena::row_tensor(slot, self.arena.lanes(), lane).map_err(error)?;
        let id = node.states[0];
        let mut state = self
            .states
            .remove(&id)
            .ok_or_else(|| error("missing checkpoint state allocation"))?;
        match node.op {
            TensorOp::Conv {
                channels,
                kernel: crate::constants::CONV_KERNEL_SIZE,
            } => self.checkpoint_conv(node, row, &mut state, snapshot, channels)?,
            TensorOp::Delta {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
            } => self.checkpoint_delta(
                node,
                row,
                &mut state,
                snapshot,
                (key_heads, value_heads, key_dim, value_dim),
            )?,
            _ => return Err(error("unsupported checkpoint state operation")),
        }
        self.states.insert(id, state);
        self.arena.buffers[index] = Some(
            Arc::try_unwrap(shared)
                .map_err(|_| error("checkpoint activation row tensor still aliased"))?,
        );
        Ok(())
    }

    fn checkpoint_conv(
        &self,
        node: &TensorNode,
        mut output: Tensor<f32>,
        state: &mut [Tensor<f32>],
        snapshot: &mut [Tensor<f32>],
        channels: usize,
    ) -> Result<(), DeviceError> {
        let [h0, h1, h2] = state else {
            return Err(error("checkpoint convolution state layout"));
        };
        let [c0, c1, c2] = snapshot else {
            return Err(error("checkpoint convolution snapshot layout"));
        };
        self.scope.record(checkpoint::conv4_checkpoint(
            (&mut output).partition([crate::constants::CONV_KERNEL_TILE]),
            h0.partition([crate::constants::CONV_KERNEL_TILE]),
            h1.partition([crate::constants::CONV_KERNEL_TILE]),
            h2.partition([crate::constants::CONV_KERNEL_TILE]),
            c0.partition([crate::constants::CONV_KERNEL_TILE]),
            c1.partition([crate::constants::CONV_KERNEL_TILE]),
            c2.partition([crate::constants::CONV_KERNEL_TILE]),
            &self.input(node.inputs[0])?,
            &self
                .input(node.inputs[1])?
                .view(&[channels, crate::constants::CONV_KERNEL_SIZE])?,
            self.metadata,
        ))?;
        Ok(())
    }

    fn checkpoint_delta(
        &self,
        node: &TensorNode,
        output: Tensor<f32>,
        state: &mut [Tensor<f32>],
        snapshot: &mut [Tensor<f32>],
        (key_heads, value_heads, key_dim, value_dim): (usize, usize, usize, usize),
    ) -> Result<(), DeviceError> {
        if key_dim != value_dim
            || !key_dim.is_power_of_two()
            || key_heads == 0
            || !value_heads.is_multiple_of(key_heads)
        {
            return Err(error("checkpoint delta dimensions"));
        }
        let [state] = state else {
            return Err(error("checkpoint delta state layout"));
        };
        let [snapshot] = snapshot else {
            return Err(error("checkpoint delta snapshot layout"));
        };
        let block = value_block(value_dim)?;
        let mut output = output.reshape(&[value_heads, 1, value_dim])?;
        self.scope.record(
            checkpoint::delta_checkpoint(
                (&mut output).partition([1, 1, block]),
                state.partition([1, key_dim, block]),
                snapshot.partition([1, key_dim, block]),
                &self.input(node.inputs[0])?,
                &self.input(node.inputs[1])?,
                &self.input(node.inputs[2])?,
                &self.input(node.inputs[3])?,
                &self.input(node.inputs[4])?,
                self.metadata,
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
}
