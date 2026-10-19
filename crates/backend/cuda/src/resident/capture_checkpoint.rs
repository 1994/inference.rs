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
        let _ = self.checkpoint_state(node, row, snapshot)?;
        self.arena.buffers[index] = Some(
            Arc::try_unwrap(shared)
                .map_err(|_| error("checkpoint activation row tensor still aliased"))?,
        );
        Ok(())
    }

    /// Flat verification lanes own their activation slots; return that allocation
    /// after recording the same state writer used by pooled checkpoint rows.
    pub(super) fn record_flat_checkpoint(
        &mut self,
        node: &TensorNode,
        snapshot: &mut [Tensor<f32>],
    ) -> Result<(), DeviceError> {
        if self.mode != CaptureMode::Flat || node.outputs.len() != 1 || node.states.len() != 1 {
            return Err(error("checkpoint flat capture bindings"));
        }
        let index = self.arena.slot(node.outputs[0]).map_err(error)?;
        let output = self.arena.buffers[index]
            .take()
            .ok_or_else(|| error("missing checkpoint output buffer"))?;
        let shape = output
            .shape()
            .iter()
            .map(|&dimension| usize::try_from(dimension).map_err(error))
            .collect::<Result<Vec<_>, _>>()?;
        let size = output.size();
        let output = self.checkpoint_state(node, output.reshape(&[size])?, snapshot)?;
        self.arena.buffers[index] = Some(output.reshape(&shape)?);
        Ok(())
    }

    fn checkpoint_state(
        &mut self,
        node: &TensorNode,
        output: Tensor<f32>,
        snapshot: &mut [Tensor<f32>],
    ) -> Result<Tensor<f32>, DeviceError> {
        let id = node.states[0];
        let mut state = self
            .states
            .remove(&id)
            .ok_or_else(|| error("missing checkpoint state allocation"))?;
        let output = match node.op {
            TensorOp::Conv {
                channels,
                kernel: crate::constants::CONV_KERNEL_SIZE,
            } => self.checkpoint_conv(node, output, &mut state, snapshot, channels)?,
            TensorOp::Delta {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
            } => self.checkpoint_delta(
                node,
                output,
                &mut state,
                snapshot,
                (key_heads, value_heads, key_dim, value_dim),
            )?,
            _ => return Err(error("unsupported checkpoint state operation")),
        };
        self.states.insert(id, state);
        Ok(output)
    }

    fn checkpoint_conv(
        &self,
        node: &TensorNode,
        mut output: Tensor<f32>,
        state: &mut [Tensor<f32>],
        snapshot: &mut [Tensor<f32>],
        channels: usize,
    ) -> Result<Tensor<f32>, DeviceError> {
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
        Ok(output)
    }

    fn checkpoint_delta(
        &self,
        node: &TensorNode,
        output: Tensor<f32>,
        state: &mut [Tensor<f32>],
        snapshot: &mut [Tensor<f32>],
        (key_heads, value_heads, key_dim, value_dim): (usize, usize, usize, usize),
    ) -> Result<Tensor<f32>, DeviceError> {
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
        Ok(output)
    }
}

/// Bindings of one lane of [`record_delta3`]: its output, rollback snapshot and row inputs
/// (`qkv`, `beta`, `alpha`, metadata).
pub(super) struct DeltaLane<'a> {
    pub output: &'a mut Tensor<f32>,
    pub snapshot: &'a mut Tensor<f32>,
    pub rows: [TensorView<'a, f32>; 3],
    pub metadata: &'a Tensor<i32>,
}

/// Record the three verification lanes' delta-rule state writes as one launch: the state block
/// is read once and every lane writes its own output and snapshot.
pub(super) fn record_delta3(
    scope: &Scope,
    state: &mut Tensor<f32>,
    lanes: [DeltaLane<'_>; 3],
    (a_log, bias): (&TensorView<'_, f32>, &TensorView<'_, f32>),
    (key_heads, value_heads, key_dim, value_dim): (usize, usize, usize, usize),
) -> Result<(), DeviceError> {
    if key_dim != value_dim
        || !key_dim.is_power_of_two()
        || key_heads == 0
        || !value_heads.is_multiple_of(key_heads)
    {
        return Err(error("checkpoint delta dimensions"));
    }
    let block = value_block(value_dim)?;
    let [first, second, third] = lanes;
    scope.record(
        checkpoint::delta_checkpoint3(
            first.output.partition([1, 1, block]),
            second.output.partition([1, 1, block]),
            third.output.partition([1, 1, block]),
            state.partition([1, key_dim, block]),
            first.snapshot.partition([1, key_dim, block]),
            second.snapshot.partition([1, key_dim, block]),
            third.snapshot.partition([1, key_dim, block]),
            &first.rows[0],
            &second.rows[0],
            &third.rows[0],
            &first.rows[1],
            &second.rows[1],
            &third.rows[1],
            &first.rows[2],
            &second.rows[2],
            &third.rows[2],
            a_log,
            bias,
            first.metadata,
            second.metadata,
            third.metadata,
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

#[cfg(not(test))]
pub(super) const fn flat_checkpoint_enabled() -> bool {
    true
}

#[cfg(test)]
pub(super) fn flat_checkpoint_enabled() -> bool {
    !LEGACY_FLAT_CAPTURE.get()
}

#[cfg(test)]
thread_local! {
    pub(super) static LEGACY_FLAT_CAPTURE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

#[cfg(test)]
#[path = "../../tests/unit/resident_capture_checkpoint.rs"]
mod tests;
