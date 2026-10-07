//! Parallel causal convolution across a prompt chunk, followed by one state commit.
#[cutile::module]
pub(crate) mod kernels {
    use cutile::core::{
        BroadcastScalar, LoadTileLike, Reshape, Shape_0, Shape_1, Shape_2, StoreTileAtCurrentBlock,
        Tensor_1, Tensor_2, Tile_1, exp, get_tile_block_id, tile_to_scalar,
    };
    #[expect(
        clippy::useless_let_if_seq,
        reason = "cuTile DSL uses statement control flow"
    )]
    #[cutile::entry()]
    fn forward(
        out: &mut Tensor<f32, { [1, 128] }>,
        h0: &Tensor<f32, { [-1] }>,
        h1: &Tensor<f32, { [-1] }>,
        h2: &Tensor<f32, { [-1] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        w: &Tensor<f32, { [-1, 4] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let meta = metadata.partition(shape![1]);
        let base: i32 = tile_to_scalar(meta.load([0i32]).reshape(shape![]));
        let count: i32 = tile_to_scalar(meta.load([1i32]).reshape(shape![]));
        let offset: i32 = tile_to_scalar(meta.load([2i32]).reshape(shape![]));
        let mut first = 0i32;
        if base + offset < 0 {
            first = 0i32 - base - offset;
        }
        let lane = pid.0;
        if lane >= first && lane < count {
            let xp = x.partition(shape![1, 128]);
            let mut oldest = h0.partition(shape![128]).load([pid.1]);
            let mut middle = h1.partition(shape![128]).load([pid.1]);
            let mut newest = h2.partition(shape![128]).load([pid.1]);
            if lane >= first + 3i32 {
                oldest = xp.load([lane - 3i32, pid.1]).reshape(shape![128]);
            } else if lane == first + 2i32 {
                oldest = newest;
            } else if lane == first + 1i32 {
                oldest = middle;
            }
            if lane >= first + 2i32 {
                middle = xp.load([lane - 2i32, pid.1]).reshape(shape![128]);
            } else if lane == first + 1i32 {
                middle = newest;
            }
            if lane >= first + 1i32 {
                newest = xp.load([lane - 1i32, pid.1]).reshape(shape![128]);
            }
            let value = xp.load([lane, pid.1]).reshape(shape![128]);
            let wp = w.partition(shape![128, 1]);
            let wa = wp.load([pid.1, 0i32]).reshape(shape![128]);
            let wb = wp.load([pid.1, 1i32]).reshape(shape![128]);
            let wc = wp.load([pid.1, 2i32]).reshape(shape![128]);
            let wd = wp.load([pid.1, 3i32]).reshape(shape![128]);
            let value = oldest * wa + middle * wb + newest * wc + value * wd;
            let negative: Tile<f32, { [128] }> = 0.0f32.broadcast(shape![128]) - value;
            out.store(
                (value / (1.0f32.broadcast(shape![128]) + exp(negative))).reshape(shape![1, 128]),
            );
        } else {
            out.store(0.0f32.broadcast(shape![1, 128]));
        }
    }
    #[expect(
        clippy::useless_let_if_seq,
        reason = "cuTile DSL uses statement control flow"
    )]
    #[cutile::entry()]
    fn commit(
        h0: &mut Tensor<f32, { [128] }>,
        h1: &mut Tensor<f32, { [128] }>,
        h2: &mut Tensor<f32, { [128] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let meta = metadata.partition(shape![1]);
        let base: i32 = tile_to_scalar(meta.load([0i32]).reshape(shape![]));
        let count: i32 = tile_to_scalar(meta.load([1i32]).reshape(shape![]));
        let offset: i32 = tile_to_scalar(meta.load([2i32]).reshape(shape![]));
        let mut first = 0i32;
        if base + offset < 0 {
            first = 0i32 - base - offset;
        }
        if count > first {
            let xp = x.partition(shape![1, 128]);
            let mut oldest = h1.load_like(h0);
            let mut middle = h2.load_like(h0);
            if count - first >= 3i32 {
                oldest = xp.load([count - 3i32, pid.0]).reshape(shape![128]);
            } else if count - first == 2i32 {
                oldest = middle;
            }
            if count - first >= 2i32 {
                middle = xp.load([count - 2i32, pid.0]).reshape(shape![128]);
            }
            h0.store(oldest);
            h1.store(middle);
            h2.store(xp.load([count - 1i32, pid.0]).reshape(shape![128]));
        }
    }
}

use super::{ActivationArena, ProgramWeights, batch::States, capture::error};
use cutile::prelude::*;
use infer_ir::{TensorNode, TensorOp};

pub(super) fn record(
    scope: &Scope,
    node: &TensorNode,
    arena: &mut ActivationArena,
    weights: &ProgramWeights,
    states: &mut States,
    metadata: &Tensor<i32>,
    lanes: usize,
) -> Result<bool, DeviceError> {
    #[cfg(test)]
    if super::recurrent_prefill::LEGACY_CAPTURE.get() {
        return Ok(false);
    }
    let TensorOp::Conv {
        channels,
        kernel: crate::constants::CONV_KERNEL_SIZE,
    } = node.op
    else {
        return Ok(false);
    };
    if weights.constants.contains_key(&node.inputs[0]) {
        return Ok(false);
    }
    let state = states
        .get_mut(&node.states[0])
        .ok_or_else(|| error("prefill convolution state"))?;
    let [h0, h1, h2] = state.as_mut_slice() else {
        return Err(error("prefill convolution layout"));
    };
    let slot = arena.slot(node.outputs[0]).map_err(error)?;
    let output = arena.buffers[slot]
        .take()
        .ok_or_else(|| error("prefill convolution output"))?;
    let original: Vec<usize> = output
        .shape()
        .iter()
        .map(|&n| usize::try_from(n).map_err(error))
        .collect::<Result<_, _>>()?;
    let mut output = output.reshape(&[lanes, channels])?;
    let input = arena
        .get(node.inputs[0])
        .map_err(error)?
        .view(&[lanes, channels])
        .map_err(error)?;
    let weight = weights
        .constants
        .get(&node.inputs[1])
        .ok_or_else(|| error("prefill convolution weight"))?
        .view(&[channels, crate::constants::CONV_KERNEL_SIZE])
        .map_err(error)?;
    scope.record(kernels::forward(
        (&mut output).partition([1, crate::constants::CONV_KERNEL_TILE]),
        &*h0,
        &*h1,
        &*h2,
        &input,
        &weight,
        metadata,
    ))?;
    scope.record(kernels::commit(
        h0.partition([crate::constants::CONV_KERNEL_TILE]),
        h1.partition([crate::constants::CONV_KERNEL_TILE]),
        h2.partition([crate::constants::CONV_KERNEL_TILE]),
        &input,
        metadata,
    ))?;
    arena.buffers[slot] = Some(output.reshape(&original)?);
    Ok(true)
}

#[cfg(test)]
mod tests;
