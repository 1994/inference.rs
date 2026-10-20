//! One rotary launch across verification rows, using each row's original position.
use super::{ActivationArena, capture::error, program::ProgramWeights};
use crate::constants::METADATA_FIELDS;
use cutile::prelude::*;
use infer_ir::{TensorNode, TensorOp};

pub(super) fn record(
    scope: &Scope,
    node: &TensorNode,
    arena: &mut ActivationArena,
    weights: &ProgramWeights,
    metadata: &Tensor<i32>,
) -> Result<bool, DeviceError> {
    let TensorOp::Rope {
        heads,
        head_dim,
        rotary_dim,
        ..
    } = node.op
    else {
        return Ok(false);
    };
    if weights.constants.contains_key(&node.inputs[0])
        || metadata.size() != arena.lanes() * METADATA_FIELDS
    {
        return Ok(false);
    }
    let half = rotary_dim / 2;
    if heads == 0 || !half.is_power_of_two() || !head_dim.is_multiple_of(half) {
        return Err(error("unsupported pooled rotary dimensions"));
    }
    let frequencies = weights
        .rope_frequencies
        .get(&node.id)
        .ok_or_else(|| error("missing pooled rotary frequencies"))?;
    let slot = arena.slot(node.outputs[0]).map_err(error)?;
    let output = arena.buffers[slot]
        .take()
        .ok_or_else(|| error("pooled rotary output"))?;
    let restore = output
        .shape()
        .iter()
        .map(|&dim| usize::try_from(dim).map_err(error))
        .collect::<Result<Vec<_>, _>>()?;
    let mut output = output.reshape(&[arena.lanes() * heads, head_dim])?;
    scope.record(
        kernels::rope(
            (&mut output).partition([1, half]),
            &arena
                .get(node.inputs[0])
                .map_err(error)?
                .view(&[arena.lanes() * heads, head_dim])?,
            frequencies,
            metadata,
        )
        .generics(
            [head_dim, half, heads, METADATA_FIELDS]
                .iter()
                .map(ToString::to_string)
                .collect(),
        ),
    )?;
    arena.buffers[slot] = Some(output.reshape(&restore)?);
    Ok(true)
}

#[cutile::module]
mod kernels {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_0, Shape_1, Shape_2, StoreTileAtCurrentBlock, Tensor_1,
        Tensor_2, Tile_1, convert_tile, cos, get_tile_block_id, sin, tile_to_scalar,
    };

    /// Same expressions as the existing per-lane rotary kernel; only the position binding varies across heads.
    #[cutile::entry()]
    fn rope<const D: i32, const HALF: i32, const HEADS: i32, const MF: i32>(
        out: &mut Tensor<f32, { [1, HALF] }>,
        x: &Tensor<f32, { [-1, D] }>,
        frequencies: &Tensor<f32, { [HALF] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![1, HALF]);
        let value = xp.load([pid.0, pid.1]);
        if pid.1 < 2 {
            let p: Tile<f32, { [1] }> =
                convert_tile(metadata.partition(shape![1]).load([pid.0 / HEADS * MF]));
            let position: f32 = tile_to_scalar(p.reshape(shape![]));
            let angles =
                frequencies.partition(shape![HALF]).load([0i32]) * position.broadcast(shape![HALF]);
            let c = cos(angles).reshape(shape![1, HALF]);
            let s = sin(angles).reshape(shape![1, HALF]);
            let other = xp.load([pid.0, 1i32 - pid.1]);
            if pid.1 == 0 {
                out.store(value * c - other * s);
            } else {
                out.store(value * c + other * s);
            }
        } else {
            out.store(value);
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/resident_slot_rope.rs"]
mod tests;
