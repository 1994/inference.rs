//! Scalar launch arguments update resident metadata without temporary device allocations.
use crate::device::device_error;
use cutile::prelude::*;
use infer_core::Result;

#[cfg(test)]
#[path = "metadata_check.rs"]
mod check;

pub(super) fn update(
    graph: &CudaGraph<()>,
    output: &mut Tensor<i32>,
    values: [i32; crate::constants::METADATA_FIELDS],
) -> Result<()> {
    // CudaGraph::update retains the output lease until replay completes. The replay
    // must use graph.stream(): same-stream ordering replaces the former upload sync.
    // Pass opaque bit patterns through float arguments: cuTile 0.4 automatically
    // specializes integer scalars by DivHint, causing runtime JIT as tokens change.
    // No floating-point arithmetic or conversion is performed on these arguments.
    graph
        .update(crate::device::GraphUpdate(kernels::write(
            output.partition([crate::constants::METADATA_FIELDS]),
            f32::from_ne_bytes(values[0].to_ne_bytes()),
            f32::from_ne_bytes(values[1].to_ne_bytes()),
            f32::from_ne_bytes(values[2].to_ne_bytes()),
            f32::from_ne_bytes(values[3].to_ne_bytes()),
        )))
        .map_err(device_error)
}

/// Update scalar control metadata and independent rotary coordinates on the same stream.
pub(super) fn update_rotary(
    graph: &CudaGraph<()>,
    output: &mut Tensor<i32>,
    values: [i32; crate::constants::METADATA_FIELDS],
    position: [usize; infer_models::mrope::ROTARY_AXES],
) -> Result<()> {
    let bits = |value: i32| f32::from_ne_bytes(value.to_ne_bytes());
    let height = i32::try_from(position[1]).map_err(device_error)?;
    let width = i32::try_from(position[2]).map_err(device_error)?;
    graph
        .update(crate::device::GraphUpdate(kernels::write_rotary(
            output.partition([crate::constants::METADATA_FIELDS * 2]),
            bits(values[0]),
            bits(values[1]),
            bits(values[2]),
            bits(values[3]),
            bits(height),
            bits(width),
        )))
        .map_err(device_error)
}

#[cutile::module]
mod kernels {
    use cutile::core::{
        BroadcastScalar, Shape_1, StoreTileAtCurrentBlock, Tensor_1, Tile_1, bitcast, eq_tile,
        iota, select,
    };

    #[cutile::entry()]
    fn write(out: &mut Tensor<i32, { [4] }>, position: f32, token: f32, kv: f32, external: f32) {
        let offsets: Tile<i32, { [4] }> = iota(shape![4]);
        let position: Tile<i32, { [4] }> = bitcast(position.broadcast(shape![4]));
        let token: Tile<i32, { [4] }> = bitcast(token.broadcast(shape![4]));
        let kv: Tile<i32, { [4] }> = bitcast(kv.broadcast(shape![4]));
        let values: Tile<i32, { [4] }> = bitcast(external.broadcast(shape![4]));
        let values = select(eq_tile(offsets, 2i32.broadcast(shape![4])), kv, values);
        let values = select(eq_tile(offsets, 1i32.broadcast(shape![4])), token, values);
        let values = select(
            eq_tile(offsets, 0i32.broadcast(shape![4])),
            position,
            values,
        );
        out.store(values);
    }
    #[cutile::entry()]
    fn write_rotary(
        out: &mut Tensor<i32, { [8] }>,
        position: f32,
        token: f32,
        kv: f32,
        external: f32,
        height: f32,
        width: f32,
    ) {
        let offsets: Tile<i32, { [8] }> = iota(shape![8]);
        let position_values: Tile<i32, { [8] }> = bitcast(position.broadcast(shape![8]));
        let token_values: Tile<i32, { [8] }> = bitcast(token.broadcast(shape![8]));
        let kv_values: Tile<i32, { [8] }> = bitcast(kv.broadcast(shape![8]));
        let external_values: Tile<i32, { [8] }> = bitcast(external.broadcast(shape![8]));
        let height_values: Tile<i32, { [8] }> = bitcast(height.broadcast(shape![8]));
        let width_values: Tile<i32, { [8] }> = bitcast(width.broadcast(shape![8]));
        let result = select(
            eq_tile(offsets, 1i32.broadcast(shape![8])),
            token_values,
            position_values,
        );
        let result = select(
            eq_tile(offsets, 2i32.broadcast(shape![8])),
            kv_values,
            result,
        );
        let result = select(
            eq_tile(offsets, 3i32.broadcast(shape![8])),
            external_values,
            result,
        );
        let result = select(
            eq_tile(offsets, 5i32.broadcast(shape![8])),
            height_values,
            result,
        );
        let result = select(
            eq_tile(offsets, 6i32.broadcast(shape![8])),
            width_values,
            result,
        );
        out.store(result);
    }
}
