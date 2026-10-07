//! Split long KV reductions across independent CTAs before stable softmax merging.
#[cutile::module]
pub(crate) mod kernels {
    use cutile::core::{
        BroadcastScalar, ElementType, Reshape, Shape_0, Shape_1, Shape_2, Shape_3,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tensor_3, Tile_0, Tile_1, Tile_2,
        cmp_ordering, cmpf, constant, convert_scalar, convert_tile, exp, ftz, ge_tile,
        get_tile_block_id, iota, le_tile, max_tile, predicate, reduce_max, reduce_sum, rsqrt,
        select, tile_to_scalar,
    };
    #[expect(
        clippy::useless_let_if_seq,
        reason = "cuTile DSL uses statement control flow and exclusive ranges"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA entry takes explicit tensor bindings and independent KV scales"
    )]
    #[cutile::entry()]
    fn partial<E: ElementType, const D: i32, const GROUP: i32, const PARTS: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
        maxima: &mut Tensor<f32, { [1, 1] }>,
        sums: &mut Tensor<f32, { [1, 1] }>,
        q: &Tensor<f32, { [-1, D] }>,
        keys: &Tensor<E, { [-1, -1, D] }>,
        values: &Tensor<E, { [-1, -1, D] }>,
        metadata: &Tensor<i32, { [-1] }>,
        window: i32,
        k_scale: f32,
        v_scale: f32,
    ) {
        let pid = get_tile_block_id();
        let position: i32 =
            tile_to_scalar(metadata.partition(shape![1]).load([2i32]).reshape(shape![]));
        if position >= 0 {
            let mut start = 0i32;
            if window > 0 && position + 1i32 > window {
                start = position + 1i32 - window;
            }
            let head = pid.0 / PARTS;
            let split = pid.0 % PARTS;
            let query = q
                .partition(shape![1, D])
                .load([head, 0i32])
                .reshape(shape![1, D])
                .broadcast(shape![32, D]);
            let d: f32 = convert_scalar(D);
            let scale: Tile<f32, { [32] }> = rsqrt(d.broadcast(shape![32]), ftz::Disabled);
            let kp = keys.partition(shape![1, 32, D]);
            let vp = values.partition(shape![1, 32, D]);
            let mut numerator: Tile<f32, { [D] }> = constant(0.0f32, shape![D]);
            let mut denominator: Tile<f32, { [1] }> = constant(0.0f32, shape![1]);
            let mut maximum: Tile<f32, { [1] }> = constant(-1.0e30f32, shape![1]);
            let first = start / 32i32;
            let end = position / 32i32 + 1i32;
            let block_count = (end - first + PARTS - 1i32) / PARTS;
            let first = first + split * block_count;
            let mut last = first + block_count;
            if last > end {
                last = end;
            }
            for block in first..last {
                let key: Tile<f32, { [32, D] }> =
                    convert_tile(kp.load([head / GROUP, block, 0i32]).reshape(shape![32, D]));
                let value: Tile<f32, { [32, D] }> =
                    convert_tile(vp.load([head / GROUP, block, 0i32]).reshape(shape![32, D]));
                let key: Tile<f32, { [32, D] }> = key * k_scale.broadcast(shape![32, D]);
                let value: Tile<f32, { [32, D] }> = value * v_scale.broadcast(shape![32, D]);
                let score: Tile<f32, { [32] }> = reduce_sum(query * key, 1i32);
                let score = score * scale;
                let offsets: Tile<i32, { [32] }> = iota(shape![32]);
                let positions = offsets + (block * 32i32).broadcast(shape![32]);
                let valid = ge_tile(positions, start.broadcast(shape![32]))
                    & le_tile(positions, position.broadcast(shape![32]));
                let masked: Tile<f32, { [32] }> = constant(-1.0e30f32, shape![32]);
                let score = select(valid, score, masked);
                let block_max: Tile<f32, { [] }> = reduce_max(score, 0i32);
                let block_max = block_max.reshape(shape![1]);
                let greater = cmpf(
                    block_max,
                    maximum,
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                );
                let next_max: Tile<f32, { [1] }> = select(greater, block_max, maximum);
                let old_scale: Tile<f32, { [1] }> = exp(maximum - next_max);
                let probabilities: Tile<f32, { [32] }> =
                    exp(score - next_max.broadcast(shape![32]));
                let block_sum: Tile<f32, { [] }> = reduce_sum(probabilities, 0i32);
                let contribution: Tile<f32, { [D] }> = reduce_sum(
                    value
                        * probabilities
                            .reshape(shape![32, 1])
                            .broadcast(shape![32, D]),
                    0i32,
                );
                numerator =
                    numerator * old_scale.broadcast(shape![D]) + contribution.reshape(shape![D]);
                denominator = denominator * old_scale + block_sum.reshape(shape![1]);
                maximum = next_max;
            }
            out.store(numerator.reshape(shape![1, D]));
            maxima.store(maximum.reshape(shape![1, 1]));
            sums.store(denominator.reshape(shape![1, 1]));
        } else {
            out.store(0.0f32.broadcast(shape![1, D]));
            maxima.store((-1.0e30f32).broadcast(shape![1, 1]));
            sums.store(0.0f32.broadcast(shape![1, 1]));
        }
    }
    #[cutile::entry()]
    fn merge<const D: i32, const PARTS: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
        numerator: &Tensor<f32, { [-1, PARTS, D] }>,
        maxima: &Tensor<f32, { [-1, PARTS] }>,
        sums: &Tensor<f32, { [-1, PARTS] }>,
    ) {
        let pid = get_tile_block_id();
        let maxima = maxima
            .partition(shape![1, PARTS])
            .load([pid.0, 0i32])
            .reshape(shape![PARTS]);
        let maximum: Tile<f32, { [] }> = reduce_max(maxima, 0i32);
        let weights = exp(maxima - maximum.reshape(shape![1]).broadcast(shape![PARTS]));
        let sums = sums
            .partition(shape![1, PARTS])
            .load([pid.0, 0i32])
            .reshape(shape![PARTS]);
        let total: Tile<f32, { [] }> = reduce_sum(sums * weights, 0i32);
        let total = max_tile(total.reshape(shape![1]), 1.0e-20f32.broadcast(shape![1]));
        let values = numerator
            .partition(shape![1, PARTS, D])
            .load([pid.0, 0i32, 0i32])
            .reshape(shape![PARTS, D]);
        let sum: Tile<f32, { [D] }> = reduce_sum(
            values
                * weights
                    .reshape(shape![PARTS, 1])
                    .broadcast(shape![PARTS, D]),
            0i32,
        );
        out.store((sum / total.broadcast(shape![D])).reshape(shape![1, D]));
    }
}

#[cfg(test)]
mod tests;

mod workspace;
pub use workspace::Workspace;
