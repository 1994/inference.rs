//! Split long KV reductions across independent CTAs before stable softmax merging.
#[cutile::module]
pub(crate) mod kernels {
    use cutile::core::{
        BroadcastScalar, ElementType, Reshape, Shape_0, Shape_1, Shape_2, Shape_3,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tensor_3, Tile_0, Tile_1, Tile_2,
        Shape_4, Tensor_4, bf16, cmp_ordering, cmpf, constant, convert_scalar, convert_tile, exp,
        ftz, ge_tile, get_tile_block_id, iota, le_tile, max_tile, mmaf, predicate, reduce_max,
        reduce_sum, rsqrt, select, tile_to_scalar,
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
    fn partial<E: ElementType, const D: i32, const GROUP: i32, const PARTS: i32, const BT: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
        maxima: &mut Tensor<f32, { [1, 1] }>,
        sums: &mut Tensor<f32, { [1, 1] }>,
        q: &Tensor<f32, { [-1, D] }>,
        keys: &Tensor<E, { [-1, -1, D] }>,
        values: &Tensor<E, { [-1, -1, D] }>,
        metadata: &Tensor<i32, { [-1] }>,
        table: &Tensor<i32, { [-1] }>,
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
            let blocks = table.partition(shape![1]);
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
                let physical: i32 = tile_to_scalar(blocks.load([block]).reshape(shape![]));
                let key: Tile<f32, { [32, D] }> = convert_tile(
                    kp.load([head / GROUP, physical, 0i32])
                        .reshape(shape![32, D]),
                );
                let value: Tile<f32, { [32, D] }> = convert_tile(
                    vp.load([head / GROUP, physical, 0i32])
                        .reshape(shape![32, D]),
                );
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
    /// Split-KV decode attention for grouped queries: one CTA per (KV head, split) reads each KV
    /// block once and scores every query head with BF16 tensor-core products. Rows of other KV
    /// heads are computed and discarded; [`merge_gqa`] reads only the rows of each head's group.
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA entry takes explicit tensor bindings and independent KV scales"
    )]
    #[cutile::entry()]
    fn partial_gqa<E: ElementType, const D: i32, const GROUP: i32, const PARTS: i32, const BT: i32>(
        out: &mut Tensor<f32, { [1, 32, D] }>,
        maxima: &mut Tensor<f32, { [1, 32] }>,
        sums: &mut Tensor<f32, { [1, 32] }>,
        q: &Tensor<f32, { [-1, D] }>,
        keys: &Tensor<E, { [-1, -1, D] }>,
        values: &Tensor<E, { [-1, -1, D] }>,
        metadata: &Tensor<i32, { [-1] }>,
        table: &Tensor<i32, { [-1] }>,
        window: i32,
        k_scale: f32,
        v_scale: f32,
    ) {
        let pid = get_tile_block_id();
        let kv = pid.0 / PARTS;
        let split = pid.0 % PARTS;
        let position: i32 =
            tile_to_scalar(metadata.partition(shape![1]).load([2i32]).reshape(shape![]));
        let mut start = 0i32;
        if window > 0 && position + 1i32 > window {
            start = position + 1i32 - window;
        }
        let d: f32 = convert_scalar(D);
        let scale: f32 = tile_to_scalar(
            (rsqrt(d.broadcast(shape![1]), ftz::Disabled) * k_scale.broadcast(shape![1]))
                .reshape(shape![]),
        );
        let query: Tile<f32, { [32, D] }> = q.partition(shape![32, D]).load([0i32, 0i32]);
        let query: Tile<bf16, { [32, D] }> = convert_tile(query);
        let kp = keys.partition(shape![1, 32, D]);
        let vp = values.partition(shape![1, 32, D]);
        let blocks = table.partition(shape![1]);
        let zero_scores: Tile<f32, { [32, 32] }> = constant(0.0f32, shape![32, 32]);
        let zero_rows: Tile<f32, { [32, D] }> = constant(0.0f32, shape![32, D]);
        let mut numerator = zero_rows;
        let mut denominator: Tile<f32, { [32] }> = constant(0.0f32, shape![32]);
        let mut maximum: Tile<f32, { [32] }> = constant(-1.0e30f32, shape![32]);
        let mut first = 0i32;
        let mut last = 0i32;
        if position >= 0i32 {
            let lowest = start / 32i32;
            let end = position / 32i32 + 1i32;
            let block_count = (end - lowest + PARTS - 1i32) / PARTS;
            first = lowest + split * block_count;
            last = first + block_count;
            if last > end {
                last = end;
            }
        }
        let offsets: Tile<i32, { [32] }> = iota(shape![32]);
        for block in first..last {
            let physical: i32 = tile_to_scalar(blocks.load([block]).reshape(shape![]));
            let key: Tile<bf16, { [32, D] }> =
                convert_tile(kp.load([kv, physical, 0i32]).reshape(shape![32, D]));
            let value: Tile<bf16, { [32, D] }> =
                convert_tile(vp.load([kv, physical, 0i32]).reshape(shape![32, D]));
            let raw: Tile<f32, { [32, 32] }> = mmaf(query, key.transpose(), zero_scores);
            let scores = raw * scale.broadcast(shape![32, 32]);
            let positions = offsets + (block * 32i32).broadcast(shape![32]);
            let valid = ge_tile(positions, start.broadcast(shape![32]))
                & le_tile(positions, position.broadcast(shape![32]));
            let valid = valid.reshape(shape![1, 32]).broadcast(shape![32, 32]);
            let masked: Tile<f32, { [32, 32] }> = constant(-1.0e30f32, shape![32, 32]);
            let scores = select(valid, scores, masked);
            let block_max: Tile<f32, { [32] }> = reduce_max(scores, 1i32);
            let next_max = max_tile(block_max, maximum);
            let old_scale = exp(maximum - next_max);
            let probabilities =
                exp(scores - next_max.reshape(shape![32, 1]).broadcast(shape![32, 32]));
            let block_sum: Tile<f32, { [32] }> = reduce_sum(probabilities, 1i32);
            let weights: Tile<bf16, { [32, 32] }> = convert_tile(probabilities);
            let contribution: Tile<f32, { [32, D] }> = mmaf(weights, value, zero_rows);
            numerator = numerator * old_scale.reshape(shape![32, 1]).broadcast(shape![32, D])
                + contribution;
            denominator = denominator * old_scale + block_sum;
            maximum = next_max;
        }
        out.store((numerator * v_scale.broadcast(shape![32, D])).reshape(shape![1, 32, D]));
        maxima.store(maximum.reshape(shape![1, 32]));
        sums.store(denominator.reshape(shape![1, 32]));
    }

    /// Merge [`partial_gqa`] splits: head `h` reads row `h` of its KV head's partials.
    #[cutile::entry()]
    fn merge_gqa<const D: i32, const PARTS: i32, const GROUP: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
        numerator: &Tensor<f32, { [-1, PARTS, 32, D] }>,
        maxima: &Tensor<f32, { [-1, PARTS, 32] }>,
        sums: &Tensor<f32, { [-1, PARTS, 32] }>,
    ) {
        let pid = get_tile_block_id();
        let head = pid.0;
        let kv = head / GROUP;
        let maxima = maxima
            .partition(shape![1, PARTS, 1])
            .load([kv, 0i32, head])
            .reshape(shape![PARTS]);
        let maximum: Tile<f32, { [] }> = reduce_max(maxima, 0i32);
        let weights = exp(maxima - maximum.reshape(shape![1]).broadcast(shape![PARTS]));
        let sums = sums
            .partition(shape![1, PARTS, 1])
            .load([kv, 0i32, head])
            .reshape(shape![PARTS]);
        let total: Tile<f32, { [] }> = reduce_sum(sums * weights, 0i32);
        let total = max_tile(total.reshape(shape![1]), 1.0e-20f32.broadcast(shape![1]));
        let values = numerator
            .partition(shape![1, PARTS, 1, D])
            .load([kv, 0i32, head, 0i32])
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
#[path = "../../tests/unit/resident_attention_decode.rs"]
mod tests;

mod workspace;
pub use workspace::Workspace;

#[cfg(test)]
pub use workspace::taps;
