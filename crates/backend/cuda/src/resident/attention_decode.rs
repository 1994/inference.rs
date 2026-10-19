//! Split long KV reductions across independent CTAs before stable softmax merging.
#[cutile::module]
pub(crate) mod kernels {
    use cutile::core::{
        BroadcastScalar, ElementType, Reshape, Shape_0, Shape_1, Shape_2, Shape_3, Shape_4,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tensor_3, Tensor_4, Tile_0, Tile_1, Tile_2,
        bf16, cmp_ordering, cmpf, constant, convert_scalar, convert_tile, exp, ftz, ge_tile,
        get_tile_block_id, iota, le_tile, max_tile, mmaf, predicate, reduce_max, reduce_sum, rsqrt,
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
    /// block once and scores the group's query heads (padded to 16 rows; `q` is viewed as
    /// `[kv_heads, group, D]`) with BF16 tensor-core products.
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA entry takes explicit tensor bindings and independent KV scales"
    )]
    #[expect(
        clippy::useless_let_if_seq,
        clippy::similar_names,
        reason = "cuTile DSL uses statement control flow; lane 0 names mirror the three-lane kernel"
    )]
    #[cutile::entry()]
    fn partial_gqa<E: ElementType, const D: i32, const PARTS: i32, const BT: i32>(
        out0: &mut Tensor<f32, { [1, 16, D] }>,
        maxima0: &mut Tensor<f32, { [1, 16] }>,
        sums0: &mut Tensor<f32, { [1, 16] }>,
        q0: &Tensor<f32, { [-1, -1, D] }>,
        metadata0: &Tensor<i32, { [-1] }>,
        keys: &Tensor<E, { [-1, -1, D] }>,
        values: &Tensor<E, { [-1, -1, D] }>,
        table: &Tensor<i32, { [-1] }>,
        window: i32,
        k_scale: f32,
        v_scale: f32,
    ) {
        let pid = get_tile_block_id();
        let kv = pid.0 / PARTS;
        let split = pid.0 % PARTS;
        let d: f32 = convert_scalar(D);
        let scale: f32 = tile_to_scalar(
            (rsqrt(d.broadcast(shape![1]), ftz::Disabled) * k_scale.broadcast(shape![1]))
                .reshape(shape![]),
        );
        let zero_scores: Tile<f32, { [16, 32] }> = constant(0.0f32, shape![16, 32]);
        let zero_rows: Tile<f32, { [16, D] }> = constant(0.0f32, shape![16, D]);
        let masked: Tile<f32, { [16, 32] }> = constant(-1.0e30f32, shape![16, 32]);
        let position0: i32 = tile_to_scalar(
            metadata0
                .partition(shape![1])
                .load([2i32])
                .reshape(shape![]),
        );
        let mut start0 = 0i32;
        if window > 0 && position0 + 1i32 > window {
            start0 = position0 + 1i32 - window;
        }
        let query0: Tile<f32, { [16, D] }> = q0
            .partition(shape![1, 16, D])
            .load([kv, 0i32, 0i32])
            .reshape(shape![16, D]);
        let query0: Tile<bf16, { [16, D] }> = convert_tile(query0);
        let mut numerator0 = zero_rows;
        let mut denominator0: Tile<f32, { [16] }> = constant(0.0f32, shape![16]);
        let mut maximum0: Tile<f32, { [16] }> = constant(-1.0e30f32, shape![16]);
        // Blocks up to the furthest lane; earlier lanes mask the rows past their own position.
        let position = position0;
        let start = start0;
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
        let kp = keys.partition(shape![1, 32, D]);
        let vp = values.partition(shape![1, 32, D]);
        let blocks = table.partition(shape![1]);
        let offsets: Tile<i32, { [32] }> = iota(shape![32]);
        for block in first..last {
            let physical: i32 = tile_to_scalar(blocks.load([block]).reshape(shape![]));
            let key: Tile<bf16, { [32, D] }> =
                convert_tile(kp.load([kv, physical, 0i32]).reshape(shape![32, D]));
            let value: Tile<bf16, { [32, D] }> =
                convert_tile(vp.load([kv, physical, 0i32]).reshape(shape![32, D]));
            let key_t: Tile<bf16, { [D, 32] }> = key.transpose();
            let positions = offsets + (block * 32i32).broadcast(shape![32]);
            let raw0: Tile<f32, { [16, 32] }> = mmaf(query0, key_t, zero_scores);
            let valid0 = ge_tile(positions, start0.broadcast(shape![32]))
                & le_tile(positions, position0.broadcast(shape![32]));
            let scores0 = select(
                valid0.reshape(shape![1, 32]).broadcast(shape![16, 32]),
                raw0 * scale.broadcast(shape![16, 32]),
                masked,
            );
            let block_max0: Tile<f32, { [16] }> = reduce_max(scores0, 1i32);
            let next0 = max_tile(block_max0, maximum0);
            let old0 = exp(maximum0 - next0);
            let p0 = exp(scores0 - next0.reshape(shape![16, 1]).broadcast(shape![16, 32]));
            let sum0: Tile<f32, { [16] }> = reduce_sum(p0, 1i32);
            let w0: Tile<bf16, { [16, 32] }> = convert_tile(p0);
            let c0: Tile<f32, { [16, D] }> = mmaf(w0, value, zero_rows);
            numerator0 = numerator0 * old0.reshape(shape![16, 1]).broadcast(shape![16, D]) + c0;
            denominator0 = denominator0 * old0 + sum0;
            maximum0 = next0;
        }
        out0.store((numerator0 * v_scale.broadcast(shape![16, D])).reshape(shape![1, 16, D]));
        maxima0.store(maximum0.reshape(shape![1, 16]));
        sums0.store(denominator0.reshape(shape![1, 16]));
    }

    fn query_position(metadata: &Tensor<i32, { [-1] }>) -> i32 {
        tile_to_scalar(metadata.partition(shape![1]).load([2i32]).reshape(shape![]))
    }

    #[expect(
        clippy::useless_let_if_seq,
        reason = "cuTile DSL uses statement control flow for scalar window bounds"
    )]
    fn window_start(position: i32, window: i32) -> i32 {
        let mut start = 0i32;
        if window > 0 && position + 1i32 > window {
            start = position + 1i32 - window;
        }
        start
    }

    fn grouped_scale<const D: i32>(k_scale: f32) -> f32 {
        let d: f32 = convert_scalar(D);
        tile_to_scalar(
            (rsqrt(d.broadcast(shape![1]), ftz::Disabled) * k_scale.broadcast(shape![1]))
                .reshape(shape![]),
        )
    }

    fn grouped_query<const D: i32>(
        q: &Tensor<f32, { [-1, -1, D] }>,
        kv: i32,
    ) -> Tile<bf16, { [16, D] }> {
        let query: Tile<f32, { [16, D] }> = q
            .partition(shape![1, 16, D])
            .load([kv, 0i32, 0i32])
            .reshape(shape![16, D]);
        convert_tile(query)
    }

    fn grouped_scores<const D: i32>(
        query: Tile<bf16, { [16, D] }>,
        key: Tile<bf16, { [D, 32] }>,
        positions: Tile<i32, { [32] }>,
        start: i32,
        position: i32,
        scale: f32,
    ) -> Tile<f32, { [16, 32] }> {
        let zero: Tile<f32, { [16, 32] }> = constant(0.0f32, shape![16, 32]);
        let raw: Tile<f32, { [16, 32] }> = mmaf(query, key, zero);
        let valid = ge_tile(positions, start.broadcast(shape![32]))
            & le_tile(positions, position.broadcast(shape![32]));
        select(
            valid.reshape(shape![1, 32]).broadcast(shape![16, 32]),
            raw * scale.broadcast(shape![16, 32]),
            constant(-1.0e30f32, shape![16, 32]),
        )
    }

    fn grouped_numerator<const D: i32>(
        numerator: Tile<f32, { [16, D] }>,
        old_scale: Tile<f32, { [16] }>,
        probabilities: Tile<f32, { [16, 32] }>,
        value: Tile<bf16, { [32, D] }>,
    ) -> Tile<f32, { [16, D] }> {
        let weights: Tile<bf16, { [16, 32] }> = convert_tile(probabilities);
        let zero: Tile<f32, { [16, D] }> = constant(0.0f32, shape![16, D]);
        let contribution: Tile<f32, { [16, D] }> = mmaf(weights, value, zero);
        numerator * old_scale.reshape(shape![16, 1]).broadcast(shape![16, D]) + contribution
    }

    /// [`partial_gqa`] for the three lanes of an MTP verification step: each KV block is read
    /// once and scored against every lane's queries, each masked to its own position.
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA entry takes explicit tensor bindings and independent KV scales"
    )]
    #[expect(
        clippy::similar_names,
        clippy::useless_let_if_seq,
        reason = "cuTile DSL uses statement control flow and matching names for the three lanes"
    )]
    #[cutile::entry()]
    fn partial_gqa3<E: ElementType, const D: i32, const PARTS: i32, const BT: i32>(
        out0: &mut Tensor<f32, { [1, 16, D] }>,
        maxima0: &mut Tensor<f32, { [1, 16] }>,
        sums0: &mut Tensor<f32, { [1, 16] }>,
        out1: &mut Tensor<f32, { [1, 16, D] }>,
        maxima1: &mut Tensor<f32, { [1, 16] }>,
        sums1: &mut Tensor<f32, { [1, 16] }>,
        out2: &mut Tensor<f32, { [1, 16, D] }>,
        maxima2: &mut Tensor<f32, { [1, 16] }>,
        sums2: &mut Tensor<f32, { [1, 16] }>,
        q0: &Tensor<f32, { [-1, -1, D] }>,
        q1: &Tensor<f32, { [-1, -1, D] }>,
        q2: &Tensor<f32, { [-1, -1, D] }>,
        metadata0: &Tensor<i32, { [-1] }>,
        metadata1: &Tensor<i32, { [-1] }>,
        metadata2: &Tensor<i32, { [-1] }>,
        keys: &Tensor<E, { [-1, -1, D] }>,
        values: &Tensor<E, { [-1, -1, D] }>,
        table: &Tensor<i32, { [-1] }>,
        window: i32,
        k_scale: f32,
        v_scale: f32,
    ) {
        let pid = get_tile_block_id();
        let kv = pid.0 / PARTS;
        let split = pid.0 % PARTS;
        let scale = grouped_scale::<D>(k_scale);
        let zero_rows: Tile<f32, { [16, D] }> = constant(0.0f32, shape![16, D]);
        let position0 = query_position(metadata0);
        let start0 = window_start(position0, window);
        let query0 = grouped_query(q0, kv);
        let mut numerator0 = zero_rows;
        let mut denominator0: Tile<f32, { [16] }> = constant(0.0f32, shape![16]);
        let mut maximum0: Tile<f32, { [16] }> = constant(-1.0e30f32, shape![16]);
        let position1 = query_position(metadata1);
        let start1 = window_start(position1, window);
        let query1 = grouped_query(q1, kv);
        let mut numerator1 = zero_rows;
        let mut denominator1: Tile<f32, { [16] }> = constant(0.0f32, shape![16]);
        let mut maximum1: Tile<f32, { [16] }> = constant(-1.0e30f32, shape![16]);
        let position2 = query_position(metadata2);
        let start2 = window_start(position2, window);
        let query2 = grouped_query(q2, kv);
        let mut numerator2 = zero_rows;
        let mut denominator2: Tile<f32, { [16] }> = constant(0.0f32, shape![16]);
        let mut maximum2: Tile<f32, { [16] }> = constant(-1.0e30f32, shape![16]);
        // Blocks up to the furthest lane; earlier lanes mask the rows past their own position.
        let mut position = position0;
        if position1 > position {
            position = position1;
        }
        if position2 > position {
            position = position2;
        }
        let mut start = start0;
        if start1 < start {
            start = start1;
        }
        if start2 < start {
            start = start2;
        }
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
        let kp = keys.partition(shape![1, 32, D]);
        let vp = values.partition(shape![1, 32, D]);
        let blocks = table.partition(shape![1]);
        let offsets: Tile<i32, { [32] }> = iota(shape![32]);
        for block in first..last {
            let physical: i32 = tile_to_scalar(blocks.load([block]).reshape(shape![]));
            let key: Tile<bf16, { [32, D] }> =
                convert_tile(kp.load([kv, physical, 0i32]).reshape(shape![32, D]));
            let value: Tile<bf16, { [32, D] }> =
                convert_tile(vp.load([kv, physical, 0i32]).reshape(shape![32, D]));
            let key_t: Tile<bf16, { [D, 32] }> = key.transpose();
            let positions = offsets + (block * 32i32).broadcast(shape![32]);
            let scores0 = grouped_scores(query0, key_t, positions, start0, position0, scale);
            let block_max0: Tile<f32, { [16] }> = reduce_max(scores0, 1i32);
            let next0 = max_tile(block_max0, maximum0);
            let old0 = exp(maximum0 - next0);
            let p0 = exp(scores0 - next0.reshape(shape![16, 1]).broadcast(shape![16, 32]));
            let sum0: Tile<f32, { [16] }> = reduce_sum(p0, 1i32);
            numerator0 = grouped_numerator(numerator0, old0, p0, value);
            denominator0 = denominator0 * old0 + sum0;
            maximum0 = next0;
            let scores1 = grouped_scores(query1, key_t, positions, start1, position1, scale);
            let block_max1: Tile<f32, { [16] }> = reduce_max(scores1, 1i32);
            let next1 = max_tile(block_max1, maximum1);
            let old1 = exp(maximum1 - next1);
            let p1 = exp(scores1 - next1.reshape(shape![16, 1]).broadcast(shape![16, 32]));
            let sum1: Tile<f32, { [16] }> = reduce_sum(p1, 1i32);
            numerator1 = grouped_numerator(numerator1, old1, p1, value);
            denominator1 = denominator1 * old1 + sum1;
            maximum1 = next1;
            let scores2 = grouped_scores(query2, key_t, positions, start2, position2, scale);
            let block_max2: Tile<f32, { [16] }> = reduce_max(scores2, 1i32);
            let next2 = max_tile(block_max2, maximum2);
            let old2 = exp(maximum2 - next2);
            let p2 = exp(scores2 - next2.reshape(shape![16, 1]).broadcast(shape![16, 32]));
            let sum2: Tile<f32, { [16] }> = reduce_sum(p2, 1i32);
            numerator2 = grouped_numerator(numerator2, old2, p2, value);
            denominator2 = denominator2 * old2 + sum2;
            maximum2 = next2;
        }
        out0.store((numerator0 * v_scale.broadcast(shape![16, D])).reshape(shape![1, 16, D]));
        maxima0.store(maximum0.reshape(shape![1, 16]));
        sums0.store(denominator0.reshape(shape![1, 16]));
        out1.store((numerator1 * v_scale.broadcast(shape![16, D])).reshape(shape![1, 16, D]));
        maxima1.store(maximum1.reshape(shape![1, 16]));
        sums1.store(denominator1.reshape(shape![1, 16]));
        out2.store((numerator2 * v_scale.broadcast(shape![16, D])).reshape(shape![1, 16, D]));
        maxima2.store(maximum2.reshape(shape![1, 16]));
        sums2.store(denominator2.reshape(shape![1, 16]));
    }

    /// Merge grouped splits: head `h` reads row `h % GROUP` of its KV head's partials.
    #[cutile::entry()]
    fn merge_gqa<const D: i32, const PARTS: i32, const GROUP: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
        numerator: &Tensor<f32, { [-1, PARTS, 16, D] }>,
        maxima: &Tensor<f32, { [-1, PARTS, 16] }>,
        sums: &Tensor<f32, { [-1, PARTS, 16] }>,
    ) {
        let pid = get_tile_block_id();
        let kv = pid.0 / GROUP;
        let row = pid.0 % GROUP;
        let maxima = maxima
            .partition(shape![1, PARTS, 1])
            .load([kv, 0i32, row])
            .reshape(shape![PARTS]);
        let maximum: Tile<f32, { [] }> = reduce_max(maxima, 0i32);
        let weights = exp(maxima - maximum.reshape(shape![1]).broadcast(shape![PARTS]));
        let sums = sums
            .partition(shape![1, PARTS, 1])
            .load([kv, 0i32, row])
            .reshape(shape![PARTS]);
        let total: Tile<f32, { [] }> = reduce_sum(sums * weights, 0i32);
        let total = max_tile(total.reshape(shape![1]), 1.0e-20f32.broadcast(shape![1]));
        let values = numerator
            .partition(shape![1, PARTS, 1, D])
            .load([kv, 0i32, row, 0i32])
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
pub use workspace::VERIFY_LANES;
pub use workspace::Workspace;

#[cfg(test)]
pub use workspace::taps;
