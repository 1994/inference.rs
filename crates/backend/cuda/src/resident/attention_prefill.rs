//! Parallel causal queries over a contiguous prompt chunk; KV layout remains head-major.
#[cutile::module]
pub(crate) mod kernels {
    use cutile::core::{
        BroadcastScalar, ElementType, LoadTileLike, Reshape, Shape_0, Shape_1, Shape_2, Shape_3,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tensor_3, Tile_0, Tile_1, Tile_2, Tile_3,
        bf16, cmp_ordering, cmpf, constant, convert_scalar, convert_tile, exp, ftz, ge_tile,
        get_tile_block_id, gt_tile, iota, le_tile, lt_tile, max_tile, min_tile, mmaf, predicate,
        reduce_max, reduce_sum, rsqrt, select, tile_to_scalar,
    };
    /// Finite masked score keeps online rescaling defined before the first visible key.
    const MASKED: f32 = -1.0e30;
    /// Compensated BF16 tensor-core product with F32 inputs and accumulation; copied from
    /// `attention::kernels::sdpa`, the only other place in the tree that uses `mmaf` attention.
    fn precise_mma<const M: i32, const K: i32, const N: i32>(
        left: Tile<f32, { [M, K] }>,
        right: Tile<f32, { [K, N] }>,
    ) -> Tile<f32, { [M, N] }> {
        let lh: Tile<bf16, { [M, K] }> = convert_tile(left);
        let rh: Tile<bf16, { [K, N] }> = convert_tile(right);
        let lr: Tile<f32, { [M, K] }> = convert_tile(lh);
        let rr: Tile<f32, { [K, N] }> = convert_tile(rh);
        let ll: Tile<bf16, { [M, K] }> = convert_tile(left - lr);
        let rl: Tile<bf16, { [K, N] }> = convert_tile(right - rr);
        let zero: Tile<f32, { [M, N] }> = constant(0.0f32, shape![M, N]);
        let low = mmaf(ll, rh, zero);
        let low = mmaf(lh, rl, low);
        mmaf(lh, rh, low)
    }
    /// Append all prompt rows before the causal query kernels read the shared cache.
    #[cutile::entry()]
    fn append<E: ElementType, const D: i32, const CAP: i32, const QUANT: i32, const BT: i32>(
        keys: &mut Tensor<E, { [1, CAP, D] }>,
        values: &mut Tensor<E, { [1, CAP, D] }>,
        k: &Tensor<f32, { [-1, -1, D] }>,
        v: &Tensor<f32, { [-1, -1, D] }>,
        metadata: &Tensor<i32, { [-1] }>,
        table: &Tensor<i32, { [-1] }>,
        k_scale: f32,
        v_scale: f32,
    ) {
        let pid = get_tile_block_id();
        let meta = metadata.partition(shape![1]);
        let base: i32 = tile_to_scalar(meta.load([0i32]).reshape(shape![]));
        let count: i32 = tile_to_scalar(meta.load([1i32]).reshape(shape![]));
        let offset: i32 = tile_to_scalar(meta.load([2i32]).reshape(shape![]));
        let kp = k.partition(shape![1, 1, D]);
        let vp = v.partition(shape![1, 1, D]);
        let blocks = table.partition(shape![1]);
        for lane in 0i32..count {
            let position = base + lane + offset;
            if position >= 0 {
                // Logical block to physical block, then the row inside it. Step 1's table is the
                // identity, so this resolves to `position`; step 2's allocator is what makes the
                // physical placement differ from the logical one.
                let physical: i32 = tile_to_scalar(blocks.load([position / BT]).reshape(shape![]));
                let position = physical * BT + position % BT;
                let mut key = kp.load([lane, pid.0, 0i32]);
                let mut value = vp.load([lane, pid.0, 0i32]);
                if QUANT == 1 {
                    let low = (-448.0f32).broadcast(shape![1, 1, D]);
                    let high = 448.0f32.broadcast(shape![1, 1, D]);
                    key = min_tile(
                        max_tile(key / k_scale.broadcast(shape![1, 1, D]), low),
                        high,
                    );
                    value = min_tile(
                        max_tile(value / v_scale.broadcast(shape![1, 1, D]), low),
                        high,
                    );
                }
                let key: Tile<E, { [1, 1, D] }> = convert_tile(key);
                let value: Tile<E, { [1, 1, D] }> = convert_tile(value);
                keys.partition_mut(shape![1, 1, D])
                    .store(key, [0i32, position, 0i32]);
                values
                    .partition_mut(shape![1, 1, D])
                    .store(value, [0i32, position, 0i32]);
            }
        }
    }
    #[expect(
        clippy::useless_let_if_seq,
        clippy::range_plus_one,
        reason = "cuTile DSL uses statement control flow and exclusive ranges"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA entry takes explicit tensor bindings and independent KV scales"
    )]
    #[cutile::entry()]
    fn decode<E: ElementType, const D: i32, const GROUP: i32, const HEADS: i32, const BT: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
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
        let meta = metadata.partition(shape![1]);
        let base: i32 = tile_to_scalar(meta.load([0i32]).reshape(shape![]));
        let count: i32 = tile_to_scalar(meta.load([1i32]).reshape(shape![]));
        let offset: i32 = tile_to_scalar(meta.load([2i32]).reshape(shape![]));
        let lane = pid.0 / HEADS;
        let position = base + lane + offset;
        if lane < count && position >= 0 {
            let mut start = 0i32;
            if window > 0 && position + 1i32 > window {
                start = position + 1i32 - window;
            }
            let query = q
                .load_like(out)
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
            for block in (start / 32i32)..(position / 32i32 + 1i32) {
                // Logical block to physical block; an identity table keeps this the same row.
                let physical: i32 = tile_to_scalar(blocks.load([block]).reshape(shape![]));
                let key: Tile<f32, { [32, D] }> = convert_tile(
                    kp.load([(pid.0 % HEADS) / GROUP, physical, 0i32])
                        .reshape(shape![32, D]),
                );
                let value: Tile<f32, { [32, D] }> = convert_tile(
                    vp.load([(pid.0 % HEADS) / GROUP, physical, 0i32])
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
            out.store((numerator / denominator.broadcast(shape![D])).reshape(shape![1, D]));
        } else {
            out.store(0.0f32.broadcast(shape![1, D]));
        }
    }

    /// Tensor-core twin of [`decode`]: one tile of QT consecutive lanes for one head, with the
    /// same fp8 head-major cache read through the same runtime block bound. The host binds q and
    /// out as `[lanes, heads*D]` partitioned `[QT, D]`, so grid axis 0 is the lane tile and axis 1
    /// is the head; the KV arrives whole and is indexed device-side, exactly as [`decode`] does.
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA entry takes explicit tensor bindings and independent KV scales"
    )]
    #[cutile::entry()]
    fn decode_tiled<
        E: ElementType,
        const D: i32,
        const GROUP: i32,
        const QT: i32,
        const KB: i32,
        const BT: i32,
    >(
        out: &mut Tensor<f32, { [QT, D] }>,
        query: &Tensor<f32, { [-1, -1] }>,
        keys: &Tensor<E, { [-1, -1, D] }>,
        values: &Tensor<E, { [-1, -1, D] }>,
        metadata: &Tensor<i32, { [-1] }>,
        table: &Tensor<i32, { [-1] }>,
        window: i32,
        k_scale: f32,
        v_scale: f32,
        scale: f32,
    ) {
        let pid = get_tile_block_id();
        let meta = metadata.partition(shape![1]);
        let base: i32 = tile_to_scalar(meta.load([0i32]).reshape(shape![]));
        let count: i32 = tile_to_scalar(meta.load([1i32]).reshape(shape![]));
        let offset: i32 = tile_to_scalar(meta.load([2i32]).reshape(shape![]));
        let q: Tile<f32, { [QT, D] }> = query.partition(shape![QT, D]).load([pid.0, pid.1]);
        let kp = keys.partition(shape![1, KB, D]);
        let vp = values.partition(shape![1, KB, D]);
        let table_blocks = table.partition(shape![1]);
        let rows: Tile<i32, { [QT] }> = iota(shape![QT]) + (pid.0 * QT).broadcast(shape![QT]);
        let positions: Tile<i32, { [QT] }> = rows + (base + offset).broadcast(shape![QT]);
        // Only the lanes this chunk appended bound the loop; later rows are inactive and masked.
        let blocks = (base + offset + count - 1i32 + KB) / KB;
        let mut accumulator: Tile<f32, { [QT, D] }> = constant(0.0f32, shape![QT, D]);
        let mut row_max: Tile<f32, { [QT] }> = constant(MASKED, shape![QT]);
        let mut row_sum: Tile<f32, { [QT] }> = constant(0.0f32, shape![QT]);
        for block in 0i32..blocks {
            let physical: i32 = tile_to_scalar(table_blocks.load([block]).reshape(shape![]));
            let key: Tile<f32, { [KB, D] }> =
                convert_tile(kp.load([pid.1 / GROUP, block, 0i32]).reshape(shape![KB, D]))
                    * k_scale.broadcast(shape![KB, D]);
            let kidx: Tile<i32, { [KB] }> = iota(shape![KB]) + (block * KB).broadcast(shape![KB]);
            let kpos: Tile<i32, { [QT, KB] }> =
                kidx.reshape(shape![1, KB]).broadcast(shape![QT, KB]);
            let qpos: Tile<i32, { [QT, KB] }> =
                positions.reshape(shape![QT, 1]).broadcast(shape![QT, KB]);
            let mut valid: Tile<bool, { [QT, KB] }> = le_tile(kpos, qpos);
            if window > 0 {
                valid = valid & ge_tile(kpos, qpos - (window - 1i32).broadcast(shape![QT, KB]));
            }
            // Inactive rows hold no appended KV; leaving them unmasked would publish a finite but
            // meaningless row, so mask them here and store exactly zero at the end.
            valid = valid
                & lt_tile(
                    rows.reshape(shape![QT, 1]).broadcast(shape![QT, KB]),
                    count.broadcast(shape![QT, KB]),
                );
            let raw: Tile<f32, { [QT, KB] }> =
                precise_mma(q, key.transpose()) * scale.broadcast(shape![QT, KB]);
            let masked: Tile<f32, { [QT, KB] }> = constant(MASKED, shape![QT, KB]);
            let score: Tile<f32, { [QT, KB] }> = select(valid, raw, masked);
            let maxima: Tile<f32, { [QT] }> = reduce_max(score, 1i32);
            let next_max: Tile<f32, { [QT] }> = max_tile(row_max, maxima.reshape(shape![QT]));
            let correction: Tile<f32, { [QT] }> = exp(row_max - next_max);
            let probabilities = select(
                valid,
                exp(score - next_max.reshape(shape![QT, 1]).broadcast(shape![QT, KB])),
                0.0f32.broadcast(shape![QT, KB]),
            );
            let sums: Tile<f32, { [QT] }> = reduce_sum(probabilities, 1i32);
            row_sum = row_sum * correction + sums.reshape(shape![QT]);
            let value: Tile<f32, { [KB, D] }> = convert_tile(
                vp.load([pid.1 / GROUP, physical, 0i32])
                    .reshape(shape![KB, D]),
            ) * v_scale.broadcast(shape![KB, D]);
            let product: Tile<f32, { [QT, D] }> = precise_mma(probabilities, value);
            accumulator = accumulator * correction.reshape(shape![QT, 1]).broadcast(shape![QT, D])
                + product.reshape(shape![QT, D]);
            row_max = next_max;
        }
        let positive = gt_tile(row_sum, 0.0f32.broadcast(shape![QT]));
        let denominator = select(positive, row_sum, 1.0f32.broadcast(shape![QT]));
        let result = accumulator / denominator.reshape(shape![QT, 1]).broadcast(shape![QT, D]);
        let active: Tile<bool, { [QT] }> = lt_tile(rows, count.broadcast(shape![QT]));
        out.store(select(
            active.reshape(shape![QT, 1]).broadcast(shape![QT, D]),
            result,
            0.0f32.broadcast(shape![QT, D]),
        ));
    }
}

#[cfg(test)]
#[path = "../../tests/unit/resident_attention_prefill.rs"]
mod tests;
