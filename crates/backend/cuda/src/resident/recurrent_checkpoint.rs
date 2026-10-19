//! Pooled verification twins of the established per-lane kernels.
//! Keep their arithmetic separate and unchanged: extra stores can change compiler lowering,
//! so exact output, state and rollback tests compare against the original kernels.
#[cutile::module]
pub(crate) mod checkpoint {
    use cutile::core::{
        BroadcastScalar, LoadTileLike, Reshape, Shape_0, Shape_1, Shape_2, Shape_3,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tensor_3, Tile_0, Tile_1, Tile_2, Tile_3,
        cmp_ordering, cmpf, convert_scalar, exp, ftz, ge_tile, get_tile_block_id, log, predicate,
        reduce_sum, rsqrt, select, tile_to_scalar,
    };

    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA checkpoint kernel binds state and snapshot tensors explicitly"
    )]
    #[cutile::entry()]
    fn conv4_checkpoint(
        out: &mut Tensor<f32, { [128] }>,
        h0: &mut Tensor<f32, { [128] }>,
        h1: &mut Tensor<f32, { [128] }>,
        h2: &mut Tensor<f32, { [128] }>,
        c0: &mut Tensor<f32, { [128] }>,
        c1: &mut Tensor<f32, { [128] }>,
        c2: &mut Tensor<f32, { [128] }>,
        x: &Tensor<f32, { [-1] }>,
        w: &Tensor<f32, { [-1, 4] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let position: i32 =
            tile_to_scalar(metadata.partition(shape![1]).load([2i32]).reshape(shape![]));
        if position >= 0 {
            let pid = get_tile_block_id();
            let x = x.load_like(out);
            let oldest = h0.load_like(out);
            let middle = h1.load_like(out);
            let newest = h2.load_like(out);
            let wp = w.partition(shape![128, 1]);
            let wa = wp.load([pid.0, 0i32]).reshape(shape![128]);
            let wb = wp.load([pid.0, 1i32]).reshape(shape![128]);
            let wc = wp.load([pid.0, 2i32]).reshape(shape![128]);
            let wd = wp.load([pid.0, 3i32]).reshape(shape![128]);
            let value = oldest * wa + middle * wb + newest * wc + x * wd;
            let negative: Tile<f32, { [128] }> = 0.0f32.broadcast(shape![128]) - value;
            out.store(value / (1.0f32.broadcast(shape![128]) + exp(negative)));
            h0.store(middle);
            h1.store(newest);
            h2.store(x);
            c0.store(middle);
            c1.store(newest);
            c2.store(x);
        } else {
            out.store(0.0f32.broadcast(shape![128]));
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA state kernel binds explicit tensors and the padding metadata"
    )]
    #[cutile::entry()]
    fn delta_checkpoint<const KH: i32, const VH: i32, const D: i32, const SW: i32>(
        out: &mut Tensor<f32, { [1, 1, SW] }>,
        state: &mut Tensor<f32, { [1, D, SW] }>,
        checkpoint: &mut Tensor<f32, { [1, D, SW] }>,
        qkv: &Tensor<f32, { [-1] }>,
        beta: &Tensor<f32, { [-1] }>,
        alpha: &Tensor<f32, { [-1] }>,
        a_log: &Tensor<f32, { [-1] }>,
        bias: &Tensor<f32, { [-1] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let position: i32 =
            tile_to_scalar(metadata.partition(shape![1]).load([2i32]).reshape(shape![]));
        if position >= 0 {
            let pid = get_tile_block_id();
            let key = pid.0 / (VH / KH);
            let qp = qkv.partition(shape![D]);
            let mut q = qp.load([key]);
            let mut k = qp.load([KH + key]);
            // One block of the value dimension. The state's columns are mutually independent -
            // every reduction below is over the key dimension - so splitting them changes no
            // summation order while giving the grid SW times more blocks to fill the device.
            let vp = qkv.partition(shape![SW]);
            let v = vp.load([(2i32 * KH + pid.0) * (D / SW) + pid.2]);
            let qs: Tile<f32, { [] }> = reduce_sum(q * q, 0i32);
            let ks: Tile<f32, { [] }> = reduce_sum(k * k, 0i32);
            let qs = qs.reshape(shape![1]);
            let ks = ks.reshape(shape![1]);
            let qinv = rsqrt(qs + 1e-6f32.broadcast(shape![1]), ftz::Disabled);
            let kinv = rsqrt(ks + 1e-6f32.broadcast(shape![1]), ftz::Disabled);
            let d: f32 = convert_scalar(D);
            let dinv = rsqrt(d.broadcast(shape![1]), ftz::Disabled);
            q = q * (qinv * dinv).broadcast(shape![D]);
            k = k * kinv.broadcast(shape![D]);
            let one = 1.0f32.broadcast(shape![1]);
            let zero = 0.0f32.broadcast(shape![1]);
            let bt = beta.partition(shape![1]).load([pid.0]);
            let bt = one / (one + exp(zero - bt));
            let at =
                alpha.partition(shape![1]).load([pid.0]) + bias.partition(shape![1]).load([pid.0]);
            let soft = log(one + exp(at));
            let large = cmpf(
                at,
                20.0f32.broadcast(shape![1]),
                predicate::GreaterThan,
                cmp_ordering::Ordered,
            );
            let soft = select(large, at, soft);
            let decay = exp(zero - exp(a_log.partition(shape![1]).load([pid.0])) * soft);
            let old = state.load_like(state).reshape(shape![D, SW]);
            let decayed = old * decay.reshape(shape![1, 1]).broadcast(shape![D, SW]);
            let kb = k.reshape(shape![D, 1]).broadcast(shape![D, SW]);
            let predicted: Tile<f32, { [SW] }> = reduce_sum(decayed * kb, 0i32);
            let predicted = predicted.reshape(shape![SW]);
            let diff = (v - predicted) * bt.broadcast(shape![SW]);
            let updated: Tile<f32, { [D, SW] }> =
                decayed + kb * diff.reshape(shape![1, SW]).broadcast(shape![D, SW]);
            let qb = q.reshape(shape![D, 1]).broadcast(shape![D, SW]);
            let result: Tile<f32, { [SW] }> = reduce_sum(updated * qb, 0i32);
            state.store(updated.reshape(shape![1, D, SW]));
            checkpoint.store(updated.reshape(shape![1, D, SW]));
            out.store(result.reshape(shape![1, 1, SW]));
        } else {
            out.store(0.0f32.broadcast(shape![1, 1, SW]));
        }
    }

    /// Whether a lane's metadata marks a real token rather than padding.
    fn lane_active(metadata: &Tensor<i32, { [-1] }>) -> Tile<bool, { [1] }> {
        let position = metadata.partition(shape![1]).load([2i32]);
        ge_tile(position, 0i32.broadcast(shape![1]))
    }

    /// [`delta_checkpoint`] for the three lanes of an MTP verification step: the state block is
    /// read once, the lanes' updates are applied in order in registers, and each lane writes its
    /// own output and rollback snapshot before the final state is stored. A padding lane leaves
    /// the state unchanged and outputs zero.
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA state kernel binds three lanes' outputs, snapshots and inputs explicitly"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "The DSL cannot pass generic tile shapes through helpers, so the lanes are inline"
    )]
    #[cutile::entry()]
    fn delta_checkpoint3<const KH: i32, const VH: i32, const D: i32, const SW: i32>(
        out0: &mut Tensor<f32, { [1, 1, SW] }>,
        out1: &mut Tensor<f32, { [1, 1, SW] }>,
        out2: &mut Tensor<f32, { [1, 1, SW] }>,
        state: &mut Tensor<f32, { [1, D, SW] }>,
        checkpoint0: &mut Tensor<f32, { [1, D, SW] }>,
        checkpoint1: &mut Tensor<f32, { [1, D, SW] }>,
        checkpoint2: &mut Tensor<f32, { [1, D, SW] }>,
        qkv0: &Tensor<f32, { [-1] }>,
        qkv1: &Tensor<f32, { [-1] }>,
        qkv2: &Tensor<f32, { [-1] }>,
        beta0: &Tensor<f32, { [-1] }>,
        beta1: &Tensor<f32, { [-1] }>,
        beta2: &Tensor<f32, { [-1] }>,
        alpha0: &Tensor<f32, { [-1] }>,
        alpha1: &Tensor<f32, { [-1] }>,
        alpha2: &Tensor<f32, { [-1] }>,
        a_log: &Tensor<f32, { [-1] }>,
        bias: &Tensor<f32, { [-1] }>,
        metadata0: &Tensor<i32, { [-1] }>,
        metadata1: &Tensor<i32, { [-1] }>,
        metadata2: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let key = pid.0 / (VH / KH);
        let zero_out: Tile<f32, { [1, 1, SW] }> = 0.0f32.broadcast(shape![1, 1, SW]);
        let one = 1.0f32.broadcast(shape![1]);
        let zero = 0.0f32.broadcast(shape![1]);
        let epsilon = 1e-6f32.broadcast(shape![1]);
        let threshold = 20.0f32.broadcast(shape![1]);
        let d: f32 = convert_scalar(D);
        let dinv = rsqrt(d.broadcast(shape![1]), ftz::Disabled);
        let head_bias = bias.partition(shape![1]).load([pid.0]);
        let rate = exp(a_log.partition(shape![1]).load([pid.0]));
        let initial: Tile<f32, { [D, SW] }> = state.load_like(state).reshape(shape![D, SW]);

        let active0: Tile<bool, { [1] }> = lane_active(metadata0);
        let qp0 = qkv0.partition(shape![D]);
        let q0: Tile<f32, { [D] }> = qp0.load([key]);
        let k0: Tile<f32, { [D] }> = qp0.load([KH + key]);
        let v0: Tile<f32, { [SW] }> = qkv0
            .partition(shape![SW])
            .load([(2i32 * KH + pid.0) * (D / SW) + pid.2]);
        let qs0: Tile<f32, { [] }> = reduce_sum(q0 * q0, 0i32);
        let ks0: Tile<f32, { [] }> = reduce_sum(k0 * k0, 0i32);
        let qinv0 = rsqrt(qs0.reshape(shape![1]) + epsilon, ftz::Disabled);
        let kinv0 = rsqrt(ks0.reshape(shape![1]) + epsilon, ftz::Disabled);
        let q0 = q0 * (qinv0 * dinv).broadcast(shape![D]);
        let k0 = k0 * kinv0.broadcast(shape![D]);
        let bt0 = beta0.partition(shape![1]).load([pid.0]);
        let bt0 = one / (one + exp(zero - bt0));
        let at0 = alpha0.partition(shape![1]).load([pid.0]) + head_bias;
        let soft0 = log(one + exp(at0));
        let large0 = cmpf(
            at0,
            threshold,
            predicate::GreaterThan,
            cmp_ordering::Ordered,
        );
        let soft0 = select(large0, at0, soft0);
        let decay0 = exp(zero - rate * soft0);
        let decayed0 = initial * decay0.reshape(shape![1, 1]).broadcast(shape![D, SW]);
        let kb0 = k0.reshape(shape![D, 1]).broadcast(shape![D, SW]);
        let predicted0: Tile<f32, { [SW] }> = reduce_sum(decayed0 * kb0, 0i32);
        let diff0 = (v0 - predicted0.reshape(shape![SW])) * bt0.broadcast(shape![SW]);
        let next0: Tile<f32, { [D, SW] }> =
            decayed0 + kb0 * diff0.reshape(shape![1, SW]).broadcast(shape![D, SW]);
        let after0 = select(
            active0.reshape(shape![1, 1]).broadcast(shape![D, SW]),
            next0,
            initial,
        );
        let qb0 = q0.reshape(shape![D, 1]).broadcast(shape![D, SW]);
        let result0: Tile<f32, { [SW] }> = reduce_sum(after0 * qb0, 0i32);
        checkpoint0.store(after0.reshape(shape![1, D, SW]));
        out0.store(select(
            active0.reshape(shape![1, 1, 1]).broadcast(shape![1, 1, SW]),
            result0.reshape(shape![1, 1, SW]),
            zero_out,
        ));

        let active1: Tile<bool, { [1] }> = lane_active(metadata1);
        let qp1 = qkv1.partition(shape![D]);
        let q1: Tile<f32, { [D] }> = qp1.load([key]);
        let k1: Tile<f32, { [D] }> = qp1.load([KH + key]);
        let v1: Tile<f32, { [SW] }> = qkv1
            .partition(shape![SW])
            .load([(2i32 * KH + pid.0) * (D / SW) + pid.2]);
        let qs1: Tile<f32, { [] }> = reduce_sum(q1 * q1, 0i32);
        let ks1: Tile<f32, { [] }> = reduce_sum(k1 * k1, 0i32);
        let qinv1 = rsqrt(qs1.reshape(shape![1]) + epsilon, ftz::Disabled);
        let kinv1 = rsqrt(ks1.reshape(shape![1]) + epsilon, ftz::Disabled);
        let q1 = q1 * (qinv1 * dinv).broadcast(shape![D]);
        let k1 = k1 * kinv1.broadcast(shape![D]);
        let bt1 = beta1.partition(shape![1]).load([pid.0]);
        let bt1 = one / (one + exp(zero - bt1));
        let at1 = alpha1.partition(shape![1]).load([pid.0]) + head_bias;
        let soft1 = log(one + exp(at1));
        let large1 = cmpf(
            at1,
            threshold,
            predicate::GreaterThan,
            cmp_ordering::Ordered,
        );
        let soft1 = select(large1, at1, soft1);
        let decay1 = exp(zero - rate * soft1);
        let decayed1 = after0 * decay1.reshape(shape![1, 1]).broadcast(shape![D, SW]);
        let kb1 = k1.reshape(shape![D, 1]).broadcast(shape![D, SW]);
        let predicted1: Tile<f32, { [SW] }> = reduce_sum(decayed1 * kb1, 0i32);
        let diff1 = (v1 - predicted1.reshape(shape![SW])) * bt1.broadcast(shape![SW]);
        let next1: Tile<f32, { [D, SW] }> =
            decayed1 + kb1 * diff1.reshape(shape![1, SW]).broadcast(shape![D, SW]);
        let after1 = select(
            active1.reshape(shape![1, 1]).broadcast(shape![D, SW]),
            next1,
            after0,
        );
        let qb1 = q1.reshape(shape![D, 1]).broadcast(shape![D, SW]);
        let result1: Tile<f32, { [SW] }> = reduce_sum(after1 * qb1, 0i32);
        checkpoint1.store(after1.reshape(shape![1, D, SW]));
        out1.store(select(
            active1.reshape(shape![1, 1, 1]).broadcast(shape![1, 1, SW]),
            result1.reshape(shape![1, 1, SW]),
            zero_out,
        ));

        let active2: Tile<bool, { [1] }> = lane_active(metadata2);
        let qp2 = qkv2.partition(shape![D]);
        let q2: Tile<f32, { [D] }> = qp2.load([key]);
        let k2: Tile<f32, { [D] }> = qp2.load([KH + key]);
        let v2: Tile<f32, { [SW] }> = qkv2
            .partition(shape![SW])
            .load([(2i32 * KH + pid.0) * (D / SW) + pid.2]);
        let qs2: Tile<f32, { [] }> = reduce_sum(q2 * q2, 0i32);
        let ks2: Tile<f32, { [] }> = reduce_sum(k2 * k2, 0i32);
        let qinv2 = rsqrt(qs2.reshape(shape![1]) + epsilon, ftz::Disabled);
        let kinv2 = rsqrt(ks2.reshape(shape![1]) + epsilon, ftz::Disabled);
        let q2 = q2 * (qinv2 * dinv).broadcast(shape![D]);
        let k2 = k2 * kinv2.broadcast(shape![D]);
        let bt2 = beta2.partition(shape![1]).load([pid.0]);
        let bt2 = one / (one + exp(zero - bt2));
        let at2 = alpha2.partition(shape![1]).load([pid.0]) + head_bias;
        let soft2 = log(one + exp(at2));
        let large2 = cmpf(
            at2,
            threshold,
            predicate::GreaterThan,
            cmp_ordering::Ordered,
        );
        let soft2 = select(large2, at2, soft2);
        let decay2 = exp(zero - rate * soft2);
        let decayed2 = after1 * decay2.reshape(shape![1, 1]).broadcast(shape![D, SW]);
        let kb2 = k2.reshape(shape![D, 1]).broadcast(shape![D, SW]);
        let predicted2: Tile<f32, { [SW] }> = reduce_sum(decayed2 * kb2, 0i32);
        let diff2 = (v2 - predicted2.reshape(shape![SW])) * bt2.broadcast(shape![SW]);
        let next2: Tile<f32, { [D, SW] }> =
            decayed2 + kb2 * diff2.reshape(shape![1, SW]).broadcast(shape![D, SW]);
        let after2 = select(
            active2.reshape(shape![1, 1]).broadcast(shape![D, SW]),
            next2,
            after1,
        );
        let qb2 = q2.reshape(shape![D, 1]).broadcast(shape![D, SW]);
        let result2: Tile<f32, { [SW] }> = reduce_sum(after2 * qb2, 0i32);
        checkpoint2.store(after2.reshape(shape![1, D, SW]));
        out2.store(select(
            active2.reshape(shape![1, 1, 1]).broadcast(shape![1, 1, SW]),
            result2.reshape(shape![1, 1, SW]),
            zero_out,
        ));
        state.store(after2.reshape(shape![1, D, SW]));
    }
}

#[cfg(test)]
#[path = "../../tests/unit/resident_recurrent_checkpoint.rs"]
mod tests;
