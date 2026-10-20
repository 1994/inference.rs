//! Snapshot-free recurrent verification for the pooled speculation graph.
//!
//! The verification lanes of one slot are scored in one launch per recurrent node that reads the
//! slot's state and leaves it untouched; once acceptance is known, one commit launch applies the
//! accepted prefix. Rolling back therefore needs no per-lane state snapshot, which was two full
//! recurrent state copies (about 300 MB on the 27B) per slot.
#[cutile::module]
pub(crate) mod fused {
    use cutile::core::{
        BroadcastScalar, LoadTileLike, Reshape, Shape_1, Shape_2, Shape_3, StoreTileAtCurrentBlock,
        Tensor_1, Tensor_2, Tensor_3, Tile_0, Tile_1, Tile_2, Tile_3, cmp_ordering, cmpf,
        convert_scalar, exp, ftz, ge_tile, get_tile_block_id, log, predicate, reduce_sum, rsqrt,
        select,
    };

    /// Whether a lane's metadata marks a real token rather than padding.
    fn lane_active(metadata: &Tensor<i32, { [-1] }>) -> Tile<bool, { [1] }> {
        let position = metadata.partition(shape![1]).load([2i32]);
        ge_tile(position, 0i32.broadcast(shape![1]))
    }

    /// Short-convolution outputs of one slot's three verification lanes, each over the window
    /// its predecessors would leave, without shifting the state: [`conv_commit3`] shifts it by
    /// the accepted prefix. Matches `conv4`'s arithmetic; a padding lane outputs zero.
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA state kernel binds three verification lanes' rows explicitly"
    )]
    #[cutile::entry()]
    fn conv_verify3(
        out0: &mut Tensor<f32, { [128] }>,
        out1: &mut Tensor<f32, { [128] }>,
        out2: &mut Tensor<f32, { [128] }>,
        h0: &Tensor<f32, { [-1] }>,
        h1: &Tensor<f32, { [-1] }>,
        h2: &Tensor<f32, { [-1] }>,
        x0: &Tensor<f32, { [-1] }>,
        x1: &Tensor<f32, { [-1] }>,
        x2: &Tensor<f32, { [-1] }>,
        w: &Tensor<f32, { [-1, 4] }>,
        metadata0: &Tensor<i32, { [-1] }>,
        metadata1: &Tensor<i32, { [-1] }>,
        metadata2: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let oldest: Tile<f32, { [128] }> = h0.partition(shape![128]).load([pid.0]);
        let middle: Tile<f32, { [128] }> = h1.partition(shape![128]).load([pid.0]);
        let newest: Tile<f32, { [128] }> = h2.partition(shape![128]).load([pid.0]);
        let first: Tile<f32, { [128] }> = x0.partition(shape![128]).load([pid.0]);
        let second: Tile<f32, { [128] }> = x1.partition(shape![128]).load([pid.0]);
        let third: Tile<f32, { [128] }> = x2.partition(shape![128]).load([pid.0]);
        let wp = w.partition(shape![128, 1]);
        let wa = wp.load([pid.0, 0i32]).reshape(shape![128]);
        let wb = wp.load([pid.0, 1i32]).reshape(shape![128]);
        let wc = wp.load([pid.0, 2i32]).reshape(shape![128]);
        let wd = wp.load([pid.0, 3i32]).reshape(shape![128]);
        let one: Tile<f32, { [128] }> = 1.0f32.broadcast(shape![128]);
        let zero: Tile<f32, { [128] }> = 0.0f32.broadcast(shape![128]);
        let value0 = oldest * wa + middle * wb + newest * wc + first * wd;
        let value1 = middle * wa + newest * wb + first * wc + second * wd;
        let value2 = newest * wa + first * wb + second * wc + third * wd;
        let silu0 = value0 / (one + exp(zero - value0));
        let silu1 = value1 / (one + exp(zero - value1));
        let silu2 = value2 / (one + exp(zero - value2));
        out0.store(select(
            lane_active(metadata0).broadcast(shape![128]),
            silu0,
            zero,
        ));
        out1.store(select(
            lane_active(metadata1).broadcast(shape![128]),
            silu1,
            zero,
        ));
        out2.store(select(
            lane_active(metadata2).broadcast(shape![128]),
            silu2,
            zero,
        ));
    }

    /// Shift one slot's convolution window by its first `count` verification lanes' inputs; a
    /// padding lane is never applied.
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA state kernel binds three verification lanes' rows explicitly"
    )]
    #[cutile::entry()]
    fn conv_commit3(
        h0: &mut Tensor<f32, { [128] }>,
        h1: &mut Tensor<f32, { [128] }>,
        h2: &mut Tensor<f32, { [128] }>,
        x0: &Tensor<f32, { [-1] }>,
        x1: &Tensor<f32, { [-1] }>,
        x2: &Tensor<f32, { [-1] }>,
        metadata0: &Tensor<i32, { [-1] }>,
        metadata1: &Tensor<i32, { [-1] }>,
        metadata2: &Tensor<i32, { [-1] }>,
        count: i32,
    ) {
        let pid = get_tile_block_id();
        let oldest: Tile<f32, { [128] }> = h0.load_like(h0);
        let middle: Tile<f32, { [128] }> = h1.load_like(h1);
        let newest: Tile<f32, { [128] }> = h2.load_like(h2);
        let first: Tile<f32, { [128] }> = x0.partition(shape![128]).load([pid.0]);
        let second: Tile<f32, { [128] }> = x1.partition(shape![128]).load([pid.0]);
        let third: Tile<f32, { [128] }> = x2.partition(shape![128]).load([pid.0]);
        let accepted = count.broadcast(shape![1]);
        let one: Tile<bool, { [1] }> =
            lane_active(metadata0) & ge_tile(accepted, 1i32.broadcast(shape![1]));
        let two: Tile<bool, { [1] }> =
            one & lane_active(metadata1) & ge_tile(accepted, 2i32.broadcast(shape![1]));
        let three: Tile<bool, { [1] }> =
            two & lane_active(metadata2) & ge_tile(accepted, 3i32.broadcast(shape![1]));
        let one = one.broadcast(shape![128]);
        let two = two.broadcast(shape![128]);
        let three = three.broadcast(shape![128]);
        // The window after n applied lanes is [h0, h1, h2, x0, x1, x2][n..n + 3].
        h0.store(select(
            three,
            first,
            select(two, newest, select(one, middle, oldest)),
        ));
        h1.store(select(
            three,
            second,
            select(two, first, select(one, newest, middle)),
        ));
        h2.store(select(
            three,
            third,
            select(two, second, select(one, first, newest)),
        ));
    }

    /// Delta-rule outputs of the three verification lanes of one slot, each read against the
    /// state its predecessors would leave, without writing the state: [`delta_commit3`] applies
    /// the accepted prefix once acceptance is known, so no rollback snapshot is needed. A padding
    /// lane leaves the running state unchanged and outputs zero.
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA state kernel binds three verification lanes' rows explicitly"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "The DSL cannot pass generic tile shapes through helpers, so the lanes are inline"
    )]
    #[cutile::entry()]
    fn delta_verify3<const KH: i32, const VH: i32, const D: i32, const SW: i32>(
        out0: &mut Tensor<f32, { [1, 1, SW] }>,
        out1: &mut Tensor<f32, { [1, 1, SW] }>,
        out2: &mut Tensor<f32, { [1, 1, SW] }>,
        state: &Tensor<f32, { [-1, D, -1] }>,
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
        let one = 1.0f32.broadcast(shape![1]);
        let zero = 0.0f32.broadcast(shape![1]);
        let epsilon = 1e-6f32.broadcast(shape![1]);
        let threshold = 20.0f32.broadcast(shape![1]);
        let head_bias = bias.partition(shape![1]).load([pid.0]);
        let rate = exp(a_log.partition(shape![1]).load([pid.0]));
        let zero_out: Tile<f32, { [1, 1, SW] }> = 0.0f32.broadcast(shape![1, 1, SW]);
        let d: f32 = convert_scalar(D);
        let dinv = rsqrt(d.broadcast(shape![1]), ftz::Disabled);
        let initial: Tile<f32, { [D, SW] }> = state
            .partition(shape![1, D, SW])
            .load([pid.0, 0i32, pid.2])
            .reshape(shape![D, SW]);
        let active0: Tile<bool, { [1] }> = lane_active(metadata0);
        let qp0 = qkv0.partition(shape![D]);
        let q0: Tile<f32, { [D] }> = qp0.load([key]);
        let qs0: Tile<f32, { [] }> = reduce_sum(q0 * q0, 0i32);
        let qinv0 = rsqrt(qs0.reshape(shape![1]) + epsilon, ftz::Disabled);
        let q0 = q0 * (qinv0 * dinv).broadcast(shape![D]);
        let k0: Tile<f32, { [D] }> = qp0.load([KH + key]);
        let v0: Tile<f32, { [SW] }> = qkv0
            .partition(shape![SW])
            .load([(2i32 * KH + pid.0) * (D / SW) + pid.2]);
        let ks0: Tile<f32, { [] }> = reduce_sum(k0 * k0, 0i32);
        let kinv0 = rsqrt(ks0.reshape(shape![1]) + epsilon, ftz::Disabled);
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
        out0.store(select(
            active0.reshape(shape![1, 1, 1]).broadcast(shape![1, 1, SW]),
            result0.reshape(shape![1, 1, SW]),
            zero_out,
        ));
        let active1: Tile<bool, { [1] }> = lane_active(metadata1);
        let qp1 = qkv1.partition(shape![D]);
        let q1: Tile<f32, { [D] }> = qp1.load([key]);
        let qs1: Tile<f32, { [] }> = reduce_sum(q1 * q1, 0i32);
        let qinv1 = rsqrt(qs1.reshape(shape![1]) + epsilon, ftz::Disabled);
        let q1 = q1 * (qinv1 * dinv).broadcast(shape![D]);
        let k1: Tile<f32, { [D] }> = qp1.load([KH + key]);
        let v1: Tile<f32, { [SW] }> = qkv1
            .partition(shape![SW])
            .load([(2i32 * KH + pid.0) * (D / SW) + pid.2]);
        let ks1: Tile<f32, { [] }> = reduce_sum(k1 * k1, 0i32);
        let kinv1 = rsqrt(ks1.reshape(shape![1]) + epsilon, ftz::Disabled);
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
        out1.store(select(
            active1.reshape(shape![1, 1, 1]).broadcast(shape![1, 1, SW]),
            result1.reshape(shape![1, 1, SW]),
            zero_out,
        ));
        let active2: Tile<bool, { [1] }> = lane_active(metadata2);
        let qp2 = qkv2.partition(shape![D]);
        let q2: Tile<f32, { [D] }> = qp2.load([key]);
        let qs2: Tile<f32, { [] }> = reduce_sum(q2 * q2, 0i32);
        let qinv2 = rsqrt(qs2.reshape(shape![1]) + epsilon, ftz::Disabled);
        let q2 = q2 * (qinv2 * dinv).broadcast(shape![D]);
        let k2: Tile<f32, { [D] }> = qp2.load([KH + key]);
        let v2: Tile<f32, { [SW] }> = qkv2
            .partition(shape![SW])
            .load([(2i32 * KH + pid.0) * (D / SW) + pid.2]);
        let ks2: Tile<f32, { [] }> = reduce_sum(k2 * k2, 0i32);
        let kinv2 = rsqrt(ks2.reshape(shape![1]) + epsilon, ftz::Disabled);
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
        out2.store(select(
            active2.reshape(shape![1, 1, 1]).broadcast(shape![1, 1, SW]),
            result2.reshape(shape![1, 1, SW]),
            zero_out,
        ));
    }

    /// Apply the first `count` verification lanes' delta-rule updates to one slot's state, with
    /// the arithmetic of [`delta_verify3`]; a padding lane is never applied.
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA state kernel binds three verification lanes' rows explicitly"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "The DSL cannot pass generic tile shapes through helpers, so the lanes are inline"
    )]
    #[cutile::entry()]
    fn delta_commit3<const KH: i32, const VH: i32, const D: i32, const SW: i32>(
        state: &mut Tensor<f32, { [1, D, SW] }>,
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
        count: i32,
    ) {
        let pid = get_tile_block_id();
        let key = pid.0 / (VH / KH);
        let one = 1.0f32.broadcast(shape![1]);
        let zero = 0.0f32.broadcast(shape![1]);
        let epsilon = 1e-6f32.broadcast(shape![1]);
        let threshold = 20.0f32.broadcast(shape![1]);
        let head_bias = bias.partition(shape![1]).load([pid.0]);
        let rate = exp(a_log.partition(shape![1]).load([pid.0]));
        let initial: Tile<f32, { [D, SW] }> = state.load_like(state).reshape(shape![D, SW]);
        let applied0: Tile<bool, { [1] }> =
            lane_active(metadata0) & ge_tile(count.broadcast(shape![1]), 1i32.broadcast(shape![1]));
        let qp0 = qkv0.partition(shape![D]);
        let k0: Tile<f32, { [D] }> = qp0.load([KH + key]);
        let v0: Tile<f32, { [SW] }> = qkv0
            .partition(shape![SW])
            .load([(2i32 * KH + pid.0) * (D / SW) + pid.2]);
        let ks0: Tile<f32, { [] }> = reduce_sum(k0 * k0, 0i32);
        let kinv0 = rsqrt(ks0.reshape(shape![1]) + epsilon, ftz::Disabled);
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
            applied0.reshape(shape![1, 1]).broadcast(shape![D, SW]),
            next0,
            initial,
        );
        let applied1: Tile<bool, { [1] }> = applied0
            & lane_active(metadata1)
            & ge_tile(count.broadcast(shape![1]), 2i32.broadcast(shape![1]));
        let qp1 = qkv1.partition(shape![D]);
        let k1: Tile<f32, { [D] }> = qp1.load([KH + key]);
        let v1: Tile<f32, { [SW] }> = qkv1
            .partition(shape![SW])
            .load([(2i32 * KH + pid.0) * (D / SW) + pid.2]);
        let ks1: Tile<f32, { [] }> = reduce_sum(k1 * k1, 0i32);
        let kinv1 = rsqrt(ks1.reshape(shape![1]) + epsilon, ftz::Disabled);
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
            applied1.reshape(shape![1, 1]).broadcast(shape![D, SW]),
            next1,
            after0,
        );
        let applied2: Tile<bool, { [1] }> = applied1
            & lane_active(metadata2)
            & ge_tile(count.broadcast(shape![1]), 3i32.broadcast(shape![1]));
        let qp2 = qkv2.partition(shape![D]);
        let k2: Tile<f32, { [D] }> = qp2.load([KH + key]);
        let v2: Tile<f32, { [SW] }> = qkv2
            .partition(shape![SW])
            .load([(2i32 * KH + pid.0) * (D / SW) + pid.2]);
        let ks2: Tile<f32, { [] }> = reduce_sum(k2 * k2, 0i32);
        let kinv2 = rsqrt(ks2.reshape(shape![1]) + epsilon, ftz::Disabled);
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
            applied2.reshape(shape![1, 1]).broadcast(shape![D, SW]),
            next2,
            after1,
        );
        state.store(after2.reshape(shape![1, D, SW]));
    }
}

use super::{ActivationArena, ProgramWeights, batch::States, capture::error};
use cutile::prelude::*;
use infer_core::TensorId;
use infer_ir::{TensorNode, TensorOp};

/// Candidate lanes of one slot that the fused kernels cover.
pub(super) const LANES: usize = 3;

/// One slot's verification lanes in the batched arena and their per-lane metadata.
pub(super) struct SlotLanes<'a> {
    pub lanes: [usize; LANES],
    pub metadata: [&'a Tensor<i32>; LANES],
}

/// Per (slot, node index), each commit input's three lane rows, copied out of the arena by the
/// verify graph: the arena reuses those slots for later nodes before acceptance is known.
pub(super) type Staging = std::collections::BTreeMap<(usize, usize), Vec<[Tensor<f32>; LANES]>>;

/// Activations a covered node's commit re-reads; weights are read in place.
fn commit_inputs(node: &TensorNode) -> &[TensorId] {
    match node.op {
        TensorOp::Conv { .. } => &node.inputs[..1],
        _ => &node.inputs[..3],
    }
}

/// Allocate the staging rows of every covered node for `slots` slots.
/// # Errors
/// Rejects unbatched activations and failed device allocation.
pub(super) fn allocate_staging(
    device: &crate::device::CudaDevice,
    nodes: &[TensorNode],
    arena: &ActivationArena,
    slots: usize,
) -> infer_core::Result<Staging> {
    let mut staging = Staging::new();
    for (index, node) in nodes.iter().enumerate().filter(|(_, node)| covers(node)) {
        for slot in 0..slots {
            let mut inputs = Vec::new();
            for &id in commit_inputs(node) {
                let size = arena.get(id)?.size() / arena.lanes();
                let row = || {
                    api::zeros::<f32>(&[size])
                        .sync_on(&device.stream)
                        .map_err(crate::device::device_error)
                };
                inputs.push([row()?, row()?, row()?]);
            }
            staging.insert((slot, index), inputs);
        }
    }
    Ok(staging)
}

/// Value-dimension block of the fused kernels: a quarter of the dimension, so a layer launches
/// four CTAs per value head (192 on the 27B rather than 48 for 170 SMs). Every reduction is
/// over the key dimension, so the split changes no summation order.
const fn fused_block(value_dim: usize) -> usize {
    if value_dim.is_multiple_of(FUSED_VALUE_SPLIT) {
        value_dim / FUSED_VALUE_SPLIT
    } else {
        value_dim
    }
}

/// Value-dimension split of the fused verify and commit kernels.
const FUSED_VALUE_SPLIT: usize = 4;

/// Whether `node` is a recurrent state writer the fused verify/commit pair covers.
pub(super) const fn covers(node: &TensorNode) -> bool {
    node.outputs.len() == 1
        && node.states.len() == 1
        && match node.op {
            TensorOp::Conv { kernel, .. } => kernel == crate::constants::CONV_KERNEL_SIZE,
            TensorOp::Delta {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
            } => {
                key_dim == value_dim
                    && key_dim.is_power_of_two()
                    && key_heads > 0
                    && value_heads.is_multiple_of(key_heads)
            }
            _ => false,
        }
}

/// A weight constant, or one lane's row of a batched activation.
fn lane_input<'a>(
    weights: &'a ProgramWeights,
    arena: &'a ActivationArena,
    id: TensorId,
    lane: usize,
) -> Result<TensorView<'a, f32>, DeviceError> {
    weights.constants.get(&id).map_or_else(
        || arena.row(id, lane).map_err(error),
        |constant| constant.view(&[constant.size()]).map_err(error),
    )
}

fn rows<'a>(
    weights: &'a ProgramWeights,
    arena: &'a ActivationArena,
    id: TensorId,
    slot: &SlotLanes<'_>,
) -> Result<[TensorView<'a, f32>; LANES], DeviceError> {
    let [first, second, third] = slot.lanes;
    Ok([
        lane_input(weights, arena, id, first)?,
        lane_input(weights, arena, id, second)?,
        lane_input(weights, arena, id, third)?,
    ])
}

/// Record the outputs of one slot's verification lanes for a covered recurrent node, reading
/// but never writing the slot's state.
#[expect(
    unsafe_code,
    reason = "Audited row alias: the three output rows are disjoint lanes of one arena slot, \
              which outlives the captured graph"
)]
pub(super) fn record_verify(
    scope: &Scope,
    node: &TensorNode,
    arena: &mut ActivationArena,
    states: &States,
    slot: &SlotLanes<'_>,
    weights: &ProgramWeights,
    staged: &mut [[Tensor<f32>; LANES]],
) -> Result<(), DeviceError> {
    stage_inputs(scope, node, arena, slot, staged)?;
    let index = arena.slot(node.outputs[0]).map_err(error)?;
    let buffer = arena.buffers[index]
        .take()
        .ok_or_else(|| error("missing slot verify output"))?;
    let lanes = arena.lanes();
    let recorded = (|| {
        let state = states
            .get(&node.states[0])
            .ok_or_else(|| error("slot verify state"))?;
        let [first, second, third] = slot.lanes;
        // SAFETY: see the expectation above.
        let o0 = unsafe { super::arena::row_view(&buffer, lanes, first) }.map_err(error)?;
        // SAFETY: see the expectation above.
        let o1 = unsafe { super::arena::row_view(&buffer, lanes, second) }.map_err(error)?;
        // SAFETY: see the expectation above.
        let o2 = unsafe { super::arena::row_view(&buffer, lanes, third) }.map_err(error)?;
        let [m0, m1, m2] = slot.metadata;
        match node.op {
            TensorOp::Conv { channels, .. } => {
                let [h0, h1, h2] = state.as_slice() else {
                    return Err(error("convolution state layout"));
                };
                let (mut o0, mut o1, mut o2) = (o0, o1, o2);
                let x = rows(weights, arena, node.inputs[0], slot)?;
                let weight = lane_input(weights, arena, node.inputs[1], slot.lanes[0])?;
                let w = weight.view(&[channels, crate::constants::CONV_KERNEL_SIZE])?;
                let tile = crate::constants::CONV_KERNEL_TILE;
                scope.record(fused::conv_verify3(
                    (&mut o0).partition([tile]),
                    (&mut o1).partition([tile]),
                    (&mut o2).partition([tile]),
                    &h0.view(&[h0.size()])?,
                    &h1.view(&[h1.size()])?,
                    &h2.view(&[h2.size()])?,
                    &x[0],
                    &x[1],
                    &x[2],
                    &w,
                    m0,
                    m1,
                    m2,
                ))?;
            }
            TensorOp::Delta {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
            } => {
                let [state] = state.as_slice() else {
                    return Err(error("delta state layout"));
                };
                let block = fused_block(value_dim);
                let shape = [value_heads, 1, value_dim];
                let (mut o0, mut o1, mut o2) = (
                    o0.reshape(&shape)?,
                    o1.reshape(&shape)?,
                    o2.reshape(&shape)?,
                );
                let qkv = rows(weights, arena, node.inputs[0], slot)?;
                let beta = rows(weights, arena, node.inputs[1], slot)?;
                let alpha = rows(weights, arena, node.inputs[2], slot)?;
                scope.record(
                    fused::delta_verify3(
                        (&mut o0).partition([1, 1, block]),
                        (&mut o1).partition([1, 1, block]),
                        (&mut o2).partition([1, 1, block]),
                        &state.view(&[value_heads, key_dim, value_dim])?,
                        &qkv[0],
                        &qkv[1],
                        &qkv[2],
                        &beta[0],
                        &beta[1],
                        &beta[2],
                        &alpha[0],
                        &alpha[1],
                        &alpha[2],
                        &lane_input(weights, arena, node.inputs[3], slot.lanes[0])?,
                        &lane_input(weights, arena, node.inputs[4], slot.lanes[0])?,
                        m0,
                        m1,
                        m2,
                    )
                    .generics(vec![
                        key_heads.to_string(),
                        value_heads.to_string(),
                        key_dim.to_string(),
                        block.to_string(),
                    ]),
                )?;
            }
            _ => return Err(error("slot verify recurrent node")),
        }
        Ok(())
    })();
    arena.buffers[index] = Some(buffer);
    recorded
}

/// Copy one slot's commit inputs out of the arena into its staging rows.
#[expect(
    unsafe_code,
    reason = "Audited row alias: the source rows lie inside a live arena slot that outlives the \
              captured graph"
)]
fn stage_inputs(
    scope: &Scope,
    node: &TensorNode,
    arena: &ActivationArena,
    slot: &SlotLanes<'_>,
    staged: &mut [[Tensor<f32>; LANES]],
) -> Result<(), DeviceError> {
    let ids = commit_inputs(node);
    if ids.len() != staged.len() {
        return Err(error("slot verify staging layout"));
    }
    for (&id, rows) in ids.iter().zip(staged) {
        let source = arena.get(id).map_err(error)?;
        for (lane, row) in slot.lanes.into_iter().zip(rows.iter_mut()) {
            // SAFETY: see the expectation above.
            let view =
                unsafe { super::arena::row_view(source, arena.lanes(), lane) }.map_err(error)?;
            scope.record(api::memcpy(row, &view))?;
        }
    }
    Ok(())
}

/// Record the commit of one slot's first `count` verification lanes for a covered node.
pub(super) fn record_commit(
    scope: &Scope,
    node: &TensorNode,
    arena: &ActivationArena,
    states: &mut States,
    slot: &SlotLanes<'_>,
    weights: &ProgramWeights,
    (staged, count): (&[[Tensor<f32>; LANES]], usize),
) -> Result<(), DeviceError> {
    let count = i32::try_from(count).map_err(error)?;
    let state = states
        .get_mut(&node.states[0])
        .ok_or_else(|| error("slot commit state"))?;
    let [m0, m1, m2] = slot.metadata;
    match node.op {
        TensorOp::Conv { .. } => {
            let [h0, h1, h2] = state.as_mut_slice() else {
                return Err(error("convolution state layout"));
            };
            let [x] = staged else {
                return Err(error("convolution staging layout"));
            };
            let tile = crate::constants::CONV_KERNEL_TILE;
            scope.record(fused::conv_commit3(
                h0.partition([tile]),
                h1.partition([tile]),
                h2.partition([tile]),
                &x[0],
                &x[1],
                &x[2],
                m0,
                m1,
                m2,
                count,
            ))?;
        }
        TensorOp::Delta {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        } => {
            let [state] = state.as_mut_slice() else {
                return Err(error("delta state layout"));
            };
            let block = fused_block(value_dim);
            let [qkv, beta, alpha] = staged else {
                return Err(error("delta staging layout"));
            };
            scope.record(
                fused::delta_commit3(
                    state.partition([1, key_dim, block]),
                    &qkv[0],
                    &qkv[1],
                    &qkv[2],
                    &beta[0],
                    &beta[1],
                    &beta[2],
                    &alpha[0],
                    &alpha[1],
                    &alpha[2],
                    &lane_input(weights, arena, node.inputs[3], slot.lanes[0])?,
                    &lane_input(weights, arena, node.inputs[4], slot.lanes[0])?,
                    m0,
                    m1,
                    m2,
                    count,
                )
                .generics(vec![
                    key_heads.to_string(),
                    value_heads.to_string(),
                    key_dim.to_string(),
                    block.to_string(),
                ]),
            )?;
        }
        _ => return Err(error("slot commit recurrent node")),
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/resident_slot_recurrent.rs"]
mod tests;
