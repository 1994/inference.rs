//! Token-chunk recurrent execution: keep the state in registers across prompt rows.
#[cutile::module]
pub(crate) mod kernels {
    use cutile::core::{
        BroadcastScalar, LoadTileLike, Reshape, Shape_0, Shape_1, Shape_2, Shape_3,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tensor_3, Tile_0, Tile_1, Tile_2,
        cmp_ordering, cmpf, convert_scalar, exp, ftz, get_tile_block_id, log, predicate,
        reduce_sum, rsqrt, select, tile_to_scalar,
    };

    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA state kernel binds explicit tensors and the padding metadata"
    )]
    #[cutile::entry()]
    fn delta<const KH: i32, const VH: i32, const D: i32, const LANES: i32, const SW: i32>(
        state: &mut Tensor<f32, { [1, D, SW] }>,
        out: &mut Tensor<f32, { [1, LANES, SW] }>,
        qkv: &Tensor<f32, { [-1, -1, D] }>,
        beta: &Tensor<f32, { [-1, -1] }>,
        alpha: &Tensor<f32, { [-1, -1] }>,
        a_log: &Tensor<f32, { [-1] }>,
        bias: &Tensor<f32, { [-1] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let meta = metadata.partition(shape![1]);
        let base: i32 = tile_to_scalar(meta.load([0i32]).reshape(shape![]));
        let count: i32 = tile_to_scalar(meta.load([1i32]).reshape(shape![]));
        let offset: i32 = tile_to_scalar(meta.load([2i32]).reshape(shape![]));
        // One block of the value dimension. The state's columns are mutually independent - every
        // reduction below is over the key dimension - so the grid gains SW times more blocks
        // without changing any summation order.
        let mut old = state.load_like(state).reshape(shape![D, SW]);
        for lane in 0i32..LANES {
            if lane < count && base + lane + offset >= 0 {
                let key = pid.0 / (VH / KH);
                let qp = qkv.partition(shape![1, 1, D]);
                let mut q = qp.load([lane, key, 0i32]).reshape(shape![D]);
                let mut k = qp.load([lane, KH + key, 0i32]).reshape(shape![D]);
                let vp = qkv.partition(shape![1, 1, SW]);
                let v = vp
                    .load([lane, 2i32 * KH + pid.0, pid.2])
                    .reshape(shape![SW]);
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
                let bt = beta
                    .partition(shape![1, 1])
                    .load([lane, pid.0])
                    .reshape(shape![1]);
                let bt = one / (one + exp(zero - bt));
                let at = alpha
                    .partition(shape![1, 1])
                    .load([lane, pid.0])
                    .reshape(shape![1])
                    + bias.partition(shape![1]).load([pid.0]);
                let soft = log(one + exp(at));
                let large = cmpf(
                    at,
                    20.0f32.broadcast(shape![1]),
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                );
                let soft = select(large, at, soft);
                let decay = exp(zero - exp(a_log.partition(shape![1]).load([pid.0])) * soft);
                let decayed = old * decay.reshape(shape![1, 1]).broadcast(shape![D, SW]);
                let kb = k.reshape(shape![D, 1]).broadcast(shape![D, SW]);
                let predicted: Tile<f32, { [SW] }> = reduce_sum(decayed * kb, 0i32);
                let predicted = predicted.reshape(shape![SW]);
                let diff = (v - predicted) * bt.broadcast(shape![SW]);
                let updated: Tile<f32, { [D, SW] }> =
                    decayed + kb * diff.reshape(shape![1, SW]).broadcast(shape![D, SW]);
                let qb = q.reshape(shape![D, 1]).broadcast(shape![D, SW]);
                let result: Tile<f32, { [SW] }> = reduce_sum(updated * qb, 0i32);
                old = updated;
                out.partition_mut(shape![1, 1, SW])
                    .store(result.reshape(shape![1, 1, SW]), [0i32, lane, 0i32]);
            } else {
                out.partition_mut(shape![1, 1, SW])
                    .store(0.0f32.broadcast(shape![1, 1, SW]), [0i32, lane, 0i32]);
            }
        }
        state.store(old.reshape(shape![1, D, SW]));
    }
    #[cutile::entry()]
    fn transpose<const D: i32>(
        out: &mut Tensor<f32, { [1, 1, D] }>,
        input: &Tensor<f32, { [-1, -1, D] }>,
    ) {
        let pid = get_tile_block_id();
        out.store(input.partition(shape![1, 1, D]).load([pid.1, pid.0, 0i32]));
    }
}

#[cfg(test)]
#[path = "../../tests/unit/resident_recurrent_prefill.rs"]
mod tests;

mod workspace;
pub use workspace::Workspace;

#[cfg(test)]
#[path = "../../tests/unit/resident_recurrent_prefill_integration_tests.rs"]
mod integration;

#[cfg(test)]
#[path = "../../tests/unit/resident_recurrent_prefill_benchmark_tests.rs"]
mod benchmark;

#[cfg(test)]
#[path = "../../tests/unit/resident_recurrent_prefill_model_check.rs"]
mod model_check;

#[cfg(test)]
thread_local! {
    pub(super) static LEGACY_CAPTURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
