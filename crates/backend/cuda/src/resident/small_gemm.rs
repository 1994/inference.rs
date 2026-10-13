//! Small batched projections using tensor cores with a two-term BF16 input.
//! Retaining the input residual avoids reducing F32 activations to one BF16 value.
#[cutile::module]
pub(crate) mod kernels {
    use cutile::core::{
        ElementType, Reshape, Shape_1, Shape_2, StoreTileAtCurrentBlock, Tensor_1, Tensor_2,
        Tile_2, bf16, constant, convert_tile, get_tile_block_id, mmaf,
    };
    #[cutile::entry()]
    fn dense<E: ElementType, const K: i32>(
        out: &mut Tensor<f32, { [16, 32] }>,
        input: &Tensor<f32, { [-1, K] }>,
        weight: &Tensor<E, { [-1, K] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![16, 128]);
        let wp = weight.partition(shape![32, 128]);
        let mut acc: Tile<f32, { [16, 32] }> = constant(0.0f32, shape![16, 32]);
        for k in 0i32..((K + 127) / 128) {
            let input = xp.load([pid.0, k]);
            let high: Tile<bf16, { [16, 128] }> = convert_tile(input);
            let rounded: Tile<f32, { [16, 128] }> = convert_tile(high);
            let low: Tile<bf16, { [16, 128] }> = convert_tile(input - rounded);
            let weight: Tile<bf16, { [32, 128] }> = convert_tile(wp.load([pid.1, k]));
            acc = mmaf(low, weight.transpose(), acc);
            acc = mmaf(high, weight.transpose(), acc);
        }
        out.store(acc);
    }
    /// The third BF16 component reduces the residual input quantization error.
    /// This experiment is selected consistently by private and pooled capture.
    #[cutile::entry()]
    fn dense_precise<E: ElementType, const K: i32>(
        out: &mut Tensor<f32, { [16, 32] }>,
        input: &Tensor<f32, { [-1, K] }>,
        weight: &Tensor<E, { [-1, K] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![16, 128]);
        let wp = weight.partition(shape![32, 128]);
        let zero: Tile<f32, { [16, 32] }> = constant(0.0f32, shape![16, 32]);
        let mut acc: Tile<f32, { [16, 32] }> = zero;
        let mut correction: Tile<f32, { [16, 32] }> = zero;
        for k in 0i32..((K + 127) / 128) {
            let input = xp.load([pid.0, k]);
            let high: Tile<bf16, { [16, 128] }> = convert_tile(input);
            let rounded: Tile<f32, { [16, 128] }> = convert_tile(high);
            let residual = input - rounded;
            let low: Tile<bf16, { [16, 128] }> = convert_tile(residual);
            let rounded_low: Tile<f32, { [16, 128] }> = convert_tile(low);
            let tail: Tile<bf16, { [16, 128] }> = convert_tile(residual - rounded_low);
            let weight: Tile<bf16, { [32, 128] }> = convert_tile(wp.load([pid.1, k]));
            // Sum each component without injecting it into a large previous accumulator.
            // Keep block summation compensated so cancellation across K stays within the F64 gate.
            let high_product = mmaf(high, weight.transpose(), zero);
            let low_product = mmaf(low, weight.transpose(), zero);
            let tail_product = mmaf(tail, weight.transpose(), zero);
            let block = (high_product + low_product) + tail_product;
            let corrected = block - correction;
            let next = acc + corrected;
            correction = (next - acc) - corrected;
            acc = next;
        }
        out.store(acc);
    }

    #[cutile::entry()]
    fn scaled<E: ElementType, const K: i32>(
        out: &mut Tensor<f32, { [16, 32] }>,
        input: &Tensor<f32, { [-1, K] }>,
        weight: &Tensor<E, { [-1, K] }>,
        scale: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![16, 128]);
        let wp = weight.partition(shape![32, 128]);
        let mut acc: Tile<f32, { [16, 32] }> = constant(0.0f32, shape![16, 32]);
        for k in 0i32..((K + 127) / 128) {
            let input = xp.load([pid.0, k]);
            let high: Tile<bf16, { [16, 128] }> = convert_tile(input);
            let rounded: Tile<f32, { [16, 128] }> = convert_tile(high);
            let low: Tile<bf16, { [16, 128] }> = convert_tile(input - rounded);
            let weight: Tile<bf16, { [32, 128] }> = convert_tile(wp.load([pid.1, k]));
            acc = mmaf(low, weight.transpose(), acc);
            acc = mmaf(high, weight.transpose(), acc);
        }
        let scale = scale
            .partition(shape![32])
            .load([pid.1])
            .reshape(shape![1, 32])
            .broadcast(shape![16, 32]);
        out.store(acc * scale);
    }
}

#[cfg(test)]
#[path = "../../tests/unit/resident_small_gemm.rs"]
mod tests;

/// Experimental three-component BF16 arithmetic shared by private and pooled projection.
/// Default keeps the existing arithmetic until full model quality and complete costs pass.
pub(super) fn precise_dense_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if pool_f32_enabled() {
        return false;
    }
    #[cfg(test)]
    if let Some(enabled) = PRECISE_DENSE.with(std::cell::Cell::get) {
        return enabled;
    }
    *ENABLED.get_or_init(|| {
        std::env::var("INFER_CUDA_PRECISE_SMALL_DENSE")
            .is_ok_and(|value| value == "1" || value == "true")
    })
}

#[cfg(test)]
thread_local! {
    static PRECISE_DENSE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}
#[cfg(test)]
pub fn with_precise_dense<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PRECISE_DENSE.with(|cell| cell.set(self.0));
        }
    }
    let restore = Restore(PRECISE_DENSE.with(|cell| cell.replace(Some(enabled))));
    let result = run();
    drop(restore);
    result
}

/// Experimental four-slot F32 GEMV; private GEMV arithmetic and tile choice stay shared.
pub(super) fn pool_f32_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    #[cfg(test)]
    if let Some(enabled) = POOL_F32.with(std::cell::Cell::get) {
        return enabled;
    }
    *ENABLED.get_or_init(|| {
        std::env::var("INFER_CUDA_POOL_SMALL_F32")
            .is_ok_and(|value| value == "1" || value == "true")
    })
}
#[cfg(test)]
thread_local! { static POOL_F32: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) }; }
#[cfg(test)]
pub fn with_pool_f32<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            POOL_F32.with(|cell| cell.set(self.0));
        }
    }
    let restore = Restore(POOL_F32.with(|cell| cell.replace(Some(enabled))));
    let result = run();
    drop(restore);
    result
}

#[cfg(test)]
thread_local! { static POOL_RECORDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
#[cfg(test)]
pub(super) fn note_pool_record() {
    POOL_RECORDS.with(|count| count.set(count.get().saturating_add(1)));
}
#[cfg(test)]
pub fn pool_f32_capture_records() -> usize {
    POOL_RECORDS.with(std::cell::Cell::get)
}
