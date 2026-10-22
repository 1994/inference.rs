//! Exact token FP8 quantization with parallel conversion for large aligned rows.
#[cutile::module]
pub(crate) mod kernels {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_2, StoreTileAtCurrentBlock, Tensor_2, Tile_1, Tile_2, absf,
        convert_tile, f8e4m3fn, get_tile_block_id, max_tile, min_tile, reduce_max,
    };
    #[cutile::entry()]
    fn maximum<const K: i32, const PAD: i32>(
        scale: &mut Tensor<f32, { [1, 1] }>,
        input: &Tensor<f32, { [-1, K] }>,
    ) {
        let pid = get_tile_block_id();
        let x = input.partition(shape![1, PAD]).load([pid.0, 0i32]);
        let maximum: Tile<f32, { [1] }> = reduce_max(absf(x), 1i32);
        let s = max_tile(
            maximum.reshape(shape![1, 1]) / 448.0f32.broadcast(shape![1, 1]),
            1.0e-12f32.broadcast(shape![1, 1]),
        );
        scale.store(s);
    }
    #[cutile::entry()]
    fn quantize<const K: i32>(
        codes: &mut Tensor<f8e4m3fn, { [1, 512] }>,
        scale: &Tensor<f32, { [-1, 1] }>,
        input: &Tensor<f32, { [-1, K] }>,
    ) {
        let pid = get_tile_block_id();
        let s = scale.partition(shape![1, 1]).load([pid.0, 0i32]);
        let x = input.partition(shape![1, 512]).load([pid.0, pid.1]);
        let x = x / s.broadcast(shape![1, 512]);
        let clipped = min_tile(
            max_tile(x, (-448.0f32).broadcast(shape![1, 512])),
            448.0f32.broadcast(shape![1, 512]),
        );
        let packed: Tile<f8e4m3fn, { [1, 512] }> = convert_tile(clipped);
        codes.store(packed);
    }
}

use cutile::prelude::DeviceError;
use cutile::prelude::Scope;
use cutile::prelude::Tensor as PreludeTensor;
use cutile::prelude::TensorView;
use cutile::prelude::{PartitionMut, TileKernel};
#[cfg(test)]
use std::cell::Cell;
const SERIAL_COLUMNS: usize = 4096;
const CONVERSION_COLUMNS: usize = 512;

pub(super) const fn parallel_columns(columns: usize) -> bool {
    columns > SERIAL_COLUMNS && columns.is_multiple_of(CONVERSION_COLUMNS)
}

/// Consume a maximum produced by the immediately preceding normalization, if present.
pub(super) fn record_with_maximum(
    scope: &Scope,
    codes: &mut PreludeTensor<cuda_core::f8e4m3fn>,
    scale: &mut PreludeTensor<f32>,
    input: &TensorView<'_, f32>,
    columns: usize,
    ready: bool,
) -> Result<(), DeviceError> {
    if !ready {
        return record(scope, codes, scale, input, columns);
    }
    scope.record(
        kernels::quantize(codes.partition([1, CONVERSION_COLUMNS]), &*scale, input)
            .generics(vec![columns.to_string()]),
    )?;
    Ok(())
}

/// Large-row conversion is partitioned independently from the original row maximum.
/// Both stages reuse the existing quantized activation and scale buffers.
pub(super) fn record(
    scope: &Scope,
    codes: &mut PreludeTensor<cuda_core::f8e4m3fn>,
    scale: &mut PreludeTensor<f32>,
    input: &TensorView<'_, f32>,
    columns: usize,
) -> Result<(), DeviceError> {
    if !parallel_columns(columns) || !enabled() {
        scope.record(
            super::fp8_gemm::kernels::quantize(
                (&mut *codes).partition([1, columns.next_power_of_two()]),
                (&mut *scale).partition([1, 1]),
                input,
            )
            .generics(vec![
                columns.to_string(),
                columns.next_power_of_two().to_string(),
            ]),
        )?;
    } else {
        scope.record(
            kernels::maximum((&mut *scale).partition([1, 1]), input).generics(vec![
                columns.to_string(),
                columns.next_power_of_two().to_string(),
            ]),
        )?;
        scope.record(
            kernels::quantize(
                (&mut *codes).partition([1, CONVERSION_COLUMNS]),
                &*scale,
                input,
            )
            .generics(vec![columns.to_string()]),
        )?;
        #[cfg(test)]
        CAPTURES.with(|value| value.set(value.get() + 1));
    }
    Ok(())
}
#[cfg(not(test))]
const fn enabled() -> bool {
    true
}
#[cfg(test)]
std::thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(true) };
    static CAPTURES: Cell<usize> = const { Cell::new(0) };
}
#[cfg(test)]
fn enabled() -> bool {
    ENABLED.with(Cell::get)
}
#[cfg(test)]
pub fn captures() -> usize {
    CAPTURES.with(Cell::get)
}
#[cfg(test)]
pub fn with_parallel_conversion<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ENABLED.with(|value| value.set(self.0));
        }
    }
    let _restore = Restore(ENABLED.with(|value| value.replace(enabled)));
    run()
}

#[cfg(test)]
#[path = "../../tests/unit/resident_fp8_quant_reduction.rs"]
mod tests;
