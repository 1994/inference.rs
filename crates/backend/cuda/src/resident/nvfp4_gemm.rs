//! Model-declared W4A4 projection with block-16 E4M3 activation scales.
mod quant_reuse;
mod workspace;
pub use quant_reuse::QuantPairPlan;
#[cfg(test)]
pub use quant_reuse::with_quant_reuse_disabled;
pub use workspace::Workspace;
#[cutile::module]
pub(crate) mod kernels {
    use cutile::core::{
        BroadcastScalar, PackF4e2m1fnx2Tile, Reshape, Shape_2, Shape_3, StoreTileAtCurrentBlock,
        Tensor_2, Tile_2, UnpackF4e2m1fnx2Tile, absf, constant, convert_tile, eq_tile, exp,
        f4e2m1fn, f4e2m1fnx2, f8e4m3fn, get_tile_block_id, max_tile, min_tile, mmaf_scaled,
        reduce_max, select,
    };

    /// Fused activation quantization, kept at a fixed 16-row tile. The production W4A4 path
    /// quantizes once per program and then runs the row-tiled [`packed`] kernel; this fused
    /// variant only backs the fused-versus-separated measurement in `tests`.
    #[cutile::entry()]
    fn matmul<const K: i32>(
        out: &mut Tensor<f32, { [16, 64] }>,
        input: &Tensor<f32, { [-1, K] }>,
        weight: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        input_global: f32,
        alpha: f32,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![16, 512]);
        let wp = weight.partition(shape![64, 256]);
        let sp = scales.partition(shape![64, 32]);
        let mut acc: Tile<f32, { [16, 64] }> = constant(0.0f32, shape![16, 64]);
        for k in 0i32..((K + 511) / 512) {
            let x = xp.load([pid.0, k]);
            let groups = x.reshape(shape![16, 32, 16]);
            let maximum: Tile<f32, { [16, 32] }> = reduce_max(absf(groups), 2i32);
            let scale = min_tile(
                maximum * (input_global / 6.0f32).broadcast(shape![16, 32]),
                448.0f32.broadcast(shape![16, 32]),
            );
            let encoded: Tile<f8e4m3fn, { [16, 32] }> = convert_tile(scale);
            let rounded: Tile<f32, { [16, 32] }> = convert_tile(encoded);
            let zero = eq_tile(rounded, 0.0f32.broadcast(shape![16, 32]));
            let denominator = select(zero, 1.0f32.broadcast(shape![16, 32]), rounded);
            let multiplier = input_global.broadcast(shape![16, 32]) / denominator;
            let normalized = groups
                * multiplier
                    .reshape(shape![16, 32, 1])
                    .broadcast(shape![16, 32, 16]);
            let clipped = min_tile(
                max_tile(normalized, (-6.0f32).broadcast(shape![16, 32, 16])),
                6.0f32.broadcast(shape![16, 32, 16]),
            );
            let quantized: Tile<f4e2m1fn, { [16, 512] }> =
                convert_tile(clipped.reshape(shape![16, 512]));
            let w = wp.load([pid.1, k]).unpack(shape![64, 512]).transpose();
            let ws = sp.load([pid.1, k]).transpose();
            acc = mmaf_scaled(quantized, w, acc, encoded, ws);
        }
        out.store(acc * alpha.broadcast(shape![16, 64]));
    }
    /// Quantize each activation group once, shared by all output-column tiles.
    #[cutile::entry()]
    fn quantize<const K: i32>(
        out: &mut Tensor<f4e2m1fnx2, { [1, 256] }>,
        scales: &mut Tensor<f8e4m3fn, { [1, 32] }>,
        input: &Tensor<f32, { [-1, K] }>,
        input_global: f32,
    ) {
        let pid = get_tile_block_id();
        let x = input.partition(shape![1, 512]).load([pid.0, pid.1]);
        let groups = x.reshape(shape![1, 32, 16]);
        let maximum: Tile<f32, { [1, 32] }> = reduce_max(absf(groups), 2i32);
        let scale = min_tile(
            maximum * (input_global / 6.0f32).broadcast(shape![1, 32]),
            448.0f32.broadcast(shape![1, 32]),
        );
        let encoded: Tile<f8e4m3fn, { [1, 32] }> = convert_tile(scale);
        let rounded: Tile<f32, { [1, 32] }> = convert_tile(encoded);
        let zero = eq_tile(rounded, 0.0f32.broadcast(shape![1, 32]));
        let denominator = select(zero, 1.0f32.broadcast(shape![1, 32]), rounded);
        let normalized = groups
            * (input_global.broadcast(shape![1, 32]) / denominator)
                .reshape(shape![1, 32, 1])
                .broadcast(shape![1, 32, 16]);
        let clipped = min_tile(
            max_tile(normalized, (-6.0f32).broadcast(shape![1, 32, 16])),
            6.0f32.broadcast(shape![1, 32, 16]),
        );
        let values: Tile<f4e2m1fn, { [1, 512] }> = convert_tile(clipped.reshape(shape![1, 512]));
        out.store(values.pack(shape![1, 256]));
        scales.store(encoded);
    }

    /// `silu(gate) * up` quantized like [`quantize`], so a gated MLP's down projection reads its
    /// activation codes without materializing the SiLU and product rows. The SiLU and product
    /// keep the element-wise kernels' arithmetic: `g / (1 + exp(-g))`, then one multiply. Each
    /// block covers sixteen rows of one 512-column window.
    #[cutile::entry()]
    fn swiglu_quantize<const K: i32>(
        out: &mut Tensor<f4e2m1fnx2, { [16, 256] }>,
        scales: &mut Tensor<f8e4m3fn, { [16, 32] }>,
        gate: &Tensor<f32, { [-1, K] }>,
        up: &Tensor<f32, { [-1, K] }>,
        input_global: f32,
    ) {
        let pid = get_tile_block_id();
        let g = gate.partition(shape![16, 512]).load([pid.0, pid.1]);
        let u = up.partition(shape![16, 512]).load([pid.0, pid.1]);
        let negative: Tile<f32, { [16, 512] }> = 0.0f32.broadcast(shape![16, 512]) - g;
        let activated = g / (1.0f32.broadcast(shape![16, 512]) + exp(negative));
        let x = activated * u;
        let groups = x.reshape(shape![16, 32, 16]);
        let maximum: Tile<f32, { [16, 32] }> = reduce_max(absf(groups), 2i32);
        let scale = min_tile(
            maximum * (input_global / 6.0f32).broadcast(shape![16, 32]),
            448.0f32.broadcast(shape![16, 32]),
        );
        let encoded: Tile<f8e4m3fn, { [16, 32] }> = convert_tile(scale);
        let rounded: Tile<f32, { [16, 32] }> = convert_tile(encoded);
        let zero = eq_tile(rounded, 0.0f32.broadcast(shape![16, 32]));
        let denominator = select(zero, 1.0f32.broadcast(shape![16, 32]), rounded);
        let normalized = groups
            * (input_global.broadcast(shape![16, 32]) / denominator)
                .reshape(shape![16, 32, 1])
                .broadcast(shape![16, 32, 16]);
        let clipped = min_tile(
            max_tile(normalized, (-6.0f32).broadcast(shape![16, 32, 16])),
            6.0f32.broadcast(shape![16, 32, 16]),
        );
        let values: Tile<f4e2m1fn, { [16, 512] }> = convert_tile(clipped.reshape(shape![16, 512]));
        out.store(values.pack(shape![16, 256]));
        scales.store(encoded);
    }

    #[cutile::entry()]
    fn packed<const K: i32, const M: i32, const N: i32>(
        out: &mut Tensor<f32, { [M, N] }>,
        input: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        input_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        weight: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        alpha: f32,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![M, 128]);
        let xs = input_scales.partition(shape![M, 16]);
        let wp = weight.partition(shape![N, 128]);
        let sp = scales.partition(shape![N, 16]);
        let mut acc: Tile<f32, { [M, N] }> = constant(0.0f32, shape![M, N]);
        for k in 0i32..((K + 255) / 256) {
            let x = xp.load([pid.0, k]).unpack(shape![M, 256]);
            let w = wp.load([pid.1, k]).unpack(shape![N, 256]).transpose();
            acc = mmaf_scaled(
                x,
                w,
                acc,
                xs.load([pid.0, k]),
                sp.load([pid.1, k]).transpose(),
            );
        }
        out.store(acc * alpha.broadcast(shape![M, N]));
    }

    /// Prefill main loop with compiler-scheduled operand and scale prefetching.
    /// Keep the native 256-element K step and block-16 scale arithmetic unchanged.
    /// Output geometry is chosen from shape and driver-reported parallelism at capture time.
    #[cutile::entry()]
    fn packed_prefill<const K: i32, const M: i32, const N: i32>(
        out: &mut Tensor<f32, { [M, N] }>,
        input: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        input_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        weight: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        alpha: f32,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![M, 128]);
        let xs = input_scales.partition(shape![M, 16]);
        let wp = weight.partition(shape![N, 128]);
        let sp = scales.partition(shape![N, 16]);
        let mut acc: Tile<f32, { [M, N] }> = constant(0.0f32, shape![M, N]);
        for k in 0i32..((K + 255) / 256) {
            let x = xp.load_pipelined::<8>([pid.0, k]).unpack(shape![M, 256]);
            let w = wp
                .load_pipelined::<8>([pid.1, k])
                .unpack(shape![N, 256])
                .transpose();
            acc = mmaf_scaled(
                x,
                w,
                acc,
                xs.load_pipelined::<8>([pid.0, k]),
                sp.load_pipelined::<8>([pid.1, k]).transpose(),
            );
        }
        out.store(acc * alpha.broadcast(shape![M, N]));
    }
}

/// Record only when the checkpoint declares an activation scale for packed FP4 weights.
pub(super) fn record(
    scope: &cutile::prelude::Scope,
    workspace: &mut Workspace,
    weight: &crate::mlp::ProjectionWeight,
    input_scale: Option<super::program::ActivationQuantization>,
    input: &cutile::prelude::TensorView<'_, f32>,
    output: &mut cutile::prelude::Tensor<f32>,
    columns: usize,
) -> Result<bool, cutile::prelude::DeviceError> {
    workspace.clear_quant_reuse();
    record_input(
        scope,
        workspace,
        weight,
        input_scale,
        input,
        output,
        columns,
        None,
    )
}

/// The caller supplies the pointer of the tensor from which this exact view was made.
#[expect(
    clippy::too_many_arguments,
    reason = "The certified view adds its actual device pointer to the unchanged projection contract"
)]
pub(super) fn record_with_input_pointer(
    scope: &cutile::prelude::Scope,
    workspace: &mut Workspace,
    weight: &crate::mlp::ProjectionWeight,
    input_scale: Option<super::program::ActivationQuantization>,
    input: &cutile::prelude::TensorView<'_, f32>,
    output: &mut cutile::prelude::Tensor<f32>,
    columns: usize,
    input_pointer: u64,
) -> Result<bool, cutile::prelude::DeviceError> {
    record_input(
        scope,
        workspace,
        weight,
        input_scale,
        input,
        output,
        columns,
        Some(input_pointer),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "One dispatcher preserves both certified and legacy projection contracts"
)]
fn record_input(
    scope: &cutile::prelude::Scope,
    workspace: &mut Workspace,
    weight: &crate::mlp::ProjectionWeight,
    input_scale: Option<super::program::ActivationQuantization>,
    input: &cutile::prelude::TensorView<'_, f32>,
    output: &mut cutile::prelude::Tensor<f32>,
    columns: usize,
    input_pointer: Option<u64>,
) -> Result<bool, cutile::prelude::DeviceError> {
    use super::program::ActivationQuantization;
    use crate::mlp::ProjectionWeight;
    match (weight, input_scale) {
        (ProjectionWeight::Fp4(..), Some(ActivationQuantization::Fp4(scale))) => {
            workspace.record_input(scope, weight, scale, input, output, columns, input_pointer)?;
        }
        (
            ProjectionWeight::Fp8(..) | ProjectionWeight::Fp8Block(..),
            Some(ActivationQuantization::Fp8Token),
        ) => {
            workspace.record_fp8_input(scope, weight, input, output, columns, input_pointer)?;
        }
        _ => {
            workspace.clear_quant_reuse();
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
#[path = "../../tests/unit/resident_nvfp4_gemm.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_nvfp4_pipeline.rs"]
mod pipeline_tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_vendor_fp8.rs"]
mod vendor_tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_quant_reuse.rs"]
mod quant_reuse_tests;
