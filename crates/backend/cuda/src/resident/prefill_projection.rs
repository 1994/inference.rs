use super::{ProgramWeights, capture::error, prefill_gemm::gemm};
use crate::mlp::ProjectionWeight;
use cutile::prelude::*;
use infer_ir::TensorNode;
use std::collections::BTreeMap;

/// Zero-copy prompt GEMM: the batched input slot holds all 32 rows and the
/// output slot is reshaped to `[32, rows]` in place; no packing kernels remain.
/// Fp4 weights split the K loop across `PREFILL_SPLIT_K` CTAs per output tile:
/// the prompt GEMM is latency-bound at one wave, so wider grids hide DRAM latency.
pub(super) fn record(
    scope: &Scope,
    node: &TensorNode,
    arena: &mut super::ActivationArena,
    weights: &ProgramWeights,
    partials: &mut BTreeMap<usize, Tensor<f32>>,
    nvfp4: &mut super::nvfp4_gemm::Workspace,
    delegated: Option<&mut crate::device::cublaslt::Support>,
) -> Result<(), DeviceError> {
    if weights.constants.contains_key(&node.inputs[0]) {
        return Err(error("prefill32 constant linear input"));
    }
    let width = arena.lanes();
    let slot = arena.slot(node.outputs[0]).map_err(error)?;
    let output = arena.buffers[slot]
        .take()
        .ok_or_else(|| error("prefill output"))?;
    let flat = output.size();
    let mut output = output.reshape(&[width, flat / width])?;
    let flat_input = arena.get(node.inputs[0]).map_err(error)?;
    let columns = flat_input.size() / width;
    let input = flat_input.view(&[width, columns]).map_err(error)?;
    let weight = weights
        .projections
        .get(&node.inputs[1])
        .ok_or_else(|| error("prefill projection"))?;
    if super::nvfp4_gemm::record(
        scope,
        nvfp4,
        weight,
        weights.activation_quantization(node.inputs[1]),
        &input,
        &mut output,
        columns,
    )? {
        arena.buffers[slot] = Some(output.reshape(&[flat])?);
        return Ok(());
    }
    let generics = vec![columns.to_string()];
    match weight {
        ProjectionWeight::Fp8Block(..) => {
            return Err(error("block FP8 requires the resident FP8 GEMM"));
        }
        ProjectionWeight::Dense(w) => {
            // Wide projections are worth a vendor GEMM, square ones are not: `record_dense`
            // decides from the measured shape table.
            let rows = flat / width;
            let dims = (width, rows, columns);
            if records_vendor_gemm(scope, delegated, flat_input, w, &output, dims)? {
                arena.buffers[slot] = Some(output.reshape(&[flat])?);
                return Ok(());
            }
            scope.record(
                gemm::dense(
                    (&mut output).partition([
                        crate::constants::PREFILL_LANES,
                        crate::constants::PREFILL_GEMM_TILE_N,
                    ]),
                    &input,
                    w,
                )
                .generics(generics),
            )?;
        }
        ProjectionWeight::Fp8(w, scales) => {
            scope.record(
                gemm::fp8(
                    (&mut output).partition([
                        crate::constants::PREFILL_LANES,
                        crate::constants::PREFILL_GEMM_TILE_N,
                    ]),
                    &input,
                    w,
                    scales,
                )
                .generics(generics),
            )?;
        }
        ProjectionWeight::Fp4(w, scales, global) => {
            if width != crate::constants::PREFILL_LANES {
                return Err(error("legacy FP4 split prefill requires width 32"));
            }
            let rows = flat / width;
            let partial = partials
                .get_mut(&rows)
                .ok_or_else(|| error("prefill split partials"))?;
            let windows = columns.div_ceil(crate::constants::PREFILL_GEMM_TILE_K);
            let k_tiles = i32::try_from(windows.div_ceil(crate::constants::PREFILL_SPLIT_K))
                .map_err(|_| error("prefill split window"))?;
            scope.record(
                gemm::nvfp4_split(
                    partial.partition([width, crate::constants::PREFILL_NVFP4_TILE_N]),
                    &input,
                    w,
                    scales,
                    k_tiles,
                )
                .generics(generics),
            )?;
            scope.record(gemm::reduce_split(
                (&mut output).partition([width, crate::constants::PREFILL_NVFP4_TILE_N]),
                &*partial,
                global.recip(),
                i32::try_from(crate::constants::PREFILL_SPLIT_K)
                    .map_err(|_| error("prefill split count"))?,
            ))?;
        }
    }
    arena.buffers[slot] = Some(output.reshape(&[flat])?);
    Ok(())
}

/// Whether the vendor GEMM recorded this projection instead of the tile kernel.
fn records_vendor_gemm(
    scope: &Scope,
    delegated: Option<&mut crate::device::cublaslt::Support>,
    input: &Tensor<f32>,
    weight: &Tensor<cutile::half::bf16>,
    output: &Tensor<f32>,
    (width, rows, columns): (usize, usize, usize),
) -> Result<bool, DeviceError> {
    let Some(delegated) = delegated else {
        return Ok(false);
    };
    delegated.record_dense(scope, input, weight, output, (width, rows, columns))
}

/// Gather the last valid head input while retaining checkpoint input quantization,
/// prompt tile geometry and K accumulation order; the MMA masks unused rows.
pub(super) fn record_last(
    scope: &Scope,
    node: &TensorNode,
    arena: &mut super::ActivationArena,
    weights: &ProgramWeights,
    metadata: &Tensor<i32>,
    scratch: &mut Tensor<f32>,
    nvfp4: &mut super::nvfp4_gemm::Workspace,
) -> Result<(), DeviceError> {
    let columns = scratch.size();
    let input = arena
        .get(node.inputs[0])
        .map_err(error)?
        .view(&[arena.lanes(), columns])
        .map_err(error)?;
    scope.record(last_row::gather(scratch.partition([128]), &input, metadata))?;
    let slot = arena.slot(node.outputs[0]).map_err(error)?;
    let output = arena.buffers[slot]
        .take()
        .ok_or_else(|| error("prompt logits output"))?;
    let flat = output.size();
    let mut output = output.reshape(&[1, flat])?;
    let input = scratch.view(&[1, columns])?;
    let weight = weights
        .projections
        .get(&node.inputs[1])
        .ok_or_else(|| error("prompt head weight"))?;
    if matches!(
        weight,
        ProjectionWeight::Fp8(..) | ProjectionWeight::Fp8Block(..)
    ) && matches!(
        weights.activation_quantization(node.inputs[1]),
        Some(super::program::ActivationQuantization::Fp8Token)
    ) {
        nvfp4.record_fp8_tiled(
            scope,
            weight,
            &input,
            &mut output,
            columns,
            crate::constants::quant_gemm_tile(arena.lanes()),
        )?;
    } else {
        let generics = vec![columns.to_string()];
        match weight {
            ProjectionWeight::Dense(weight) => {
                scope.record(
                    gemm::dense((&mut output).partition([32, 64]), &input, weight)
                        .generics(generics),
                )?;
            }
            ProjectionWeight::Fp8(weight, scales) => {
                scope.record(
                    gemm::fp8((&mut output).partition([32, 64]), &input, weight, scales)
                        .generics(generics),
                )?;
            }
            _ => return Err(error("unsupported compact prompt head")),
        }
    }
    arena.buffers[slot] = Some(output.reshape(&[flat])?);
    Ok(())
}

#[cutile::module]
mod last_row {
    use cutile::core::{
        Reshape, Shape_0, Shape_1, Shape_2, StoreTileAtCurrentBlock, Tensor_1, Tensor_2,
        get_tile_block_id, tile_to_scalar,
    };
    #[cutile::entry()]
    fn gather(
        out: &mut Tensor<f32, { [128] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let count: i32 =
            tile_to_scalar(metadata.partition(shape![1]).load([1i32]).reshape(shape![]));
        out.store(
            input
                .partition(shape![1, 128])
                .load([count - 1i32, pid.0])
                .reshape(shape![128]),
        );
    }
}

#[cfg(test)]
thread_local! {
    pub(super) static FULL_LOGITS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
#[path = "../../tests/unit/resident_prefill_projection.rs"]
mod tests;
