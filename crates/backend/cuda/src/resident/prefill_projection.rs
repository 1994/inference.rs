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
    let input = arena.get(node.inputs[0]).map_err(error)?;
    let columns = input.size() / width;
    let input = input.view(&[width, columns]).map_err(error)?;
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
