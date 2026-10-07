use super::{ProgramWeights, batch::Lane, capture::error, linear_batch::batched};
use crate::{
    device::{CudaDevice, device_error},
    mlp::ProjectionWeight,
};
use cutile::{half::bf16, prelude::*};
use infer_core::{Error, OpId, Result};
use infer_ir::{DataflowGraph, TensorNode, TensorOp};
use std::collections::BTreeMap;

/// Rows of one verify tile: one row group of the fused GEMV. The kernel walks row groups
/// on grid dim 0, so any row count works; a partial group is predicated out.
const VERIFY_TILE_ROWS: usize = crate::constants::FUSED_VERIFY_LANES + 1;
/// Output columns per block. Measured on the slot decode hot path (3 lanes, n3 sweep):
/// 8 → 39.3 ms/step of linear, 16 → 28.6 ms; 32 crosses the register spill cliff of the
/// `[rows, BN, depth]` accumulator (16k f32 entries) and collapses to 88.9 ms.
const VERIFY_TILE_COLUMNS: usize = 16;
/// Inner depth of the verify GEMV k-loop.
const VERIFY_TILE_DEPTH: usize = 256;
/// Packed FP4 bytes per inner step: two nibble values share one byte.
const VERIFY_PACKED_DEPTH: usize = VERIFY_TILE_DEPTH / 2;
/// Scale groups per inner step: one E4M3 scale per NVFP4 group of 16 values.
const VERIFY_SCALE_DEPTH: usize = VERIFY_TILE_DEPTH / crate::constants::NVFP4_GROUP_SIZE;
/// Output tile of the batched verification GEMV.
const VERIFY_OUTPUT_TILE: [usize; 2] = [VERIFY_TILE_ROWS, VERIFY_TILE_COLUMNS];

pub(super) struct Workspace {
    nodes: BTreeMap<OpId, (usize, usize)>,
    buffers: BTreeMap<(usize, usize), (Tensor<f32>, Tensor<f32>)>,
}

pub(super) fn allocate(
    device: &CudaDevice,
    graph: &DataflowGraph,
    width: usize,
) -> Result<Workspace> {
    let mut buffers = Workspace {
        nodes: BTreeMap::new(),
        buffers: BTreeMap::new(),
    };
    if width < 2 {
        return Ok(buffers);
    }
    let sizes: BTreeMap<_, _> = graph
        .tensors
        .iter()
        .map(|t| Ok((t.id, t.elements()?)))
        .collect::<Result<_>>()?;
    let mut bytes = 0usize;
    for node in &graph.nodes {
        if node.op != TensorOp::Linear {
            continue;
        }
        let columns = sizes[&node.inputs[0]];
        let rows = sizes[&node.outputs[0]];
        let key = (rows, columns);
        buffers.nodes.insert(node.id, key);
        if buffers.buffers.contains_key(&key) {
            continue;
        }
        bytes = rows
            .checked_add(columns)
            .and_then(|n| n.checked_mul(width * crate::constants::F32_BYTES))
            .and_then(|n| bytes.checked_add(n))
            .ok_or_else(|| Error::invalid("batched projection workspace overflow"))?;
        if u64::try_from(bytes).map_err(device_error)? > device.profile()?.arena_budget_bytes() {
            return Err(Error::invalid("batched projection workspace budget"));
        }
        buffers.buffers.insert(
            key,
            (
                api::zeros::<f32>(&[width, columns])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
                api::zeros::<f32>(&[width, rows])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
            ),
        );
    }
    Ok(buffers)
}

pub(super) fn record(
    scope: &Scope,
    node: &TensorNode,
    lanes: &mut [Lane],
    weights: &ProgramWeights,
    buffers: &mut Workspace,
    nvfp4: &mut super::nvfp4_gemm::Workspace,
) -> std::result::Result<(), DeviceError> {
    let key = buffers
        .nodes
        .get(&node.id)
        .ok_or_else(|| error("batch workspace key"))?;
    let (input, output) = buffers
        .buffers
        .get_mut(key)
        .ok_or_else(|| error("batch projection workspace"))?;
    let width = lanes.len();
    let columns = input.size() / width;
    if width == crate::constants::FUSED_VERIFY_LANES {
        scope.record(batched::pack_inputs(
            (&mut *input).partition([1, crate::constants::AUX_KERNEL_TILE]),
            weights.constants.get(&node.inputs[0]).map_or_else(
                || lanes[0].arena.get(node.inputs[0]).map_err(error),
                |value| Ok(value.as_ref()),
            )?,
            weights.constants.get(&node.inputs[0]).map_or_else(
                || lanes[1].arena.get(node.inputs[0]).map_err(error),
                |value| Ok(value.as_ref()),
            )?,
            weights.constants.get(&node.inputs[0]).map_or_else(
                || lanes[2].arena.get(node.inputs[0]).map_err(error),
                |value| Ok(value.as_ref()),
            )?,
        ))?;
    } else {
        for (row, lane) in lanes.iter().enumerate() {
            let source = weights.constants.get(&node.inputs[0]).map_or_else(
                || lane.arena.get(node.inputs[0]).map_err(error),
                |value| Ok(value.as_ref()),
            )?;
            scope.record(batched::pack_row(
                (&mut *input).partition([1, crate::constants::AUX_KERNEL_TILE]),
                source,
                i32::try_from(row).map_err(error)?,
            ))?;
        }
    }
    let weight = weights
        .projections
        .get(&node.inputs[1])
        .ok_or_else(|| error("batch projection weight"))?;
    let packed = input.view(&[width, columns]).map_err(error)?;
    record_projection(
        scope,
        nvfp4,
        weight,
        weights.activation_quantization(node.inputs[1]),
        &packed,
        output,
        columns,
    )?;
    let mut outputs = Vec::new();
    let mut slots = Vec::new();
    for lane in lanes.iter_mut() {
        let slot = lane.arena.slot(node.outputs[0]).map_err(error)?;
        outputs.push(
            lane.arena.buffers[slot]
                .take()
                .ok_or_else(|| error("batch output"))?,
        );
        slots.push(slot);
    }
    if let [first, second, third] = outputs.as_mut_slice() {
        scope.record(batched::unpack_outputs(
            first.partition([crate::constants::AUX_KERNEL_TILE]),
            second.partition([crate::constants::AUX_KERNEL_TILE]),
            third.partition([crate::constants::AUX_KERNEL_TILE]),
            &*output,
        ))?;
    } else {
        for (row, target) in outputs.iter_mut().enumerate() {
            scope.record(batched::unpack_row(
                target.partition([crate::constants::AUX_KERNEL_TILE]),
                &*output,
                i32::try_from(row).map_err(error)?,
            ))?;
        }
    }
    for ((lane, slot), output) in lanes.iter_mut().zip(slots).zip(outputs) {
        lane.arena.buffers[slot] = Some(output);
    }
    Ok(())
}

/// Slot-decode twin of the verify [`record`]: the batched arena slot already holds every
/// lane, so the fused GEMV reads the `[lanes, K]` slot in place and stores straight into
/// the reshaped `[lanes, N]` output slot — the pack and unpack copies disappear.
pub(super) fn record_slots(
    scope: &Scope,
    node: &TensorNode,
    arena: &mut super::ActivationArena,
    weights: &ProgramWeights,
    nvfp4: &mut super::nvfp4_gemm::Workspace,
) -> std::result::Result<(), DeviceError> {
    if weights.constants.contains_key(&node.inputs[0]) {
        return Err(error("slot decode constant linear input"));
    }
    let lanes = arena.lanes();
    let slot = arena.slot(node.outputs[0]).map_err(error)?;
    let output = arena.buffers[slot]
        .take()
        .ok_or_else(|| error("slot decode output"))?;
    let input = arena.get(node.inputs[0]).map_err(error)?;
    let columns = input.size() / lanes;
    let input = input.view(&[lanes, columns]).map_err(error)?;
    let flat = output.size();
    let mut output = output.reshape(&[lanes, flat / lanes]).map_err(error)?;
    let weight = weights
        .projections
        .get(&node.inputs[1])
        .ok_or_else(|| error("slot projection weight"))?;
    record_projection(
        scope,
        nvfp4,
        weight,
        weights.activation_quantization(node.inputs[1]),
        &input,
        &mut output,
        columns,
    )?;
    arena.buffers[slot] = Some(output.reshape(&[flat]).map_err(error)?);
    Ok(())
}

fn record_projection(
    scope: &Scope,
    nvfp4: &mut super::nvfp4_gemm::Workspace,
    weight: &ProjectionWeight,
    input_scale: Option<super::program::ActivationQuantization>,
    x: &TensorView<'_, f32>,
    output: &mut Tensor<f32>,
    columns: usize,
) -> std::result::Result<(), DeviceError> {
    if super::nvfp4_gemm::record(scope, nvfp4, weight, input_scale, x, output, columns)? {
        return Ok(());
    }
    // Tensor cores amortize weights across candidate rows. Keep the measured
    // GEMV path for narrow batches with a very wide output (e.g. a vocabulary head).
    let rows = usize::try_from(x.shape()[0]).map_err(error)?;
    let outputs = output.size() / rows;
    if rows > crate::constants::SMALL_GEMV_MAX_ROWS
        || outputs <= columns.saturating_mul(crate::constants::GEMV_WIDE_OUTPUT_RATIO)
    {
        match weight {
            ProjectionWeight::Dense(w) => {
                scope.record(
                    super::small_gemm::kernels::dense(
                        output.partition(crate::constants::SMALL_GEMM_TILE),
                        x,
                        w,
                    )
                    .generics(vec![bf16::DTYPE.as_str().into(), columns.to_string()]),
                )?;
                return Ok(());
            }
            ProjectionWeight::Fp8(w, scale) => {
                scope.record(
                    super::small_gemm::kernels::scaled(
                        output.partition(crate::constants::SMALL_GEMM_TILE),
                        x,
                        w,
                        scale,
                    )
                    .generics(vec![
                        cuda_core::f8e4m3fn::DTYPE.as_str().into(),
                        columns.to_string(),
                    ]),
                )?;
                return Ok(());
            }
            ProjectionWeight::Fp4(..) => {}
        }
    }
    let mut generics = vec![
        VERIFY_TILE_COLUMNS.to_string(),
        VERIFY_TILE_DEPTH.to_string(),
        columns.to_string(),
    ];
    match weight {
        ProjectionWeight::Dense(w) => {
            generics.insert(0, bf16::DTYPE.as_str().into());
            scope.record(
                batched::dense(output.partition(VERIFY_OUTPUT_TILE), x, w).generics(generics),
            )?;
        }
        ProjectionWeight::Fp8(w, s) => {
            scope.record(
                batched::fp8(output.partition(VERIFY_OUTPUT_TILE), x, w, s).generics(generics),
            )?;
        }
        ProjectionWeight::Fp4(w, s, g) => {
            generics.extend([
                VERIFY_PACKED_DEPTH.to_string(),
                VERIFY_SCALE_DEPTH.to_string(),
            ]);
            scope.record(
                batched::nvfp4(output.partition(VERIFY_OUTPUT_TILE), x, w, s, g.recip())
                    .generics(generics),
            )?;
        }
    }
    Ok(())
}
