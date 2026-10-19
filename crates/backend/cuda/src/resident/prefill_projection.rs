use super::{ProgramWeights, capture::error, prefill_gemm::gemm};
use crate::{device::DeviceProfile, mlp::ProjectionWeight};
use cutile::prelude::*;
use infer_ir::TensorNode;
use std::collections::BTreeMap;

const NARROW_DENSE_SHAPE: (usize, usize, usize) = (64, 48, 5120);
const NARROW_DENSE_TILE: [usize; 2] = [crate::constants::PREFILL_LANES, 16];
/// Output columns and reduction width of the GDN `in_proj_b`/`in_proj_a` pair.
pub(super) const GDN_BA_SHAPE: [i32; 2] = [48, 5120];

/// Down projection absorbing `silu(gate) * up`: node indices of the `SiLU` and multiply it
/// replaces, then the gate and up tensors it reads.
pub(super) type Swiglu = (usize, usize, infer_core::TensorId, infer_core::TensorId);

/// Target-graph NVFP4 down projections whose input is `silu(gate) * up` with no other reader of
/// the `SiLU` or product rows. Their prompt capture quantizes straight from gate and up.
pub(super) fn swiglu_plan(
    graph: &infer_ir::DataflowGraph,
    weights: &ProgramWeights,
) -> BTreeMap<usize, Swiglu> {
    use infer_ir::TensorOp;
    let mut plan = BTreeMap::new();
    if weights.fusion.is_some()
        || std::env::var("INFER_CUDA_SWIGLU_FUSION").is_ok_and(|value| value == "0")
    {
        return plan;
    }
    let mut producer = BTreeMap::new();
    let mut readers: BTreeMap<_, Vec<usize>> = BTreeMap::new();
    for (index, node) in graph.nodes.iter().enumerate() {
        for id in &node.outputs {
            producer.insert(*id, index);
        }
        for id in &node.inputs {
            readers.entry(*id).or_default().push(index);
        }
    }
    let results = [graph.hidden, graph.logits];
    let sole = |id, reader: usize| {
        !results.contains(&Some(id)) && readers.get(&id).is_some_and(|r| r[..] == [reader])
    };
    for (index, node) in graph.nodes.iter().enumerate() {
        if node.op != TensorOp::Linear
            || !matches!(
                weights.projections.get(&node.inputs[1]),
                Some(ProjectionWeight::Fp4(..))
            )
            || !matches!(
                weights.activation_quantization(node.inputs[1]),
                Some(super::program::ActivationQuantization::Fp4(_))
            )
        {
            continue;
        }
        let product = node.inputs[0];
        let Some(&multiply) = producer.get(&product) else {
            continue;
        };
        let multiply_node = &graph.nodes[multiply];
        if multiply_node.op != TensorOp::Multiply || !sole(product, index) {
            continue;
        }
        let [left, right] = [multiply_node.inputs[0], multiply_node.inputs[1]];
        for (activated, up) in [(left, right), (right, left)] {
            if let Some(&silu) = producer.get(&activated)
                && graph.nodes[silu].op == TensorOp::Silu
                && sole(activated, multiply)
            {
                plan.insert(index, (silu, multiply, graph.nodes[silu].inputs[0], up));
                break;
            }
        }
    }
    plan
}

/// Record a [`swiglu_plan`] down projection. False leaves nothing recorded.
pub(super) fn record_swiglu(
    scope: &Scope,
    node: &TensorNode,
    (gate, up): (infer_core::TensorId, infer_core::TensorId),
    arena: &mut super::ActivationArena,
    weights: &ProgramWeights,
    nvfp4: &mut super::nvfp4_gemm::Workspace,
) -> Result<bool, DeviceError> {
    let Some(super::program::ActivationQuantization::Fp4(scale)) =
        weights.activation_quantization(node.inputs[1])
    else {
        return Ok(false);
    };
    let weight = weights
        .projections
        .get(&node.inputs[1])
        .ok_or_else(|| error("prefill projection"))?;
    let width = arena.lanes();
    let slot = arena.slot(node.outputs[0]).map_err(error)?;
    let output = arena.buffers[slot]
        .take()
        .ok_or_else(|| error("prefill output"))?;
    let flat = output.size();
    let mut output = output.reshape(&[width, flat / width])?;
    let gate = arena.get(gate).map_err(error)?;
    let columns = gate.size() / width;
    let gate = gate.view(&[width, columns]).map_err(error)?;
    let up = arena
        .get(up)
        .map_err(error)?
        .view(&[width, columns])
        .map_err(error)?;
    nvfp4.record_swiglu_input(scope, weight, scale, &gate, &up, &mut output, columns)?;
    drop((gate, up));
    arena.buffers[slot] = Some(output.reshape(&[flat])?);
    Ok(true)
}

/// Zero-copy prompt GEMM:/// Zero-copy prompt GEMM: the batched input slot holds all 32 rows and the
/// output slot is reshaped to `[32, rows]` in place; no packing kernels remain.
/// Fp4 weights split the K loop across `PREFILL_SPLIT_K` CTAs per output tile:
/// the prompt GEMM is latency-bound at one wave, so wider grids hide DRAM latency.
pub(super) fn record(
    scope: &Scope,
    node: &TensorNode,
    arena: &mut super::ActivationArena,
    (weights, profile): (&ProgramWeights, &DeviceProfile),
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
    let input_pointer = flat_input.device_pointer().cu_deviceptr();
    let columns = flat_input.size() / width;
    let input = flat_input.view(&[width, columns]).map_err(error)?;
    let weight = weights
        .projections
        .get(&node.inputs[1])
        .ok_or_else(|| error("prefill projection"))?;
    if super::nvfp4_gemm::record_with_input_pointer(
        scope,
        nvfp4,
        weight,
        weights.activation_quantization(node.inputs[1]),
        &input,
        &mut output,
        columns,
        input_pointer,
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
            if records_precise_sparse(scope, weights, &input, w, &mut output, columns)? {
                arena.buffers[slot] = Some(output.reshape(&[flat])?);
                return Ok(());
            }
            // Wide projections are worth a vendor GEMM, square ones are not: `record_dense`
            // decides from the measured shape table.
            let rows = flat / width;
            let dims = (width, rows, columns);
            if records_vendor_gemm(scope, delegated, flat_input, w, &output, dims)? {
                arena.buffers[slot] = Some(output.reshape(&[flat])?);
                return Ok(());
            }
            if records_narrow_dense(scope, profile, &input, w, &mut output, dims)? {
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
        ProjectionWeight::Fp4(..) => {
            record_legacy_fp4(
                scope,
                weight,
                &input,
                &mut output,
                partials,
                (width, columns),
                generics,
            )?;
        }
    }
    arena.buffers[slot] = Some(output.reshape(&[flat])?);
    Ok(())
}

/// Record GDN `in_proj_b` + `in_proj_a` as one launch: both are Dense BF16 N48
/// projections of the same normalized input. The pair kernel keeps each output's
/// independent K accumulation order; only the launch (and the shared `x` load) is fused.
///
/// Returns false when either node would take a different path (narrow N16 layout,
/// sparse experts, vendor GEMM, or unexpected shape). A 48x5120 weight is below
/// `MIN_DELEGATED_WEIGHT_ELEMENTS`, so the vendor path is impossible for this shape.
pub(super) fn record_pair(
    scope: &Scope,
    first: &TensorNode,
    second: &TensorNode,
    arena: &mut super::ActivationArena,
    (weights, profile): (&ProgramWeights, &DeviceProfile),
) -> Result<bool, DeviceError> {
    if first.inputs.len() != 2
        || second.inputs.len() != 2
        || first.inputs[0] != second.inputs[0]
        || first.outputs.len() != 1
        || second.outputs.len() != 1
        || !weights.expert_banks.is_empty()
    {
        return Ok(false);
    }
    let (Some(ProjectionWeight::Dense(weight_b)), Some(ProjectionWeight::Dense(weight_a))) = (
        weights.projections.get(&first.inputs[1]),
        weights.projections.get(&second.inputs[1]),
    ) else {
        return Ok(false);
    };
    if weight_b.shape() != GDN_BA_SHAPE || weight_a.shape() != weight_b.shape() {
        return Ok(false);
    }
    let width = arena.lanes();
    let expected = usize::try_from(GDN_BA_SHAPE[0]).map_err(error)? * width;
    let input_slot = arena.slot(first.inputs[0]).map_err(error)?;
    let input_elements = arena.buffers[input_slot]
        .as_ref()
        .ok_or_else(|| error("paired projection input"))?
        .size();
    let columns = input_elements / width;
    if columns * width != input_elements
        || columns != usize::try_from(GDN_BA_SHAPE[1]).map_err(error)?
        || arena.get(first.outputs[0]).map_err(error)?.size() != expected
        || arena.get(second.outputs[0]).map_err(error)?.size() != expected
        || narrow_dense_layout(
            &profile.architecture,
            (
                width,
                usize::try_from(GDN_BA_SHAPE[0]).map_err(error)?,
                columns,
            ),
        )
    {
        return Ok(false);
    }
    let slot_b = arena.slot(first.outputs[0]).map_err(error)?;
    let slot_a = arena.slot(second.outputs[0]).map_err(error)?;
    if slot_b == slot_a {
        return Ok(false);
    }
    let output_b = arena.buffers[slot_b]
        .take()
        .ok_or_else(|| error("paired projection b output"))?;
    let output_a = arena.buffers[slot_a]
        .take()
        .ok_or_else(|| error("paired projection a output"))?;
    let rows = usize::try_from(GDN_BA_SHAPE[0]).map_err(error)?;
    let mut output_b = output_b.reshape(&[width, rows])?;
    let mut output_a = output_a.reshape(&[width, rows])?;
    let flat_input = arena.buffers[input_slot]
        .as_ref()
        .ok_or_else(|| error("paired projection input"))?;
    let input = flat_input.view(&[width, columns]).map_err(error)?;
    scope.record(
        gemm::dense_pair(
            (&mut output_b).partition([
                crate::constants::PREFILL_LANES,
                crate::constants::PREFILL_GEMM_TILE_N,
            ]),
            (&mut output_a).partition([
                crate::constants::PREFILL_LANES,
                crate::constants::PREFILL_GEMM_TILE_N,
            ]),
            &input,
            weight_b,
            weight_a,
        )
        .generics(vec![columns.to_string()]),
    )?;
    arena.buffers[slot_b] = Some(output_b.reshape(&[expected])?);
    arena.buffers[slot_a] = Some(output_a.reshape(&[expected])?);
    Ok(true)
}

/// Record the established split-K projection with its existing operands and reduction.
fn record_legacy_fp4(
    scope: &Scope,
    weight: &ProjectionWeight,
    input: &TensorView<'_, f32>,
    output: &mut Tensor<f32>,
    partials: &mut BTreeMap<usize, Tensor<f32>>,
    (width, columns): (usize, usize),
    generics: Vec<String>,
) -> Result<(), DeviceError> {
    let ProjectionWeight::Fp4(w, scales, global) = weight else {
        return Err(error("legacy FP4 prompt weight"));
    };
    if width != crate::constants::PREFILL_LANES {
        return Err(error("legacy FP4 split prefill requires width 32"));
    }
    let rows = output.size() / width;
    let partial = partials
        .get_mut(&rows)
        .ok_or_else(|| error("prefill split partials"))?;
    let windows = columns.div_ceil(crate::constants::PREFILL_GEMM_TILE_K);
    let k_tiles = i32::try_from(windows.div_ceil(crate::constants::PREFILL_SPLIT_K))
        .map_err(|_| error("prefill split window"))?;
    scope.record(
        gemm::nvfp4_split(
            partial.partition([width, crate::constants::PREFILL_NVFP4_TILE_N]),
            input,
            w,
            scales,
            k_tiles,
        )
        .generics(generics),
    )?;
    scope.record(gemm::reduce_split(
        output.partition([width, crate::constants::PREFILL_NVFP4_TILE_N]),
        &*partial,
        global.recip(),
        i32::try_from(crate::constants::PREFILL_SPLIT_K)
            .map_err(|_| error("prefill split count"))?,
    ))?;
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

// Preserve F32 router inputs through a high BF16 term and a low residual term.
fn records_precise_sparse(
    scope: &Scope,
    weights: &ProgramWeights,
    input: &TensorView<'_, f32>,
    weight: &Tensor<cutile::half::bf16>,
    output: &mut Tensor<f32>,
    columns: usize,
) -> Result<bool, DeviceError> {
    if weights.expert_banks.is_empty() {
        return Ok(false);
    }
    scope.record(
        super::small_gemm::kernels::dense(
            output.partition(crate::constants::SMALL_GEMM_TILE),
            input,
            weight,
        )
        .generics(vec![
            cutile::half::bf16::DTYPE.as_str().into(),
            columns.to_string(),
        ]),
    )?;
    Ok(true)
}

fn narrow_dense_layout(architecture: &str, shape: (usize, usize, usize)) -> bool {
    architecture == "sm_120" && shape == NARROW_DENSE_SHAPE
}

fn records_narrow_dense(
    scope: &Scope,
    profile: &DeviceProfile,
    input: &TensorView<'_, f32>,
    weight: &Tensor<cutile::half::bf16>,
    output: &mut Tensor<f32>,
    shape: (usize, usize, usize),
) -> Result<bool, DeviceError> {
    #[cfg(test)]
    if DENSE_REFERENCE.get() {
        return Ok(false);
    }
    if !narrow_dense_layout(&profile.architecture, shape) {
        return Ok(false);
    }
    scope.record(
        gemm::dense_n16(output.partition(NARROW_DENSE_TILE), input, weight)
            .generics(vec![shape.2.to_string()]),
    )?;
    #[cfg(test)]
    NARROW_DENSE_CAPTURES.set(NARROW_DENSE_CAPTURES.get() + 1);
    Ok(true)
}

#[cfg(test)]
thread_local! {
    static DENSE_REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static NARROW_DENSE_CAPTURES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub fn narrow_dense_captures() -> usize {
    NARROW_DENSE_CAPTURES.get()
}

/// Captures the original dense kernel for a numerical comparison on this test thread.
#[cfg(test)]
pub fn with_dense_reference<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            DENSE_REFERENCE.set(self.0);
        }
    }
    let restore = Restore(DENSE_REFERENCE.replace(true));
    let value = run();
    drop(restore);
    value
}

#[cfg(test)]
#[path = "../../tests/unit/resident_prefill_projection.rs"]
mod tests;
