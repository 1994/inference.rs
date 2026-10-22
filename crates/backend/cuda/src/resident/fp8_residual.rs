//! Adjacent FP8 projection/residual pairs with no exposed intermediate value.
use super::{
    ActivationArena, ProgramWeights, capture::error, nvfp4_gemm::Workspace,
    program::ActivationQuantization,
};
use crate::constants::F32_BYTES;
use crate::mlp::ProjectionWeight;
use cutile::prelude::*;
use infer_core::TensorId;
use infer_ir::{DType, DataflowGraph, TensorNode, TensorOp, TensorStorage};
#[cfg(test)]
use std::cell::Cell;
use std::collections::BTreeMap;

const CAPTURE_OPERANDS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Pair {
    residual: TensorId,
    projection_first: bool,
}

pub(super) fn plan(graph: &DataflowGraph, weights: &ProgramWeights) -> BTreeMap<usize, Pair> {
    #[cfg(test)]
    if DISABLED.get() {
        return BTreeMap::new();
    }
    if weights.fusion.is_some() {
        return BTreeMap::new();
    }
    from_contracts(
        graph,
        |id| {
            matches!(
                weights.projections.get(&id),
                Some(ProjectionWeight::Fp8(..))
            ) && matches!(
                weights.activation_quantization(id),
                Some(ActivationQuantization::Fp8Token)
            )
        },
        |id| weights.constants.contains_key(&id),
    )
}

fn from_contracts(
    graph: &DataflowGraph,
    eligible: impl Fn(TensorId) -> bool,
    constant: impl Fn(TensorId) -> bool,
) -> BTreeMap<usize, Pair> {
    let mut readers = BTreeMap::<TensorId, Vec<usize>>::new();
    for (index, node) in graph.nodes.iter().enumerate() {
        for id in &node.inputs {
            readers.entry(*id).or_default().push(index);
        }
    }
    let spec = |id| graph.tensors.iter().find(|tensor| tensor.id == id);
    let mut pairs = BTreeMap::new();
    for (index, nodes) in graph.nodes.windows(2).enumerate() {
        let [linear, add] = nodes else {
            continue;
        };
        if linear.op != TensorOp::Linear
            || add.op != TensorOp::Add
            || linear.inputs.len() != 2
            || linear.outputs.len() != 1
            || add.inputs.len() != 2
            || add.outputs.len() != 1
            || !linear.states.is_empty()
            || !add.states.is_empty()
            || !eligible(linear.inputs[1])
            || constant(linear.inputs[0])
        {
            continue;
        }
        let raw = linear.outputs[0];
        if [graph.hidden, graph.logits].contains(&Some(raw))
            || readers.get(&raw).is_none_or(|uses| uses[..] != [index + 1])
        {
            continue;
        }
        let projection_first = add.inputs[0] == raw;
        let residual = if projection_first {
            add.inputs[1]
        } else {
            add.inputs[0]
        };
        if !add.inputs.contains(&raw)
            || residual == raw
            || constant(residual)
            || [linear.inputs[0], raw, residual].contains(&add.outputs[0])
        {
            continue;
        }
        let Some(shape) = spec(raw) else {
            continue;
        };
        if shape.shape.is_empty()
            || shape.shape.contains(&0)
            || [raw, residual, add.outputs[0]].iter().any(|&id| {
                spec(id).is_none_or(|value| {
                    value.dtype != DType::F32
                        || value.storage != TensorStorage::Activation
                        || value.shape != shape.shape
                })
            })
        {
            continue;
        }
        pairs.insert(
            index,
            Pair {
                residual,
                projection_first,
            },
        );
    }
    pairs
}

/// Guard actual allocations before removing any slot. A matching logical graph does not prove
/// nonaliasing: the arena may recycle a projection input for the following Add output.
fn disjoint(
    arena: &ActivationArena,
    ids: [TensorId; CAPTURE_OPERANDS],
) -> Result<bool, DeviceError> {
    let mut spans = Vec::new();
    for id in ids {
        let tensor = arena.get(id).map_err(error)?;
        if !tensor.is_contiguous() {
            return Ok(false);
        }
        let start = tensor.device_pointer().cu_deviceptr();
        let Some(end) = u64::try_from(tensor.size())
            .ok()
            .and_then(|n| n.checked_mul(u64::try_from(F32_BYTES).ok()?))
            .and_then(|bytes| start.checked_add(bytes))
        else {
            return Ok(false);
        };
        spans.push((start, end));
    }
    Ok(disjoint_outputs(&spans))
}

fn disjoint_outputs(spans: &[(u64, u64)]) -> bool {
    // Input and residual may overlap each other. Both output spans must be independent.
    let [input, residual, raw, output] = spans else {
        return false;
    };
    [
        (input, raw),
        (input, output),
        (residual, raw),
        (residual, output),
        (raw, output),
    ]
    .into_iter()
    .all(|(a, b)| a.1 <= b.0 || b.1 <= a.0)
}

pub(super) fn record(
    scope: &Scope,
    nodes: (&TensorNode, &TensorNode),
    pair: Pair,
    arena: &mut ActivationArena,
    weights: &ProgramWeights,
    workspace: &mut Workspace,
) -> Result<bool, DeviceError> {
    let (linear, add) = nodes;
    if !disjoint(
        arena,
        [
            linear.inputs[0],
            pair.residual,
            linear.outputs[0],
            add.outputs[0],
        ],
    )? {
        return Ok(false);
    }
    let width = arena.lanes();
    let raw_slot = arena.slot(linear.outputs[0]).map_err(error)?;
    let output_slot = arena.slot(add.outputs[0]).map_err(error)?;
    let flat = arena.get(linear.outputs[0]).map_err(error)?.size();
    if width == 0
        || !flat.is_multiple_of(width)
        || arena.get(add.outputs[0]).map_err(error)?.size() != flat
        || arena.get(pair.residual).map_err(error)?.size() != flat
    {
        return Ok(false);
    }
    let weight = weights
        .projections
        .get(&linear.inputs[1])
        .ok_or_else(|| error("FP8 residual weight"))?;
    let raw = arena.buffers[raw_slot]
        .take()
        .ok_or_else(|| error("FP8 residual raw slot"))?;
    let output = arena.buffers[output_slot]
        .take()
        .ok_or_else(|| error("FP8 residual output slot"))?;
    let mut raw = raw.reshape(&[width, flat / width])?;
    let mut output = output.reshape(&[width, flat / width])?;
    let result = (|| {
        let input = arena.get(linear.inputs[0]).map_err(error)?;
        let columns = input.size() / width;
        let pointer = input.device_pointer().cu_deviceptr();
        let input = input.view(&[width, columns]).map_err(error)?;
        let residual = arena
            .get(pair.residual)
            .map_err(error)?
            .view(&[width, flat / width])
            .map_err(error)?;
        workspace.record_fp8_add(
            scope,
            weight,
            (&input, columns, pointer),
            (&mut raw, &mut output),
            (&residual, pair.projection_first),
        )
    })();
    arena.buffers[raw_slot] = Some(raw.reshape(&[flat])?);
    arena.buffers[output_slot] = Some(output.reshape(&[flat])?);
    #[cfg(test)]
    if matches!(result, Ok(true)) {
        CAPTURES.set(CAPTURES.get() + 1);
    }
    result
}

#[cfg(test)]
#[path = "../../tests/unit/resident_fp8_residual_plan.rs"]
mod tests;

#[cfg(test)]
thread_local! {
    static DISABLED: Cell<bool> = const { Cell::new(false) };
    static CAPTURES: Cell<usize> = const { Cell::new(0) };
}
#[cfg(test)]
pub fn captures() -> usize {
    CAPTURES.get()
}

#[cfg(test)]
pub fn with_separate<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            DISABLED.set(self.0);
        }
    }
    let restore = Restore(DISABLED.replace(true));
    let value = run();
    drop(restore);
    value
}
