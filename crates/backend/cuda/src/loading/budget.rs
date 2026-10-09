//! Admission accounting for allocated tensor storage, plus driver graph headroom.
use super::LoadedModel;
use crate::resident::ProgramWeights;
use infer_core::{Error, Result};
use infer_ir::{DataflowGraph, OutputReadout, StateKind, TensorOp, TensorStorage};

/// Bytes per F32 element in u64 budget arithmetic.
const F32: u64 = crate::constants::F32_BYTES as u64;
/// Full-history readback stages a device copy and returns a second host copy.
const READBACK_COPIES: u64 = 2;

impl LoadedModel {
    /// Upper bound for known tensor storage plus 256 MiB of graph/allocator headroom.
    /// This is admission accounting, not a measurement of CUDA driver allocations.
    /// # Errors
    /// Rejects invalid capacity or arithmetic overflow before any device allocation.
    pub fn sequence_budget(&self, capacity: usize, readout: OutputReadout) -> Result<u64> {
        if capacity == 0 || capacity > crate::constants::MAX_CAPACITY_TOKENS {
            return Err(Error::invalid(
                "CUDA sequence capacity must be in 1..=32768",
            ));
        }
        let capacity = capacity.next_power_of_two() as u64;
        let hidden = self.model.hidden_size;
        let vocabulary = self.model.vocab_size;
        let mut bytes = self.device().profile()?.graph_headroom_bytes();
        bytes = add(
            bytes,
            program(
                &self.graph,
                &self.weights,
                capacity,
                hidden,
                vocabulary,
                readout,
            )?,
        )?;
        if let Some(draft) = &self.draft {
            // Draft priming has a prefill arena even though draft decode runs one token at a time.
            bytes = add(
                bytes,
                program(
                    &draft.graph,
                    &draft.weights,
                    capacity,
                    hidden,
                    vocabulary,
                    readout,
                )?,
            )?;
        }
        add(bytes, mul(capacity, F32)?)
    }
}

/// Per-program admission: activation arenas, KV pages, projection workspaces and readout staging.
fn program(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    capacity: u64,
    hidden: usize,
    vocabulary: usize,
    readout: OutputReadout,
) -> Result<u64> {
    let verify = weights.batch_width as u64;
    let prompt = weights.prompt_lane_total() as u64;
    let lanes = 1 + if verify > 1 { verify } else { 0 } + prompt;
    let mut bytes = 0;
    // Split-KV scratch is shared by sequential attention nodes with the same geometry.
    // Charge the supported upper bound (16 partitions), independent of device tuning.
    let mut attention_shapes = std::collections::BTreeSet::new();
    for node in &graph.nodes {
        if let TensorOp::Attention {
            query_heads,
            head_dim,
            ..
        } = node.op
            && attention_shapes.insert((query_heads, head_dim))
        {
            bytes = add(
                bytes,
                mul(
                    mul(
                        query_heads as u64,
                        crate::constants::ATTENTION_SPLIT_MAX_PARTS as u64,
                    )?,
                    mul(head_dim as u64 + 2, F32)?,
                )?,
            )?;
        }
    }
    // Activations live in a reuse arena sized by the peak live set, not by the sum of every
    // tensor in the graph; charge what the arena will actually allocate.
    bytes = add(
        bytes,
        u64::try_from(crate::resident::arena::ActivationArena::required_bytes(
            graph,
            usize::try_from(lanes).map_err(|_| Error::invalid("activation lanes"))?,
        )?)
        .map_err(|_| Error::invalid("activation budget"))?,
    )?;
    let mut recurrent_shapes = std::collections::BTreeSet::new();
    for node in &graph.nodes {
        if let TensorOp::Delta {
            value_heads,
            value_dim,
            ..
        } = node.op
            && weights.input_scales.is_empty()
            && weights.fp8_inputs.is_empty()
            && recurrent_shapes.insert((value_heads, value_dim))
        {
            bytes = add(
                bytes,
                mul(
                    mul(value_heads as u64, value_dim as u64)?,
                    mul(prompt, F32)?,
                )?,
            )?;
        }
    }
    for spec in &graph.tensors {
        let elements = spec.elements()? as u64;
        let amount = match spec.storage {
            TensorStorage::Activation | TensorStorage::Weight { .. } => 0,
            TensorStorage::State {
                kind: StateKind::AttentionKv,
                ..
            } => {
                let columns = *spec
                    .shape
                    .get(1)
                    .ok_or_else(|| Error::invalid("KV shape"))?
                    as u64;
                let size = if weights.kv_scales.contains_key(&spec.id) {
                    1
                } else {
                    F32
                };
                mul(mul(capacity, columns)?, 2 * size)?
            }
            TensorStorage::State { .. } => mul(elements, mul(1 + verify, F32)?)?,
        };
        bytes = add(bytes, amount)?;
    }
    bytes = add(bytes, projection_workspace(graph, weights)?)?;
    // External hidden buffers exist for the scalar graph and every prompt lane.
    // Fused draft programs additionally retain embedding and two normalized rows per lane.
    let external_lanes = 1 + weights.prefill_width.max(weights.batch_width).max(1) as u64;
    let fusion_rows = if weights.fusion.is_some() {
        crate::constants::MTP_FUSION_ROWS
    } else {
        1
    };
    bytes = add(
        bytes,
        mul(mul(hidden as u64, external_lanes * fusion_rows)?, F32)?,
    )?;
    // Pinned readback staging is retained with each graph/state. Count all
    // lanes conservatively even though prompt mode reads only the last logits.
    let readout_elements = add(hidden as u64, vocabulary as u64)?;
    bytes = add(bytes, mul(mul(readout_elements, lanes)?, F32)?)?;
    if readout == OutputReadout::Full {
        // Full-history CPU readback and its returned copy are included in admission.
        bytes = add(
            bytes,
            mul(mul(capacity, hidden as u64)?, READBACK_COPIES * F32)?,
        )?;
    }
    Ok(bytes)
}
/// Projection buffers are shared by geometry; prefill views use the activation arena.
fn projection_workspace(graph: &DataflowGraph, weights: &ProgramWeights) -> Result<u64> {
    let sizes = graph
        .tensors
        .iter()
        .map(|t| Ok((t.id, t.elements()? as u64)))
        .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
    let mut verification = std::collections::BTreeSet::new();
    let mut partials = std::collections::BTreeSet::new();
    let mut quantized = std::collections::BTreeSet::new();
    let mut fp8_quantized = std::collections::BTreeSet::new();
    let quantized_rows = [1, weights.batch_width]
        .into_iter()
        .chain(weights.prompt_widths())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .sum::<usize>() as u64;
    let mut bytes = 0;
    for node in &graph.nodes {
        if node.op != TensorOp::Linear {
            continue;
        }
        let columns = *sizes
            .get(&node.inputs[0])
            .ok_or_else(|| Error::invariant("projection input budget"))?;
        let rows = *sizes
            .get(&node.outputs[0])
            .ok_or_else(|| Error::invariant("projection output budget"))?;
        if weights.fp8_inputs.contains(&node.inputs[1]) && fp8_quantized.insert(columns) {
            bytes = add(bytes, mul(columns + F32, quantized_rows)?)?;
        }
        if weights.input_scales.contains_key(&node.inputs[1]) && quantized.insert(columns) {
            // Two FP4 values per byte, plus one E4M3 scale per block of 16.
            bytes = add(
                bytes,
                mul(
                    columns / 2 + columns / crate::constants::NVFP4_GROUP_SIZE as u64,
                    quantized_rows,
                )?,
            )?;
        }
        if weights.batch_width > 1 && verification.insert((rows, columns)) {
            bytes = add(
                bytes,
                mul(add(rows, columns)?, weights.batch_width as u64 * F32)?,
            )?;
        }
        if weights.prefill_width == crate::constants::PREFILL_LANES
            && !weights.input_scales.contains_key(&node.inputs[1])
            && matches!(
                weights.projections.get(&node.inputs[1]),
                Some(crate::mlp::ProjectionWeight::Fp4(..))
            )
            && partials.insert(rows)
        {
            bytes = add(
                bytes,
                mul(
                    rows,
                    (crate::constants::PREFILL_SPLIT_K * crate::constants::PREFILL_LANES) as u64
                        * F32,
                )?,
            )?;
        }
    }
    Ok(bytes)
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| Error::invalid("CUDA state budget overflow"))
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| Error::invalid("CUDA state budget overflow"))
}

#[cfg(test)]
#[path = "../../tests/unit/loading_budget.rs"]
mod tests;
