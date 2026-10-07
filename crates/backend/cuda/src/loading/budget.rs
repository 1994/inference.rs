//! Conservative admission accounting: no activation reuse credit, plus driver graph headroom.
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
        let verify = self.weights.batch_width as u64;
        let prompt = u64::from(self.weights.prefill_width == crate::constants::PREFILL_LANES)
            * crate::constants::PREFILL_LANES as u64;
        let lanes = 1 + if verify > 1 { verify } else { 0 } + prompt;
        let hidden = self.model.hidden_size;
        let vocabulary = self.model.vocab_size;
        let mut bytes = self.device().profile()?.graph_headroom_bytes();
        bytes = add(
            bytes,
            program(
                &self.graph,
                &self.weights,
                capacity,
                lanes,
                hidden,
                vocabulary,
                readout,
            )?,
        )?;
        if let Some(draft) = &self.draft {
            // The draft always runs one token at a time with F32 KV and no scale side tables.
            bytes = add(
                bytes,
                program(
                    &draft.graph,
                    &draft.weights,
                    capacity,
                    1,
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
    lanes: u64,
    hidden: usize,
    vocabulary: usize,
    readout: OutputReadout,
) -> Result<u64> {
    let verify = weights.batch_width as u64;
    let prompt = u64::from(weights.prefill_width == crate::constants::PREFILL_LANES)
        * crate::constants::PREFILL_LANES as u64;
    let mut bytes = 0;
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
    // Overcount shared projection workspaces deliberately; never count immutable model weights.
    for node in &graph.nodes {
        if node.op != TensorOp::Linear {
            continue;
        }
        for id in [node.inputs[0], node.outputs[0]] {
            let spec = graph
                .tensors
                .iter()
                .find(|t| t.id == id)
                .ok_or_else(|| Error::invariant("projection tensor budget"))?;
            bytes = add(
                bytes,
                mul(spec.elements()? as u64, mul(verify + prompt, F32)?)?,
            )?;
        }
    }
    bytes = add(bytes, mul(hidden as u64, F32)?)?;
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
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| Error::invalid("CUDA state budget overflow"))
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| Error::invalid("CUDA state budget overflow"))
}
