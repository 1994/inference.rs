//! Admission accounting for allocated tensor storage, plus driver graph headroom.
use super::LoadedModel;
use crate::{
    device::DeviceProfile,
    resident::{ProgramGeometry, ProgramWeights},
};
use infer_core::{Error, Result};
use infer_ir::{DataflowGraph, OutputReadout, StateKind, TensorOp, TensorStorage};

/// Bytes per F32 element in u64 budget arithmetic.
const F32: u64 = crate::constants::F32_BYTES as u64;
/// Full-history readback stages a device copy and returns a second host copy.
const READBACK_COPIES: u64 = 2;
/// Private/shared verification crossed with full/other readout.
const PRICE_GEOMETRIES: usize = 4;

/// Exact storage price for one immutable program geometry.
#[derive(Clone, Copy, Default)]
struct SequencePrice {
    fixed: u64,
    per_token: u64,
}

impl SequencePrice {
    fn at(self, capacity: u64) -> Result<u64> {
        add(self.fixed, mul(self.per_token, capacity)?)
    }

    fn combine(self, other: Self) -> Result<Self> {
        Ok(Self {
            fixed: add(self.fixed, other.fixed)?,
            per_token: add(self.per_token, other.per_token)?,
        })
    }
}

/// Four bounded, successful prices: private/shared verification × full/other readout.
pub(super) struct SequencePrices([std::sync::OnceLock<SequencePrice>; PRICE_GEOMETRIES]);

impl Default for SequencePrices {
    fn default() -> Self {
        Self(std::array::from_fn(|_| std::sync::OnceLock::new()))
    }
}

impl LoadedModel {
    pub(super) fn validate_capacity(&self, capacity: usize) -> Result<usize> {
        crate::resident::capacity::for_model(capacity, self.model.max_sequence)
    }

    /// Upper bound for known tensor storage plus 256 MiB of graph/allocator headroom.
    /// This is admission accounting, not a measurement of CUDA driver allocations.
    ///
    /// `private_verification` prices the rollback snapshots and batch workspaces of a
    /// sequence-owned verification graph. A sequence that will speculate through the shared
    /// slot pool omits it unless prefill needs the private batch as its fallback graph.
    /// # Errors
    /// Rejects invalid capacity or arithmetic overflow before any device allocation.
    pub fn sequence_budget(
        &self,
        capacity: usize,
        readout: OutputReadout,
        private_verification: bool,
    ) -> Result<u64> {
        let capacity = self.validate_capacity(capacity)? as u64;
        // Graph geometry is immutable after load; only KV and full-history storage grow
        // with capacity. Keep one exact affine price per readout/verification geometry.
        let index =
            usize::from(private_verification) * 2 + usize::from(readout == OutputReadout::Full);
        let cached = &self.sequence_prices.0[index];
        let price = if let Some(price) = cached.get() {
            *price
        } else {
            let price = self.sequence_price(readout, private_verification)?;
            // Concurrent first quotes can calculate the same immutable price.
            let _ = cached.set(price);
            price
        };
        price.at(capacity)
    }

    fn sequence_price(
        &self,
        readout: OutputReadout,
        private_verification: bool,
    ) -> Result<SequencePrice> {
        let hidden = self.model.hidden_size;
        let vocabulary = self.model.vocab_size;
        let profile = self.device().profile()?;
        let mut price = SequencePrice {
            fixed: profile.graph_headroom_bytes(),
            per_token: F32,
        };
        price = price.combine(program_price(
            &self.graph,
            &self.weights,
            hidden,
            vocabulary,
            readout,
            ProgramGeometry::new(&self.weights, private_verification),
            profile,
        )?)?;
        if let Some(draft) = &self.draft {
            // Draft priming retains a prefill arena and always captures private verification.
            price = price.combine(program_price(
                &draft.graph,
                &draft.weights,
                hidden,
                vocabulary,
                readout,
                ProgramGeometry::new(&draft.weights, true),
                profile,
            )?)?;
        }
        Ok(price)
    }

    /// Device bytes of the prompt scratch the target and draft programs share across all of
    /// their sequences. Charged once against the state budget, never per sequence.
    /// # Errors
    /// Rejects arithmetic overflow.
    pub fn shared_scratch_bytes(&self) -> Result<u64> {
        let profile = self.device().profile()?;
        let mut bytes = prompt_scratch_bytes(
            &self.graph,
            &self.weights,
            ProgramGeometry::new(&self.weights, true),
            profile,
        )?;
        if let Some(draft) = &self.draft {
            bytes = add(
                bytes,
                prompt_scratch_bytes(
                    &draft.graph,
                    &draft.weights,
                    ProgramGeometry::new(&draft.weights, true),
                    profile,
                )?,
            )?;
        }
        Ok(bytes)
    }

    /// KV bytes one block of `KV_BLOCK_TOKENS` positions occupies across every attention
    /// state of the target graph, fp8-packed where the checkpoint carries scales. This is
    /// the physical page price the slot pool's block accounting reports to the engine; the
    /// recurrent state does not grow with tokens and is priced by admission, not per block.
    /// # Errors
    /// Rejects a malformed KV shape or arithmetic overflow.
    pub fn kv_block_bytes(&self) -> Result<u64> {
        kv_block_bytes(&self.graph, &self.weights)
    }
}

/// One KV block's bytes over the graph's attention states: `columns` per position, keys and
/// values, packed to one byte where the checkpoint carries fp8 scales.
fn kv_block_bytes(graph: &DataflowGraph, weights: &ProgramWeights) -> Result<u64> {
    let block = u64::try_from(crate::constants::KV_BLOCK_TOKENS)
        .map_err(|_| Error::invalid("KV block size"))?;
    let mut bytes = 0;
    for spec in &graph.tensors {
        if !matches!(
            spec.storage,
            TensorStorage::State {
                kind: StateKind::AttentionKv,
                ..
            }
        ) {
            continue;
        }
        let columns = *spec
            .shape
            .get(1)
            .ok_or_else(|| Error::invalid("KV shape"))? as u64;
        let size = if weights.kv_scales.contains_key(&spec.id) {
            1
        } else {
            F32
        };
        bytes = add(bytes, mul(mul(block, columns)?, 2 * size)?)?;
    }
    Ok(bytes)
}

/// Per-program admission: activation arenas, KV pages, projection workspaces and readout staging.
fn program_price(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    hidden: usize,
    vocabulary: usize,
    readout: OutputReadout,
    geometry: ProgramGeometry,
    profile: &DeviceProfile,
) -> Result<SequencePrice> {
    let verify = geometry.verification as u64;
    let prompt = geometry.prompt_lanes() as u64;
    let lanes = geometry.arena_lanes();
    let mut bytes = 0;
    let mut per_token = 0;
    bytes = add(bytes, attention_scratch(graph)?)?;
    let max_prompt = geometry.prompts.into_iter().max().unwrap_or(0);
    if crate::resident::attention_mma::eligible(&profile.architecture, graph, weights, max_prompt) {
        let fixed = crate::resident::attention_mma::catalog::workspace_bytes(0, max_prompt)?;
        bytes = add(
            bytes,
            u64::try_from(fixed).map_err(|_| Error::invalid("MMA scratch budget"))?,
        )?;
        // One BF16 K/V pair shared across every attention layer, not one pair per layer.
        let token = crate::resident::attention_mma::catalog::KV_HEADS
            * crate::resident::attention_mma::catalog::DIM
            * 2
            * size_of::<cutile::half::bf16>();
        per_token = add(
            per_token,
            u64::try_from(token).map_err(|_| Error::invalid("MMA token budget"))?,
        )?;
    }
    // Activations live in a reuse arena sized by the peak live set, not by the sum of every
    // tensor in the graph; charge what the arena will actually allocate. Prompt arenas and the
    // rest of the prompt scratch are shared by every sequence ([`prompt_scratch_bytes`]).
    let other_arenas = crate::resident::arena::ActivationArena::required_bytes(
        graph,
        lanes - geometry.prompt_lanes(),
    )?;
    bytes = add(
        bytes,
        u64::try_from(other_arenas).map_err(|_| Error::invalid("activation budget"))?,
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
                per_token = add(per_token, mul(columns, 2 * size)?)?;
                0
            }
            TensorStorage::State { .. } => mul(elements, mul(1 + verify, F32)?)?,
        };
        bytes = add(bytes, amount)?;
    }
    bytes = add(bytes, projection_workspace(graph, weights, geometry)?)?;
    bytes = add(bytes, external_bytes(weights, hidden, profile)?)?;
    // Pinned readback is allocated on demand and capped per batch (device/readback.rs), so
    // charge what a step actually stages: one decode lane, the verification batch when it is
    // private, and the last prompt lane (prompt modes read only the last logits row).
    let readout_elements = add(hidden as u64, vocabulary as u64)?;
    let readout_lanes = 1 + if verify > 1 { verify } else { 0 } + u64::from(prompt > 0);
    bytes = add(bytes, mul(mul(readout_elements, readout_lanes)?, F32)?)?;
    if readout == OutputReadout::Full {
        // Full-history CPU readback and its returned copy are included in admission.
        per_token = add(per_token, mul(hidden as u64, READBACK_COPIES * F32)?)?;
    }
    Ok(SequencePrice {
        fixed: bytes,
        per_token,
    })
}
/// Split-KV scratch is shared by sequential attention nodes with the same geometry.
/// Charge the supported upper bound (16 partitions), independent of device tuning.
/// External hidden rows for every lane; fused draft programs also retain embedding and two
/// normalized rows per lane, except where batched prompt fusion covers the prompt graph and only
/// decode and verification lanes keep their own.
/// Prompt-graph scratch one program allocates once and shares across its sequences
/// (`resident::prompt_scratch`): per prompt width, the arena with its one logits row
/// (`prompt_logits`), the chunked recurrent workspace and the fusion workspace.
fn prompt_scratch_bytes(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    geometry: ProgramGeometry,
    profile: &DeviceProfile,
) -> Result<u64> {
    let mut bytes = 0;
    for width in geometry.prompts.into_iter().filter(|width| *width > 0) {
        let arena = crate::resident::prompt_logits::required_bytes(graph, weights, width)?;
        bytes = add(
            bytes,
            u64::try_from(arena).map_err(|_| Error::invalid("activation budget"))?,
        )?;
        bytes = add(
            bytes,
            crate::resident::recurrent_prefill::required_bytes(graph, weights, width)?,
        )?;
        bytes = add(
            bytes,
            crate::resident::fusion_batch::required_bytes(weights, width, profile),
        )?;
    }
    Ok(bytes)
}

fn external_bytes(weights: &ProgramWeights, hidden: usize, profile: &DeviceProfile) -> Result<u64> {
    let width = weights.prefill_width.max(weights.batch_width).max(1) as u64;
    let external_lanes = 1 + width;
    let fused_lanes = if weights.fusion.is_none() {
        0
    } else if crate::resident::fusion_batch::batched(weights, weights.prefill_width, profile) {
        1 + weights.batch_width.max(1) as u64
    } else {
        external_lanes
    };
    let rows = external_lanes + fused_lanes * (crate::constants::MTP_FUSION_ROWS - 1);
    mul(mul(hidden as u64, rows)?, F32)
}

/// Partial rows per KV head of the grouped decode kernel: 64 splits of the single-lane kernel's
/// 16 query rows plus the three-lane kernel's 32 stacked rows.
const GROUPED_DECODE_ROWS: usize = 3
    * crate::tuning::attention::decode::MAX_GROUPED_PARTS
    * crate::tuning::attention::decode::GROUPED_ROWS;
fn attention_scratch(graph: &DataflowGraph) -> Result<u64> {
    let mut bytes = 0;
    let mut shapes = std::collections::BTreeSet::new();
    for node in &graph.nodes {
        if let TensorOp::Attention {
            query_heads,
            kv_heads,
            head_dim,
            ..
        } = node.op
            && shapes.insert((query_heads, head_dim))
        {
            // Either one partial row per query head and split, or the grouped decode layout of
            // 32 rows per KV head and split (resident/attention_decode/workspace.rs).
            let rows = (query_heads * crate::constants::ATTENTION_SPLIT_MAX_PARTS)
                .max(kv_heads * GROUPED_DECODE_ROWS);
            bytes = add(bytes, mul(rows as u64, mul(head_dim as u64 + 2, F32)?)?)?;
        }
    }
    Ok(bytes)
}
/// Projection buffers are shared by geometry; prefill views use the activation arena.
fn projection_workspace(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    geometry: ProgramGeometry,
) -> Result<u64> {
    let sizes = graph
        .tensors
        .iter()
        .map(|t| Ok((t.id, t.elements()? as u64)))
        .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
    let mut verification = std::collections::BTreeSet::new();
    let mut partials = std::collections::BTreeSet::new();
    let mut quantized = std::collections::BTreeSet::new();
    let mut fp8_quantized = std::collections::BTreeSet::new();
    let quantized_rows = geometry
        .widths()
        .into_iter()
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
        if geometry.verification > 1 && verification.insert((rows, columns)) {
            bytes = add(
                bytes,
                mul(add(rows, columns)?, geometry.verification as u64 * F32)?,
            )?;
        }
        if geometry.prompts.contains(&crate::constants::PREFILL_LANES)
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
