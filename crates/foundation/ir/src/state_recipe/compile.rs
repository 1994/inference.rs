use super::{StateExtent, StateMemory, StateRecipe, StateRegion, StateRegionKind, StateReset};
use crate::{DataflowGraph, ModelIr, StateKind, TensorStorage};
use infer_core::{Error, Result};

/// Region slots reserved beyond per-tensor regions when pre-allocating the region list;
/// a capacity hint only.
const AUXILIARY_REGION_RESERVE: usize = 9;
/// Bytes per token id or page id in the auxiliary token and page-table state regions.
const ID_ELEMENT_BYTES: usize = size_of::<u32>();

impl StateRecipe {
    /// Compile the backend's actual storage width, which can differ from graph compute precision.
    /// # Errors
    /// Rejects malformed state shapes, zero capacities and overflowing physical dimensions.
    pub fn compile(
        model: &ModelIr,
        graph: &DataflowGraph,
        block_size: usize,
        element_bytes: usize,
        probes: bool,
        lease_bytes: usize,
    ) -> Result<Self> {
        if block_size == 0 || element_bytes == 0 || model.max_sequence == 0 {
            return Err(Error::invalid("invalid physical state recipe dimensions"));
        }
        let mut regions = Vec::with_capacity(graph.tensors.len() + AUXILIARY_REGION_RESERVE);
        Self::compile_tensors(graph, block_size, element_bytes, &mut regions)?;
        Self::compile_auxiliary(model, element_bytes, lease_bytes, &mut regions);
        if probes {
            let width = model
                .hidden_size
                .checked_mul(model.mixers.len())
                .ok_or_else(|| Error::invalid("probe state shape overflow"))?;
            regions.push(StateRegion {
                kind: StateRegionKind::Probes,
                memory: StateMemory::DevicePrivate,
                extent: StateExtent::Tokens(width),
                element_bytes,
                reset: StateReset::Zero,
            });
        }
        let recipe = Self {
            block_size,
            max_tokens: model.max_sequence,
            regions,
        };
        recipe.layout(model.max_sequence, crate::OutputReadout::Full)?;
        Ok(recipe)
    }
    fn compile_tensors(
        graph: &DataflowGraph,
        block_size: usize,
        element_bytes: usize,
        regions: &mut Vec<StateRegion>,
    ) -> Result<()> {
        for spec in &graph.tensors {
            let TensorStorage::State { kind, .. } = spec.storage else {
                continue;
            };
            let (memory, elements, reset) = match kind {
                StateKind::AttentionKv => {
                    let width = spec
                        .shape
                        .get(1)
                        .copied()
                        .filter(|_| spec.shape.len() == 2)
                        .ok_or_else(|| Error::invalid("KV state must have two dimensions"))?;
                    (
                        StateMemory::KvBlock,
                        block_size.checked_mul(width).and_then(|n| n.checked_mul(2)),
                        StateReset::ReleaseLease,
                    )
                }
                StateKind::Conv => {
                    let channels = spec.shape.first().copied();
                    let history = spec.shape.get(1).copied().and_then(|n| n.checked_sub(1));
                    (
                        StateMemory::DevicePrivate,
                        channels
                            .zip(history)
                            .filter(|_| spec.shape.len() == 2)
                            .and_then(|(c, h)| c.checked_mul(h)),
                        StateReset::Zero,
                    )
                }
                StateKind::LinearAttention
                | StateKind::Recurrent
                | StateKind::Ssm
                | StateKind::Speculation
                | StateKind::Multimodal => (
                    StateMemory::DevicePrivate,
                    Some(spec.elements()?),
                    StateReset::Zero,
                ),
            };
            let elements =
                elements.ok_or_else(|| Error::invalid("physical state shape overflow"))?;
            regions.push(StateRegion {
                kind: StateRegionKind::Tensor {
                    tensor: spec.id,
                    kind,
                },
                memory,
                extent: StateExtent::Fixed(elements.max(1)),
                element_bytes,
                reset,
            });
        }
        Ok(())
    }
    fn compile_auxiliary(
        model: &ModelIr,
        element_bytes: usize,
        lease_bytes: usize,
        regions: &mut Vec<StateRegion>,
    ) {
        for (kind, extent, bytes, reset) in [
            (
                StateRegionKind::Hidden,
                StateExtent::Hidden(model.hidden_size),
                element_bytes,
                StateReset::Zero,
            ),
            (
                StateRegionKind::Logits,
                StateExtent::Fixed(model.vocab_size),
                element_bytes,
                StateReset::Zero,
            ),
            (
                StateRegionKind::Tokens,
                StateExtent::Tokens(1),
                ID_ELEMENT_BYTES,
                StateReset::Zero,
            ),
            (
                StateRegionKind::PageTable,
                StateExtent::Pages,
                ID_ELEMENT_BYTES,
                StateReset::InvalidPage,
            ),
        ] {
            regions.push(StateRegion {
                kind,
                memory: StateMemory::DevicePrivate,
                extent,
                element_bytes: bytes,
                reset,
            });
        }
        Self::compile_mirrors(model, element_bytes, lease_bytes, regions);
    }
    fn compile_mirrors(
        model: &ModelIr,
        element_bytes: usize,
        lease_bytes: usize,
        regions: &mut Vec<StateRegion>,
    ) {
        for (kind, extent, bytes, reset) in [
            (
                StateRegionKind::Tokens,
                StateExtent::Tokens(1),
                ID_ELEMENT_BYTES,
                StateReset::ClearCursor,
            ),
            (
                StateRegionKind::PageTable,
                StateExtent::Pages,
                ID_ELEMENT_BYTES,
                StateReset::InvalidPage,
            ),
        ] {
            regions.push(StateRegion {
                kind,
                memory: StateMemory::HostMirror,
                extent,
                element_bytes: bytes,
                reset,
            });
        }
        for (kind, extent, bytes) in [
            (
                StateRegionKind::Hidden,
                StateExtent::FullTokens(model.hidden_size),
                element_bytes,
            ),
            (
                StateRegionKind::Logits,
                StateExtent::Fixed(model.vocab_size),
                element_bytes,
            ),
            (
                StateRegionKind::ReadbackRows,
                StateExtent::FullTokens(size_of::<Vec<f32>>() * 2),
                1,
            ),
            (
                StateRegionKind::BlockLeases,
                StateExtent::Pages,
                lease_bytes,
            ),
        ] {
            regions.push(StateRegion {
                kind,
                memory: StateMemory::HostMirror,
                extent,
                element_bytes: bytes,
                reset: StateReset::ClearCursor,
            });
        }
    }
}
