//! One KV read serves the three verification candidates of each pool slot.
use super::{
    ActivationArena,
    arena::row_tensor,
    attention_decode::VERIFY_LANES,
    batch::{BatchBuilder, States},
    capture::{Capture, CaptureMode, error},
    fp8_cache::Fp8Caches,
};
use crate::constants::ATTENTION_SPLIT_THRESHOLD;
use cutile::prelude::*;
use infer_ir::{TensorNode, TensorOp};
use std::result::Result as StdResult;
use std::sync::Arc;

impl BatchBuilder<'_> {
    /// Append all candidates before attending; per-lane metadata still enforces causality.
    pub(super) fn record_slot_verify_attention(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        arena: &mut ActivationArena,
        (states, caches): (&mut [States], &mut [Fp8Caches]),
        (verify, metadata): (usize, &[Tensor<i32>]),
    ) -> StdResult<bool, DeviceError> {
        let TensorOp::Attention {
            query_heads,
            kv_heads,
            head_dim,
            window,
        } = node.op
        else {
            return Ok(false);
        };
        let Some(&state) = node.states.first() else {
            return Ok(false);
        };
        if verify != VERIFY_LANES
            || window.is_some_and(|w| w < ATTENTION_SPLIT_THRESHOLD)
            || !self.attention.supports_verify(query_heads, head_dim)
            || self.weights.constants.contains_key(&node.inputs[0])
            || !self.weights.kv_scales.contains_key(&state)
            || caches.iter().any(|cache| !cache.contains_key(&state))
        {
            return Ok(false);
        }
        self.append_slot_candidates(scope, node, arena, (states, caches), metadata)?;
        let slot = arena.slot(node.outputs[0]).map_err(error)?;
        let output = arena.buffers[slot]
            .take()
            .ok_or_else(|| error("slot attention output"))?;
        let (mut rows, output) =
            attention_output_rows(output, metadata.len(), query_heads, head_dim)?;
        for (index, [first, second, third]) in rows
            .as_chunks_mut::<VERIFY_LANES>()
            .0
            .iter_mut()
            .enumerate()
        {
            let base = index * VERIFY_LANES;
            let queries = [
                arena.row(node.inputs[0], base).map_err(error)?,
                arena.row(node.inputs[0], base + 1).map_err(error)?,
                arena.row(node.inputs[0], base + 2).map_err(error)?,
            ];
            let (keys, values) = &caches[index][&state];
            self.attention.record_verify(
                scope,
                [first, second, third],
                [
                    &queries[0].view(&[query_heads, head_dim])?,
                    &queries[1].view(&[query_heads, head_dim])?,
                    &queries[2].view(&[query_heads, head_dim])?,
                ],
                [&metadata[base], &metadata[base + 1], &metadata[base + 2]],
                (keys, values, &self.tables[index]),
                (query_heads, kv_heads, head_dim),
                i32::try_from(window.unwrap_or(0)).map_err(error)?,
                self.weights.kv_scales[&state],
            )?;
        }
        arena.buffers[slot] = Some(output);
        Ok(true)
    }

    fn append_slot_candidates(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        arena: &mut ActivationArena,
        (states, caches): (&mut [States], &mut [Fp8Caches]),
        metadata: &[Tensor<i32>],
    ) -> StdResult<(), DeviceError> {
        for (lane, metadata) in metadata.iter().enumerate() {
            let slot = lane / VERIFY_LANES;
            Capture {
                scope,
                arena,
                weights: self.weights,
                nvfp4: self.nvfp4,
                attention: self.attention,
                states: &mut states[slot],
                fp8_states: &mut caches[slot],
                metadata,
                table: &self.tables[slot],
                external: &self.lane_external[lane],
                capacity: self.capacity,
                fusion: &mut self.lane_fusion[lane],
                mode: CaptureMode::Row(lane),
            }
            .record_attention_append(node)?;
        }
        Ok(())
    }
}

/// Preserve the arena owner's exclusive storage while capture binds borrowed row outputs.
/// Owned shared aliases would prevent a later node from reusing this arena buffer mutably.
fn attention_output_rows(
    mut output: Tensor<f32>,
    lanes: usize,
    heads: usize,
    dim: usize,
) -> StdResult<(Vec<Tensor<f32>>, Tensor<f32>), DeviceError> {
    let mut rows = Vec::with_capacity(lanes);
    for lane in 0..lanes {
        let (row, owner) = row_tensor(output, lanes, lane).map_err(error)?;
        output = Arc::try_unwrap(owner)
            .map_err(|_| error("slot attention arena owner still aliased"))?;
        rows.push(row.reshape(&[heads, dim])?);
    }
    Ok((rows, output))
}
