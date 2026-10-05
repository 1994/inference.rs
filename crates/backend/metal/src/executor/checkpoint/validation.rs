//! Validation responsibilities.
use super::{MetalBackend, RestoredBlock, SavedSequence};
use infer_core::{Error, ErrorCode, Result, StateId, TensorId};
use infer_ir::{StateKind, TensorSpec, TensorStorage};
use infer_state::blocks::BlockLease;
use std::{collections::BTreeMap, collections::BTreeSet};

impl MetalBackend {
    pub(in crate::executor) fn validate_saved_sequence(
        &self,
        saved: &SavedSequence,
        capacity: usize,
        shared: &BTreeMap<BlockLease, RestoredBlock>,
    ) -> Result<Vec<TensorSpec>> {
        if saved.capacity == 0
            || saved.capacity > self.model.max_sequence
            || saved.tokens.len() > saved.capacity
            || saved.tokens.len() > capacity
            || saved
                .tokens
                .iter()
                .any(|t| *t as usize >= self.model.vocab_size)
            || saved.hidden.len() != self.hidden_elements(saved.tokens.len(), saved.readout)?
            || saved.logits.len()
                != if saved.tokens.is_empty() {
                    0
                } else {
                    self.model.vocab_size
                }
            || saved
                .hidden
                .iter()
                .chain(&saved.logits)
                .any(|v| !v.is_finite())
        {
            return Err(Error::invalid(
                "corrupt paged Metal checkpoint cursor/output",
            ));
        }
        let count = saved.tokens.len().div_ceil(self.config.page_tokens);
        if saved.blocks.len() != count
            || saved.blocks.iter().any(|l| l.generation == 0)
            || saved
                .blocks
                .iter()
                .map(|l| l.index)
                .collect::<BTreeSet<_>>()
                .len()
                != count
        {
            return Err(Error::invalid(
                "checkpoint page table lease/cursor mismatch",
            ));
        }
        let needed = saved
            .blocks
            .iter()
            .filter(|l| !shared.contains_key(l))
            .count();
        if needed > self.kv.free_blocks() {
            return Err(Error::new(
                ErrorCode::Capacity,
                "checkpoint KV pool exhausted",
            ));
        }
        let states: Vec<_> = self
            .specs
            .values()
            .filter(|s| matches!(s.storage, TensorStorage::State { .. }))
            .cloned()
            .collect();
        if saved.tensors.len() != states.len() {
            return Err(Error::invalid("checkpoint state bindings"));
        }
        for spec in &states {
            let data = saved
                .tensors
                .get(&spec.id)
                .ok_or_else(|| Error::invalid("checkpoint tensor missing"))?;
            if data.len() != Self::state_elements(spec, saved.capacity)?.max(1)
                || data.iter().any(|v| !v.is_finite())
            {
                return Err(Error::invalid("checkpoint state shape/numeric"));
            }
        }
        Ok(states)
    }
    pub(in crate::executor) fn saved_page_payload(
        &self,
        saved: &SavedSequence,
        states: &[TensorSpec],
        at: usize,
    ) -> BTreeMap<TensorId, Vec<f32>> {
        let mut payload = BTreeMap::new();
        for spec in states {
            if !matches!(
                spec.storage,
                TensorStorage::State {
                    kind: StateKind::AttentionKv,
                    ..
                }
            ) {
                continue;
            }
            let width = spec.shape[1];
            let n = self.config.page_tokens * width;
            let mut data = vec![0.0; n * 2];
            let source = &saved.tensors[&spec.id];
            for row in 0..self.config.page_tokens {
                let absolute = at * self.config.page_tokens + row;
                if absolute >= saved.tokens.len() {
                    break;
                }
                data[row * width..(row + 1) * width]
                    .copy_from_slice(&source[absolute * width..(absolute + 1) * width]);
                data[n + row * width..n + (row + 1) * width].copy_from_slice(
                    &source[saved.capacity * width + absolute * width
                        ..saved.capacity * width + (absolute + 1) * width],
                );
            }
            payload.insert(spec.id, data);
        }
        payload
    }
}

impl MetalBackend {
    pub(in crate::executor) fn provider_validate_state_ownership(
        &self,
        states: &[(StateId, usize, usize)],
    ) -> Result<()> {
        self.kv.check_owners(
            self.sequences
                .values()
                .map(|s| s.blocks.as_slice())
                .chain(std::iter::once(self.inflight_pins.as_slice())),
        )?;
        if states.len() != self.sequences.len()
            || states.iter().any(|(id, cap, pos)| {
                !self.sequences.get(id).is_some_and(|s| {
                    s.capacity == *cap
                        && s.tokens.len() == *pos
                        && s.blocks.len() == pos.div_ceil(self.config.page_tokens)
                })
            })
        {
            return Err(Error::invalid(
                "Metal physical state ownership/cursor mismatch",
            ));
        }
        Ok(())
    }
}
