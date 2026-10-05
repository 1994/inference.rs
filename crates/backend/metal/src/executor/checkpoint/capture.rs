//! Capture responsibilities.
use super::{Checkpoint, MetalBackend, MetalDevice, SavedSequence, Sequence};
use infer_core::{Error, ErrorCode, Result};
use infer_ir::{StateKind, TensorStorage};
use metal::MTLCommandBufferStatus;
use std::collections::BTreeMap;

impl MetalBackend {
    pub(in crate::executor) fn save_hidden(
        &self,
        s: &Sequence,
        buffer: &metal::BufferRef,
    ) -> Result<Vec<f32>> {
        let count = self.hidden_elements(s.tokens.len(), s.readout)?;
        let data = MetalDevice::read_idle(
            buffer,
            usize::try_from(buffer.length() / 4)
                .map_err(|_| Error::invalid("hidden buffer exceeds address space"))?,
        )?;
        if s.pending_prefix
            .as_ref()
            .is_some_and(|c| c.readout == infer_ir::OutputReadout::Full)
            && s.readout != infer_ir::OutputReadout::Full
        {
            Ok(data[data.len().saturating_sub(count)..].to_vec())
        } else {
            Ok(data[..count].to_vec())
        }
    }
    pub(in crate::executor) fn save(&self, s: &Sequence) -> Result<SavedSequence> {
        let source_tensors = s.pending_prefix.as_ref().map_or(&s.tensors, |c| &c.tensors);
        let mut tensors = source_tensors
            .iter()
            .map(|(id, b)| {
                Ok((
                    *id,
                    MetalDevice::read_idle(
                        b,
                        usize::try_from(b.length() / 4)
                            .map_err(|_| Error::invalid("Metal buffer exceeds address space"))?,
                    )?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        for spec in self.specs.values().filter(|s| {
            matches!(
                s.storage,
                TensorStorage::State {
                    kind: StateKind::AttentionKv,
                    ..
                }
            )
        }) {
            let width = spec.shape[1];
            let mut data = vec![0.0; s.capacity * width * 2];
            if !s.tokens.is_empty() {
                let buffer = self
                    .kv_buffers
                    .get(&spec.id)
                    .ok_or_else(|| Error::invariant("KV pool missing"))?;
                let pool = MetalDevice::read_idle(
                    buffer,
                    usize::try_from(buffer.length() / 4)
                        .map_err(|_| Error::invalid("Metal buffer exceeds address space"))?,
                )?;
                let plane = self.kv.capacity() * self.config.page_tokens * width;
                for row in 0..s.tokens.len() {
                    let physical = s.blocks[row / self.config.page_tokens].index as usize
                        * self.config.page_tokens
                        + row % self.config.page_tokens;
                    data[row * width..(row + 1) * width]
                        .copy_from_slice(&pool[physical * width..(physical + 1) * width]);
                    data[s.capacity * width + row * width..s.capacity * width + (row + 1) * width]
                        .copy_from_slice(
                            &pool[plane + physical * width..plane + (physical + 1) * width],
                        );
                }
            }
            tensors.insert(spec.id, data);
        }
        let hidden = s.pending_prefix.as_ref().map_or(&s.hidden, |c| &c.hidden);
        let logits = s.pending_prefix.as_ref().map_or(&s.logits, |c| &c.logits);
        Ok(SavedSequence {
            readout: s.readout,
            capacity: s.capacity,
            tokens: s.tokens.clone(),
            tensors,
            hidden: self.save_hidden(s, hidden)?,
            logits: if s.tokens.is_empty() {
                vec![]
            } else {
                MetalDevice::read_idle(logits, self.model.vocab_size)?
            },
            blocks: s.blocks.clone(),
        })
    }
}

impl MetalBackend {
    pub(in crate::executor) fn provider_capture_execution_state(&self) -> Result<Option<Vec<u8>>> {
        if self.busy.is_some()
            || self
                .transfers
                .iter()
                .any(|c| c.status() != MTLCommandBufferStatus::Completed)
        {
            return Err(Error::new(
                ErrorCode::Conflict,
                "quiesce Metal before checkpoint",
            ));
        }
        let checkpoint = Checkpoint {
            identity: self.identity.clone(),
            sequences: self
                .sequences
                .iter()
                .map(|(id, s)| Ok((*id, self.save(s)?)))
                .collect::<Result<_>>()?,
            tokens_executed: self.tokens_executed,
            prefix_hits: self.prefix_hits,
            page_tokens: self.config.page_tokens,
            pool_blocks: self.kv.capacity(),
        };
        Ok(Some(
            serde_json::to_vec(&checkpoint).map_err(|e| Error::invalid(e.to_string()))?,
        ))
    }
}
