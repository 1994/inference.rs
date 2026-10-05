//! Prefix responsibilities.
use super::{CachedPrefix, MetalBackend, MetalDevice, Sequence, bytes};
use infer_core::{Error, Result};

impl MetalBackend {
    pub(super) fn encode_pending_prefix(
        &self,
        command: &metal::CommandBufferRef,
        s: &Sequence,
    ) -> Result<()> {
        if let Some(c) = &s.pending_prefix {
            for (id, b) in &c.tensors {
                MetalDevice::copy(
                    command,
                    b,
                    &s.tensors[id],
                    0,
                    usize::try_from(b.length() / 4)
                        .map_err(|_| Error::invalid("Metal buffer exceeds address space"))?,
                )?;
            }
            let count = self.hidden_elements(c.tokens.len(), s.readout)?;
            let offset = if s.readout != infer_ir::OutputReadout::Full
                && c.readout == infer_ir::OutputReadout::Full
            {
                c.tokens.len().saturating_sub(1) * self.model.hidden_size
            } else {
                0
            };
            MetalDevice::copy_range(command, &c.hidden, offset, &s.hidden, 0, count)?;
            MetalDevice::copy(command, &c.logits, &s.logits, 0, self.model.vocab_size)?;
        }
        Ok(())
    }
    pub(super) fn cache_boundary(
        &self,
        command: &metal::CommandBufferRef,
        s: &Sequence,
        tokens: &[u32],
        remaining_bytes: u64,
    ) -> Result<Option<CachedPrefix>> {
        if self.config.prefix_cache_bytes == 0
            || !tokens.len().is_multiple_of(self.config.page_tokens)
            || self.kv.contains_prefix_where(tokens, |cache| {
                s.readout != infer_ir::OutputReadout::Full
                    || cache.readout == infer_ir::OutputReadout::Full
            })
        {
            return Ok(None);
        }
        let hidden_elements = self.hidden_elements(tokens.len(), s.readout)?;
        let mut size = bytes(hidden_elements + self.model.vocab_size)?;
        for b in s.tensors.values() {
            size = size
                .checked_add(b.length())
                .ok_or_else(|| Error::invalid("prefix snapshot overflow"))?;
        }
        size = size
            .checked_add(self.kv_block_bytes * (tokens.len() / self.config.page_tokens) as u64)
            .ok_or_else(|| Error::invalid("prefix snapshot overflow"))?;
        if size > remaining_bytes {
            return Ok(None);
        }
        let tensors = s
            .tensors
            .iter()
            .map(|(id, b)| {
                let snapshot = self.gpu.zeros(
                    usize::try_from(b.length() / 4)
                        .map_err(|_| Error::invalid("Metal buffer exceeds address space"))?,
                )?;
                MetalDevice::copy(
                    command,
                    b,
                    &snapshot,
                    0,
                    usize::try_from(b.length() / 4)
                        .map_err(|_| Error::invalid("Metal buffer exceeds address space"))?,
                )?;
                Ok((*id, snapshot))
            })
            .collect::<Result<_>>()?;
        let hidden = self.gpu.zeros(hidden_elements)?;
        let logits = self.gpu.zeros(self.model.vocab_size)?;
        MetalDevice::copy(command, &s.hidden, &hidden, 0, hidden_elements)?;
        MetalDevice::copy(command, &s.logits, &logits, 0, self.model.vocab_size)?;
        Ok(Some(CachedPrefix {
            readout: s.readout,
            tokens: tokens.to_vec(),
            blocks: s.blocks[..tokens.len() / self.config.page_tokens].to_vec(),
            tensors,
            hidden,
            logits,
            bytes: size,
        }))
    }
}
