//! Checkpoint lifecycle.
use super::{Checkpoint, HostBackend, validate_checkpoint_tensors};
use infer_core::{Error, ErrorCode, Result};
use infer_state::cache::PrefixCache;

impl HostBackend {
    pub(super) fn provider_restore_execution_state(
        &mut self,
        payload: Option<&[u8]>,
    ) -> Result<()> {
        let payload =
            payload.ok_or_else(|| Error::invalid("missing physical execution checkpoint"))?;
        if payload.len() as u64 > self.config.memory_bytes.saturating_mul(16) {
            return Err(Error::new(
                ErrorCode::Capacity,
                "host checkpoint exceeds budget",
            ));
        }
        let checkpoint: Checkpoint =
            serde_json::from_slice(payload).map_err(|e| Error::invalid(e.to_string()))?;
        if checkpoint.identity != self.identity {
            return Err(Error::invalid("host execution checkpoint weight mismatch"));
        }
        let mut total = 0u64;
        for sequence in checkpoint.sequences.values() {
            let expected = self.empty_sequence(sequence.capacity)?;
            if sequence.reserved_bytes != expected.reserved_bytes
                || sequence.tokens.len() != sequence.hidden.len()
                || sequence.tokens.len() > sequence.capacity
                || sequence
                    .tokens
                    .iter()
                    .any(|t| *t as usize >= self.model.vocab_size)
                || sequence
                    .hidden
                    .iter()
                    .any(|r| r.len() != self.model.hidden_size || r.iter().any(|v| !v.is_finite()))
                || (!sequence.tokens.is_empty() && sequence.logits.len() != self.model.vocab_size)
                || sequence.logits.iter().any(|v| !v.is_finite())
                || sequence.tensors.len() != expected.tensors.len()
            {
                return Err(Error::invalid("corrupt host execution checkpoint"));
            }
            validate_checkpoint_tensors(sequence, &expected)?;
            total = total
                .checked_add(sequence.reserved_bytes)
                .ok_or_else(|| Error::invalid("checkpoint budget overflow"))?;
        }
        if self
            .weight_bytes()
            .checked_add(total)
            .and_then(|n| n.checked_add(self.graph.scratch_elements as u64 * 4))
            .and_then(|n| n.checked_add(self.config.prefix_cache_bytes))
            .and_then(|n| n.checked_add(self.config.probe_bytes))
            .is_none_or(|n| n > self.config.memory_bytes)
        {
            return Err(Error::new(
                ErrorCode::Capacity,
                "checkpoint physical state exceeds budget",
            ));
        }
        self.sequences = checkpoint.sequences;
        self.tokens_executed = checkpoint.tokens_executed;
        self.prefix_hits = checkpoint.prefix_hits;
        self.prefixes = PrefixCache::new(
            self.identity.as_bytes(),
            self.config.page_tokens,
            self.config.prefix_cache_bytes,
            4096,
        )?;
        Ok(())
    }
}
