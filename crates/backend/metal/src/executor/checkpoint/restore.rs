//! Restore responsibilities.
use super::{Checkpoint, MetalBackend, MetalDevice, RestoredBlock, SavedSequence, Sequence};
use infer_core::{Error, ErrorCode, Result};
use infer_ir::{StateKind, TensorStorage};
use infer_state::blocks::BlockLease;
use metal::MTLCommandBufferStatus;
use std::collections::BTreeMap;

impl MetalBackend {
    pub(in crate::executor) fn restore_sequence(
        &mut self,
        saved: &SavedSequence,
        capacity: usize,
        shared: &mut BTreeMap<BlockLease, RestoredBlock>,
    ) -> Result<Sequence> {
        let states = self.validate_saved_sequence(saved, capacity, shared)?;
        let readout = saved.readout;
        let mut s = self.empty(capacity, readout)?;
        for (at, old) in saved.blocks.iter().enumerate() {
            let payload = self.saved_page_payload(saved, &states, at);
            let tokens = saved.tokens
                [..((at + 1) * self.config.page_tokens).min(saved.tokens.len())]
                .to_vec();
            let lease = if let Some(existing) = shared.get(old) {
                if existing.payload != payload || existing.tokens != tokens {
                    return Err(Error::invalid("shared checkpoint page payload differs"));
                }
                self.kv.retain(&[existing.lease])?;
                existing.lease
            } else {
                let lease = self.kv.allocate(1)?[0];
                for (id, data) in &payload {
                    let n = self.config.page_tokens * self.specs[id].shape[1];
                    let plane = self.kv.capacity() * n;
                    MetalDevice::write_idle(
                        &self.kv_buffers[id],
                        lease.index as usize * n,
                        &data[..n],
                    )?;
                    MetalDevice::write_idle(
                        &self.kv_buffers[id],
                        plane + lease.index as usize * n,
                        &data[n..],
                    )?;
                }
                shared.insert(
                    *old,
                    RestoredBlock {
                        lease,
                        payload,
                        tokens,
                    },
                );
                lease
            };
            s.blocks.push(lease);
        }
        s.table_mirror.fill(u32::MAX);
        for (at, l) in s.blocks.iter().enumerate() {
            s.table_mirror[at] = l.index;
        }
        MetalDevice::write_indices_idle(&s.page_table, &s.table_mirror)?;
        for spec in states {
            let data = &saved.tensors[&spec.id];
            if matches!(
                spec.storage,
                TensorStorage::State {
                    kind: StateKind::AttentionKv,
                    ..
                }
            ) {
                continue;
            }
            s.tensors.insert(spec.id, self.gpu.upload(data)?);
        }
        let mut hidden = vec![0.0; self.hidden_elements(capacity, readout)?];
        hidden[..saved.hidden.len()].copy_from_slice(&saved.hidden);
        s.hidden = self.gpu.upload(&hidden)?;
        if !saved.logits.is_empty() {
            s.logits = self.gpu.upload(&saved.logits)?;
        }
        s.tokens.clone_from(&saved.tokens);
        Ok(s)
    }
}

impl MetalBackend {
    pub(in crate::executor) fn provider_restore_execution_state(
        &mut self,
        data: Option<&[u8]>,
    ) -> Result<()> {
        if self.busy.is_some()
            || self
                .transfers
                .iter()
                .any(|c| c.status() != MTLCommandBufferStatus::Completed)
        {
            return Err(Error::new(
                ErrorCode::Conflict,
                "Metal command still in flight",
            ));
        }
        let data = data.ok_or_else(|| Error::invalid("Metal checkpoint required"))?;
        if data.len() as u64 > self.config.memory_bytes.saturating_mul(16) {
            return Err(Error::new(
                ErrorCode::Capacity,
                "Metal checkpoint size exceeds budget",
            ));
        }
        let saved: Checkpoint =
            serde_json::from_slice(data).map_err(|e| Error::invalid(e.to_string()))?;
        if saved.identity != self.identity
            || saved.page_tokens != self.config.page_tokens
            || saved.pool_blocks >= u32::MAX as usize
            || saved
                .sequences
                .values()
                .flat_map(|s| s.blocks.iter())
                .any(|l| l.index as usize >= saved.pool_blocks)
        {
            return Err(Error::invalid("Metal checkpoint weight mismatch"));
        }
        let mut generations = BTreeMap::new();
        for lease in saved.sequences.values().flat_map(|s| &s.blocks) {
            if generations
                .insert(lease.index, (lease.owner, lease.generation))
                .is_some_and(|identity| identity != (lease.owner, lease.generation))
            {
                return Err(Error::invalid(
                    "checkpoint mixes owners or generations of one physical page",
                ));
            }
        }
        let required = saved.sequences.values().try_fold(
            self.base_bytes()?
                .checked_add(self.config.probe_bytes)
                .ok_or_else(|| Error::invalid("Metal budget overflow"))?,
            |n, s| {
                n.checked_add(self.state_bytes(s.capacity, s.readout)?)
                    .ok_or_else(|| Error::invalid("Metal budget overflow"))
            },
        )?;
        if required > self.config.memory_bytes {
            return Err(Error::new(
                ErrorCode::Capacity,
                "Metal checkpoint state exceeds budget",
            ));
        }
        // Validate and rebuild into a fresh pool; a failed load cannot corrupt
        // pages still owned by the current executor.
        let mut staged = self.fresh()?;
        let mut shared = BTreeMap::new();
        for (id, s) in &saved.sequences {
            let sequence = staged.restore_sequence(s, s.capacity, &mut shared)?;
            staged.sequences.insert(*id, sequence);
        }
        self.sequences = std::mem::take(&mut staged.sequences);
        std::mem::swap(&mut self.kv, &mut staged.kv);
        std::mem::swap(&mut self.kv_buffers, &mut staged.kv_buffers);
        self.tokens_executed = saved.tokens_executed;
        self.prefix_hits = saved.prefix_hits;
        Ok(())
    }
}
