//! State responsibilities.
use super::{MetalBackend, MetalDevice, MetalKernels, Sequence};
use infer_core::{Error, ErrorCode, Result, StateId};
use infer_ir::{
    BackendKind, ExecutionProgram, KvCacheInspection, ModelIr, PrecisionPlan, StateKind,
    TensorSpec, TensorStorage,
};
use infer_spi::{BackendProvider, KernelProvider};
use std::collections::BTreeMap;

impl MetalBackend {
    /// Fork a completed sequence. Full pages remain shared; the next writer to
    /// a partial tail copies it. Recurrent/history buffers are copied on GPU.
    ///
    /// # Errors
    /// Returns a not-found, conflict, invalid-input, or capacity error if sequence state cannot be shared within the target capacity.
    pub fn fork_sequence(
        &mut self,
        source: StateId,
        target: StateId,
        capacity: usize,
    ) -> Result<()> {
        if self.busy.is_some() || self.sequences.contains_key(&target) {
            return Err(Error::new(
                ErrorCode::Conflict,
                "fork requires completed source and unused target",
            ));
        }
        let original = self
            .sequences
            .get(&source)
            .ok_or_else(|| Error::invalid("fork source missing"))?;
        if original.tokens.len() > capacity {
            return Err(Error::invalid("fork capacity shorter than source"));
        }
        let readout = original.readout;
        let required = self.state_bytes(capacity, readout)?;
        if self.free_state_bytes()?.is_some_and(|free| required > free) {
            return Err(Error::new(
                ErrorCode::Capacity,
                "fork metadata budget exhausted",
            ));
        }
        let mut child = self.empty(capacity, readout)?;
        let command = self.gpu.queue.new_command_buffer().to_owned();
        let tensors = original
            .pending_prefix
            .as_ref()
            .map_or(&original.tensors, |c| &c.tensors);
        for (id, b) in tensors {
            MetalDevice::copy(
                &command,
                b,
                &child.tensors[id],
                0,
                usize::try_from(b.length() / crate::constants::F32_BYTES_U64)
                    .map_err(|_| Error::invalid("Metal buffer exceeds address space"))?,
            )?;
        }
        let hidden = original
            .pending_prefix
            .as_ref()
            .map_or(&original.hidden, |c| &c.hidden);
        let logits = original
            .pending_prefix
            .as_ref()
            .map_or(&original.logits, |c| &c.logits);
        let source_readout = original
            .pending_prefix
            .as_ref()
            .map_or(original.readout, |cache| cache.readout);
        let hidden_offset = if source_readout == infer_ir::OutputReadout::Full
            && readout != infer_ir::OutputReadout::Full
        {
            original.tokens.len().saturating_sub(1) * self.model.hidden_size
        } else {
            0
        };
        MetalDevice::copy_range(
            &command,
            hidden,
            hidden_offset,
            &child.hidden,
            0,
            self.hidden_elements(original.tokens.len(), readout)?,
        )?;
        if !original.tokens.is_empty() {
            MetalDevice::copy(&command, logits, &child.logits, 0, self.model.vocab_size)?;
        }
        child.tokens.clone_from(&original.tokens);
        child.blocks = self.kv.fork_pages(&original.blocks)?;
        command.commit();
        self.transfers.push(command);
        self.sequences.insert(target, child);
        self.update_table(target)?;
        Ok(())
    }
    pub(super) fn base_bytes(&self) -> Result<u64> {
        self.weights.values().try_fold(
            self.scratch_bytes
                .checked_add(crate::constants::F32_BYTES_U64)
                .ok_or_else(|| Error::invalid("Metal base budget overflow"))?
                .checked_add(self.config.prefix_cache_bytes.saturating_mul(2))
                .and_then(|n| n.checked_add(self.kv_block_bytes * self.kv.capacity() as u64))
                .ok_or_else(|| Error::invalid("Metal base budget overflow"))?,
            |n, b| {
                n.checked_add(b.length())
                    .ok_or_else(|| Error::invalid("Metal base budget overflow"))
            },
        )
    }
    pub(super) fn state_elements(spec: &TensorSpec, capacity: usize) -> Result<usize> {
        let elements = match &spec.storage {
            TensorStorage::State {
                kind: StateKind::AttentionKv,
                ..
            } => capacity
                .checked_mul(spec.shape[1])
                .and_then(|n| n.checked_mul(2)),
            TensorStorage::State {
                kind: StateKind::Conv,
                ..
            } => spec.shape[0].checked_mul(spec.shape[1] - 1),
            TensorStorage::State {
                kind: StateKind::LinearAttention,
                ..
            } => Some(spec.elements()?),
            _ => return Err(Error::unsupported("Metal state provider required")),
        };
        elements.ok_or_else(|| Error::invalid("Metal state shape overflow"))
    }
    pub(super) fn hidden_elements(
        &self,
        tokens: usize,
        readout: infer_ir::OutputReadout,
    ) -> Result<usize> {
        tokens
            .min(if readout == infer_ir::OutputReadout::Full {
                tokens
            } else {
                1
            })
            .checked_mul(self.model.hidden_size)
            .ok_or_else(|| Error::invalid("hidden shape overflow"))
    }
    pub(super) fn state_bytes(
        &self,
        capacity: usize,
        readout: infer_ir::OutputReadout,
    ) -> Result<u64> {
        Ok(self.state_recipe.layout(capacity, readout)?.private_bytes)
    }

    pub(super) fn empty(
        &self,
        capacity: usize,
        readout: infer_ir::OutputReadout,
    ) -> Result<Sequence> {
        let mut tensors = BTreeMap::new();
        let layout = self.state_recipe.layout(capacity, readout)?;
        self.state_recipe.visit(capacity, readout, |region| {
            if region.region.memory == infer_ir::StateMemory::DevicePrivate
                && let infer_ir::StateRegionKind::Tensor { tensor, .. } = region.region.kind
            {
                tensors.insert(tensor, self.gpu.zeros(region.elements)?);
            }
            Ok(())
        })?;
        let table_mirror = vec![u32::MAX; layout.max_pages];
        let rows = if readout == infer_ir::OutputReadout::Full {
            capacity
        } else {
            0
        };
        Ok(Sequence {
            readback: Some(infer_ir::ModelOutput {
                logits: Vec::with_capacity(self.model.vocab_size),
                hidden: (0..rows)
                    .map(|_| vec![0.0; self.model.hidden_size])
                    .collect(),

                tokens: Vec::new(),
            }),
            hidden_spares: Vec::with_capacity(rows),
            readout,
            capacity,
            reserved_bytes: layout.private_bytes,
            tokens: Vec::with_capacity(capacity),
            tensors,
            hidden: self.gpu.zeros(self.hidden_elements(capacity, readout)?)?,
            logits: self.gpu.zeros(self.model.vocab_size)?,
            probes: if self.config.probe_bytes > 0 {
                Some(
                    self.gpu
                        .zeros(capacity * self.model.hidden_size * self.model.mixers.len())?,
                )
            } else {
                None
            },
            blocks: Vec::with_capacity(capacity.div_ceil(self.config.block_size)),
            page_table: self.gpu.upload_indices(&table_mirror)?,
            table_mirror,
            token_buffer: self.gpu.zeros(capacity)?,
            pending_prefix: None,
        })
    }
    pub(super) fn kv_inspection(&self) -> KvCacheInspection {
        self.kv.inspect_with_pins(
            self.sequences.values().map(|s| s.blocks.as_slice()),
            &self.inflight_pins,
        )
    }
    pub(super) fn reclaim(&mut self, count: usize) -> Result<()> {
        self.kv.reclaim(count)
    }
    pub(super) fn update_table(&mut self, id: StateId) -> Result<()> {
        let s = self
            .sequences
            .get_mut(&id)
            .ok_or_else(|| Error::invalid("unknown page table"))?;
        s.table_mirror.fill(u32::MAX);
        for (at, l) in s.blocks.iter().enumerate() {
            s.table_mirror[at] = l.index;
        }
        MetalDevice::write_indices_idle(&s.page_table, &s.table_mirror)?;
        Ok(())
    }
    pub(super) fn grow_pages(
        &mut self,
        id: StateId,
        end: usize,
        command: &metal::CommandBufferRef,
    ) -> Result<()> {
        let s = self
            .sequences
            .get_mut(&id)
            .ok_or_else(|| Error::invariant("sequence disappeared"))?;
        if let Some(copy) = self.kv.prepare_append(&mut s.blocks, s.tokens.len(), end)? {
            for (tensor, buffer) in &self.kv_buffers {
                let width = self.specs[tensor].shape[1];
                let n = self.config.block_size * width;
                for base in [0, self.kv.capacity() * n] {
                    MetalDevice::copy_range(
                        command,
                        buffer,
                        base + copy.source.index as usize * n,
                        buffer,
                        base + copy.destination.index as usize * n,
                        n,
                    )?;
                }
            }
        }
        self.update_table(id)
    }
}

impl MetalBackend {
    pub(super) fn provider_reuse_prefix(
        &mut self,
        id: StateId,
        tokens: &[u32],
        maximum: usize,
    ) -> Result<usize> {
        if !self.sequences.get(&id).is_some_and(|s| s.tokens.is_empty()) {
            return Ok(0);
        }
        let readout = self.sequences[&id].readout;
        let Some(cached) = self.kv.attach_prefix_where(tokens, maximum, |cache| {
            readout != infer_ir::OutputReadout::Full
                || cache.readout == infer_ir::OutputReadout::Full
        })?
        else {
            return Ok(0);
        };
        let s = self
            .sequences
            .get_mut(&id)
            .ok_or_else(|| Error::invariant("sequence disappeared"))?;
        s.tokens.clone_from(&cached.tokens);
        s.blocks.clone_from(&cached.blocks);
        s.pending_prefix = Some(cached);
        self.update_table(id)?;
        self.prefix_hits = self.prefix_hits.saturating_add(1);
        Ok(self.sequences[&id].tokens.len())
    }
    pub(super) fn provider_validate_program(
        &self,
        model: &ModelIr,
        p: &ExecutionProgram,
    ) -> Result<()> {
        let kernels = MetalKernels.kernels();
        if model != &self.model
            || p.backend != BackendKind::Metal
            || p.model != model.id
            || p.precision != PrecisionPlan::f32()
            || p.dataflow != self.graph
            || p.operations.len() != self.graph.nodes.len()
            || p.operations.iter().zip(&self.graph.nodes).any(|(o, n)| {
                o.op.id != n.id
                    || o.op.operation
                        != n.op
                            .operation(self.graph.logits.is_some_and(|id| n.outputs.contains(&id)))
                    || !kernels
                        .iter()
                        .any(|k| k.id == o.kernel && k.operation == o.op.operation)
            })
        {
            return Err(Error::invalid("Metal program/model/kernel target mismatch"));
        }
        Ok(())
    }
    pub(super) fn provider_reserve_state_for(
        &mut self,
        id: StateId,
        capacity: usize,
        readout: infer_ir::OutputReadout,
    ) -> Result<()> {
        if self.sequences.contains_key(&id) {
            return Err(Error::new(ErrorCode::Conflict, "Metal state exists"));
        }
        let required = self.state_bytes(capacity, readout)?;
        if self
            .base_bytes()?
            .checked_add(self.config.probe_bytes)
            .and_then(|n| n.checked_add(self.inspect().reserved_bytes))
            .and_then(|n| n.checked_add(required))
            .is_none_or(|n| n > self.config.memory_bytes)
        {
            return Err(Error::new(
                ErrorCode::Capacity,
                "Metal physical state budget exhausted",
            ));
        }
        self.sequences.insert(id, self.empty(capacity, readout)?);
        Ok(())
    }
    pub(super) fn provider_reset_state(&mut self, id: StateId) -> Result<()> {
        if self.busy.is_some() {
            return Err(Error::new(
                ErrorCode::Conflict,
                "Metal command still in flight",
            ));
        }
        if self
            .transfers
            .iter()
            .any(|command| command.status() != metal::MTLCommandBufferStatus::Completed)
        {
            return Err(Error::new(
                ErrorCode::Conflict,
                "Metal state transfer still in flight",
            ));
        }
        let sequence = self
            .sequences
            .get_mut(&id)
            .ok_or_else(|| Error::invalid("unknown Metal state"))?;
        self.kv.release(&sequence.blocks)?;
        sequence.blocks.clear();
        sequence.tokens.clear();
        sequence.pending_prefix = None;
        sequence.table_mirror.fill(u32::MAX);
        MetalDevice::write_indices_idle(&sequence.page_table, &sequence.table_mirror)?;
        for buffer in sequence
            .tensors
            .values()
            .chain([&sequence.hidden, &sequence.logits, &sequence.token_buffer])
            .chain(sequence.probes.iter())
        {
            MetalDevice::clear_idle(buffer)?;
        }
        Ok(())
    }
}
