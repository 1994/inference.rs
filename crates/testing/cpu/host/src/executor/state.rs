//! State responsibilities.
use super::{
    HostBackend, HostKernels, HostObservations, OpTrace, Sequence, kernels, publish_prefix,
};
use infer_core::{Error, ErrorCode, Result, StateId};
use infer_ir::{
    BackendKind, ExecutionProgram, ExecutionTask, ModelIr, ModelOutput, PrecisionPlan, StateKind,
    StepPlan, TensorOp, TensorStorage,
};
use infer_spi::KernelProvider;
use infer_state::{physical::PagedRows, physical::PhysicalTensor};
use std::{collections::BTreeMap, time::Instant};

/// Bytes for one KV key element and its matching value element (two `f32` values).
const KV_KEY_VALUE_BYTES: u128 = 2 * crate::constants::F32_BYTES_U128;

impl HostBackend {
    pub fn drain_traces(&mut self) -> Vec<OpTrace> {
        self.traces.drain(..).collect()
    }
    pub(super) fn weight_bytes(&self) -> u64 {
        self.weights
            .values()
            .map(|w| w.data.len() as u64 * crate::constants::F32_BYTES_U64)
            .sum()
    }
    pub(super) fn required_state_bytes(&self, capacity: usize) -> Result<u64> {
        if capacity == 0 || capacity > self.model.max_sequence {
            return Err(Error::invalid("host state capacity"));
        }
        let mut bytes = (capacity as u128)
            * (self.model.hidden_size as u128 * crate::constants::F32_BYTES_U128
                + crate::constants::TOKEN_BYTES_U128)
            + self.model.vocab_size as u128 * crate::constants::F32_BYTES_U128;
        for spec in self.specs.values() {
            let TensorStorage::State { kind, .. } = &spec.storage else {
                continue;
            };
            let add = match kind {
                StateKind::AttentionKv => {
                    capacity.div_ceil(self.config.block_size) as u128
                        * self.config.block_size as u128
                        * spec.shape[1] as u128
                        * KV_KEY_VALUE_BYTES
                }
                StateKind::Conv => {
                    spec.shape[0] as u128
                        * (spec.shape[1] - 1) as u128
                        * crate::constants::F32_BYTES_U128
                }
                StateKind::LinearAttention => {
                    spec.elements()? as u128 * crate::constants::F32_BYTES_U128
                }
                _ => return Err(Error::unsupported("host state storage provider required")),
            };
            bytes = bytes
                .checked_add(add)
                .ok_or_else(|| Error::invalid("host state overflow"))?;
        }
        u64::try_from(bytes).map_err(|_| Error::invalid("host state overflow"))
    }
    pub(super) fn empty_sequence(&self, capacity: usize) -> Result<Sequence> {
        if capacity == 0 || capacity > self.model.max_sequence {
            return Err(Error::invalid("host state capacity"));
        }
        let mut tensors = BTreeMap::new();
        let reserved_bytes = self.required_state_bytes(capacity)?;
        if reserved_bytes > self.config.memory_bytes {
            return Err(Error::new(
                ErrorCode::Capacity,
                "physical state exceeds host budget",
            ));
        }
        for spec in self.specs.values() {
            let TensorStorage::State { kind, .. } = &spec.storage else {
                continue;
            };
            let tensor = match kind {
                StateKind::AttentionKv => {
                    let width = spec.shape[1];
                    PhysicalTensor::Kv {
                        keys: PagedRows::new(width, self.config.block_size, capacity)?,
                        values: PagedRows::new(width, self.config.block_size, capacity)?,
                    }
                }
                StateKind::Conv => {
                    let channels = spec.shape[0];
                    let kernel = spec.shape[1];
                    let history = vec![0.0; channels * (kernel - 1)];
                    PhysicalTensor::Conv {
                        channels,
                        kernel,
                        history,
                    }
                }
                StateKind::LinearAttention => {
                    let recurrent = vec![0.0; spec.elements()?];
                    PhysicalTensor::Delta {
                        heads: spec.shape[0],
                        key_dim: spec.shape[1],
                        value_dim: spec.shape[2],
                        recurrent,
                    }
                }
                _ => return Err(Error::unsupported("host state storage provider required")),
            };
            tensors.insert(spec.id, tensor);
        }
        Ok(Sequence {
            capacity,
            reserved_bytes,
            tokens: vec![],
            hidden: vec![],
            logits: vec![],
            tensors,
        })
    }

    pub(super) fn forward_task(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        task: &ExecutionTask,
    ) -> Result<ModelOutput> {
        let mut observations = HostObservations::new(
            &mut self.traces,
            &mut self.trace_dropped,
            &mut self.probes,
            &mut self.probe_dropped,
            (&self.layer_outputs, &self.slots, &self.config),
            self.model.hidden_size,
        );
        let sequence = self
            .sequences
            .get_mut(&task.state)
            .ok_or_else(|| Error::invalid("host task has no reserved physical state"))?;
        sequence.validate_input(task, self.model.vocab_size)?;
        let slot_count = self.slots.values().copied().max().map_or(0, |s| s + 1);
        let mut buffers = vec![Vec::<f32>::new(); slot_count];
        let start_position = sequence.tokens.len();
        for (row, token) in task
            .tokens
            .delta(start_position)?
            .iter()
            .copied()
            .enumerate()
        {
            let position = start_position + row;
            for (node, compiled) in self.graph.nodes.iter().zip(&program.operations) {
                let start = Instant::now();
                let timestamp_ns =
                    u64::try_from(self.trace_origin.elapsed().as_nanos()).unwrap_or(u64::MAX);
                for id in &node.outputs {
                    buffers[self.slots[id]] = Vec::new();
                }
                let mut inputs = Vec::<&[f32]>::new();
                for id in &node.inputs {
                    if let Some(weight) = self.weights.get(id) {
                        inputs.push(&weight.data);
                    } else {
                        inputs.push(&buffers[self.slots[id]]);
                    }
                }
                if matches!(node.op, TensorOp::Embedding) {
                    let width = self.model.hidden_size;
                    inputs[0] = &inputs[0][token as usize * width..(token as usize + 1) * width];
                }
                let state = node
                    .states
                    .first()
                    .map(|id| {
                        sequence
                            .tensors
                            .get_mut(id)
                            .ok_or_else(|| Error::invariant("bound state"))
                    })
                    .transpose()?;
                let outputs = kernels::execute(&node.op, &inputs, state, token, position)?;
                if outputs.len() != node.outputs.len() {
                    return Err(Error::invariant("kernel output binding count"));
                }
                for (id, output) in node.outputs.iter().zip(outputs) {
                    if output.len() != self.specs[id].elements()? {
                        return Err(Error::invariant("kernel output binding shape"));
                    }
                    buffers[self.slots[id]] = output;
                }
                observations.record(
                    node,
                    compiled.kernel,
                    task,
                    (step, position),
                    (start, timestamp_ns),
                    &buffers,
                );
            }
            sequence.hidden.push(
                buffers[self.slots[&self
                    .graph
                    .hidden
                    .ok_or_else(|| Error::invariant("compiled result"))?]]
                    .clone(),
            );
            sequence.logits.clone_from(
                &buffers[self.slots[&self
                    .graph
                    .logits
                    .ok_or_else(|| Error::invariant("compiled result"))?]],
            );
            sequence.tokens.push(token);
            self.tokens_executed = self.tokens_executed.saturating_add(1);
            publish_prefix(
                &mut self.prefixes,
                sequence,
                self.config.block_size,
                self.config.prefix_cache_bytes,
            )?;
        }
        Ok(sequence.readout(task.tokens.readout()))
    }
}

impl HostBackend {
    pub(super) fn provider_reuse_prefix(
        &mut self,
        id: StateId,
        tokens: &[u32],
        maximum: usize,
    ) -> usize {
        if !self.sequences.get(&id).is_some_and(|s| s.tokens.is_empty()) {
            return 0;
        }
        let Some(mut cached) = self.prefixes.lookup(tokens, maximum) else {
            return 0;
        };
        let old = &self.sequences[&id];
        cached.capacity = old.capacity;
        cached.reserved_bytes = old.reserved_bytes;
        for tensor in cached.tensors.values_mut() {
            if let PhysicalTensor::Kv { keys, values } = tensor {
                keys.capacity = old.capacity;
                values.capacity = old.capacity;
            }
        }
        let length = cached.tokens.len();
        self.sequences.insert(id, cached);
        self.prefix_hits = self.prefix_hits.saturating_add(1);
        length
    }
    pub(super) fn provider_validate_program(
        &self,
        model: &ModelIr,
        program: &ExecutionProgram,
    ) -> Result<()> {
        if model != &self.model
            || program.model != model.id
            || program.backend != BackendKind::TestCpu
            || program.precision != PrecisionPlan::f32()
            || program.dataflow != self.graph
        {
            return Err(Error::invalid("host program/model/dataflow mismatch"));
        }
        let kernels = HostKernels.kernels();
        if program.operations.len() != self.graph.nodes.len()
            || program
                .operations
                .iter()
                .zip(&self.graph.nodes)
                .any(|(op, node)| {
                    op.op.id != node.id
                        || op.op.operation
                            != node.op.operation(
                                self.graph
                                    .logits
                                    .is_some_and(|id| node.outputs.contains(&id)),
                            )
                        || !kernels
                            .iter()
                            .any(|k| k.id == op.kernel && k.operation == op.op.operation)
                })
        {
            return Err(Error::invalid("host compiled kernel/node mismatch"));
        }
        Ok(())
    }
    pub(super) fn provider_reserve_state(&mut self, id: StateId, capacity: usize) -> Result<()> {
        if self.sequences.contains_key(&id) {
            return Err(Error::new(
                ErrorCode::Conflict,
                "physical state already reserved",
            ));
        }
        let required = self.required_state_bytes(capacity)?;
        let used = self
            .weight_bytes()
            .checked_add(self.graph.scratch_elements as u64 * crate::constants::F32_BYTES_U64)
            .and_then(|n| n.checked_add(self.inspect().reserved_bytes))
            .and_then(|n| n.checked_add(self.config.prefix_cache_bytes))
            .and_then(|n| n.checked_add(self.config.probe_bytes))
            .ok_or_else(|| Error::invalid("physical host budget overflow"))?;
        if used
            .checked_add(required)
            .is_none_or(|n| n > self.config.memory_bytes)
        {
            return Err(Error::new(
                ErrorCode::Capacity,
                "physical host state budget exhausted",
            ));
        }
        let sequence = self.empty_sequence(capacity)?;
        self.sequences.insert(id, sequence);
        Ok(())
    }
}
