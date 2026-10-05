//! Inspection responsibilities.
use super::{HostBackend, HostInspection, LayerProbe, OpTrace};
use infer_core::{Error, ErrorCode, Result};
use infer_ir::ModelIr;
use infer_state::physical::PhysicalTensor;
use std::collections::VecDeque;

impl HostBackend {
    #[must_use]
    pub const fn model(&self) -> &ModelIr {
        &self.model
    }
    #[must_use]
    pub fn inspect(&self) -> HostInspection {
        HostInspection {
            sequences: self.sequences.len(),
            reserved_bytes: self.sequences.values().map(|s| s.reserved_bytes).sum(),
            allocated_bytes: self
                .sequences
                .values()
                .map(|s| {
                    s.tensors
                        .values()
                        .map(PhysicalTensor::allocated_bytes)
                        .sum::<usize>() as u64
                        + s.hidden.len() as u64 * self.model.hidden_size as u64 * 4
                        + s.logits.len() as u64 * 4
                })
                .sum(),
            tokens_executed: self.tokens_executed,
            trace_dropped: self.trace_dropped,
            prefix_hits: self.prefix_hits,
            prefix_entries: self.prefixes.len(),
            prefix_bytes: self.prefixes.bytes(),
            probe_samples: self.probes.len(),
            probe_dropped: self.probe_dropped,
            kv_cache: None,
        }
    }
    #[must_use]
    pub const fn traces(&self) -> &VecDeque<OpTrace> {
        &self.traces
    }
    #[must_use]
    pub const fn layer_probes(&self) -> &VecDeque<LayerProbe> {
        &self.probes
    }
    pub fn drain_layer_probes(&mut self) -> Vec<LayerProbe> {
        self.probes.drain(..).collect()
    }
    ///
    /// # Errors
    /// Returns a conflict error while work is in flight, or a capacity error if the probe budget is insufficient.
    pub fn enable_layer_probes(&mut self, bytes: u64) -> Result<()> {
        if !self.sequences.is_empty() {
            return Err(Error::new(
                ErrorCode::Conflict,
                "configure probes before admission",
            ));
        }
        let used = self
            .weight_bytes()
            .checked_add(self.graph.scratch_elements as u64 * 4)
            .and_then(|n| n.checked_add(self.config.prefix_cache_bytes))
            .and_then(|n| n.checked_add(bytes));
        if used.is_none_or(|n| n > self.config.memory_bytes) {
            return Err(Error::new(
                ErrorCode::Capacity,
                "layer probe budget exhausted",
            ));
        }
        self.config.probe_bytes = bytes;
        self.probes.clear();
        self.probe_dropped = 0;
        Ok(())
    }
}
