//! Inspection responsibilities.
use super::MetalBackend;
use infer_core::{Error, ErrorCode, Result};
use infer_ir::{ExecutionStats, LayerProbe, ModelIr, OpTrace};
use std::collections::VecDeque;

impl MetalBackend {
    #[must_use]
    pub fn available() -> bool {
        metal::Device::system_default().is_some_and(|d| d.has_unified_memory())
    }
    /// Device-recommended working-set budget for automatic service sizing.
    #[must_use]
    pub fn recommended_memory_bytes() -> Option<u64> {
        metal::Device::system_default().map(|device| device.recommended_max_working_set_size())
    }
    #[must_use]
    pub const fn model(&self) -> &ModelIr {
        &self.model
    }
    #[must_use]
    pub fn device_name(&self) -> &str {
        self.gpu.device.name()
    }
    #[must_use]
    pub fn inspect(&self) -> ExecutionStats {
        let reserved = self
            .sequences
            .values()
            .map(|s| s.reserved_bytes)
            .sum::<u64>();
        let cache = self.kv_inspection();
        let allocated = reserved + cache.active_blocks as u64 * self.kv_block_bytes;
        ExecutionStats {
            sequences: self.sequences.len(),
            reserved_bytes: reserved,
            allocated_bytes: allocated,
            tokens_executed: self.tokens_executed,
            trace_dropped: self.trace_dropped,
            prefix_hits: self.prefix_hits,
            prefix_entries: self.kv.prefix_count(),
            prefix_bytes: self.kv.prefix_bytes(),
            probe_samples: self.probes.len(),
            probe_dropped: self.probe_dropped,
            kv_cache: Some(cache),
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
        if self.busy.is_some() || !self.sequences.is_empty() {
            return Err(Error::new(
                ErrorCode::Conflict,
                "configure Metal probes before admission",
            ));
        }
        if self
            .base_bytes()?
            .checked_add(bytes)
            .is_none_or(|n| n > self.config.memory_bytes)
        {
            return Err(Error::new(
                ErrorCode::Capacity,
                "Metal probe budget exhausted",
            ));
        }
        self.config.probe_bytes = bytes;
        self.probes.clear();
        self.probe_dropped = 0;
        Ok(())
    }
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "Floating-point statistics, normalization, and deterministic sampling intentionally convert bounded counts to floating-point values"
    )]
    pub fn profile(&self) -> serde_json::Value {
        let first = self.commands.front().map_or(0, |t| t.gpu_start_ns);
        serde_json::json!({"scope":"Metal GPU command-buffer timing; op trace separately measures CPU encoding",
            "traceEvents":self.commands.iter().map(|t|serde_json::json!({"name":format!("step:{}",t.step),
                "ph":"X","cat":"metal_gpu_command","pid":1,"tid":1,
                "ts":t.gpu_start_ns.saturating_sub(first) as f64/crate::constants::NANOS_PER_MICROSECOND_F64,"dur":t.gpu_duration_ns as f64/crate::constants::NANOS_PER_MICROSECOND_F64,"args":t})).collect::<Vec<_>>(),
            "op_encoding":self.traces,"inspection":self.inspect(),"device":self.device_name(),
            "weight_load":self.load_plan,"execution_layout":{"prefill_chunk_tokens":self.config.prefill_chunk_tokens,
                "scratch_bytes":self.scratch_bytes,"compute_dtype":"F32","device_depth":1}})
    }
}
