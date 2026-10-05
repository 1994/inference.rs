#[cfg(target_os = "macos")]
use super::metal;
#[cfg(feature = "test-backends")]
use super::testing;
use super::{BackendChoice, Selection, cuda, metal_available, resolve};
#[cfg(feature = "test-backends")]
use infer_backend_host::{HostBackend, HostKernels, HostTicket};
#[cfg(target_os = "macos")]
use infer_backend_metal::{MetalBackend, MetalKernels, MetalTicket};
#[cfg(feature = "test-backends")]
use infer_backend_reference::{ReferenceBackend, ReferenceKernels, ReferenceTicket};
use infer_core::{Error, Result, StateId};
use infer_ir::{
    DeviceCapabilities, ExecutionProgram, ExecutionStats, ExecutionTask, ExecutionTiming,
    KvCacheInspection, LayerProbe, ModelIr, OpTrace, PageGrowth, StepPlan, TaskOutput,
};
use infer_kernel_api::KernelRegistry;
use infer_spi::BackendProvider;

pub enum SelectedBackend {
    #[cfg(feature = "test-backends")]
    Reference(Box<ReferenceBackend>),
    #[cfg(feature = "test-backends")]
    Host(Box<HostBackend>),
    #[cfg(target_os = "macos")]
    Metal(Box<MetalBackend>),
}
pub enum SelectedTicket {
    #[cfg(feature = "test-backends")]
    Reference(ReferenceTicket),
    #[cfg(feature = "test-backends")]
    Host(HostTicket),
    #[cfg(target_os = "macos")]
    Metal(MetalTicket),
}
macro_rules! forward {
    ($s:expr,$method:ident $(,$arg:expr)*) => {match $s {
        #[cfg(feature = "test-backends")]
        Self::Reference(b)=>b.$method($($arg),*),
        #[cfg(feature = "test-backends")]
        Self::Host(b)=>b.$method($($arg),*),
        #[cfg(target_os="macos")]
        Self::Metal(b)=>b.$method($($arg),*),
    }};
}
impl SelectedBackend {
    pub fn model_ir(&self) -> &ModelIr {
        match self {
            #[cfg(feature = "test-backends")]
            Self::Reference(b) => &b.model().ir,
            #[cfg(feature = "test-backends")]
            Self::Host(b) => b.model(),
            #[cfg(target_os = "macos")]
            Self::Metal(b) => b.model(),
        }
    }
    pub fn registry(&self) -> Result<KernelRegistry> {
        let mut registry = KernelRegistry::default();
        match self {
            #[cfg(feature = "test-backends")]
            Self::Reference(_) => registry.register(&ReferenceKernels)?,
            #[cfg(feature = "test-backends")]
            Self::Host(_) => registry.register(&HostKernels)?,
            #[cfg(target_os = "macos")]
            Self::Metal(_) => registry.register(&MetalKernels)?,
        }
        Ok(registry)
    }
    pub fn fresh(&self) -> Result<Self> {
        match self {
            #[cfg(feature = "test-backends")]
            Self::Reference(b) => Ok(Self::Reference(Box::new(ReferenceBackend::new(
                b.model().clone(),
            )?))),
            #[cfg(feature = "test-backends")]
            Self::Host(b) => Ok(Self::Host(Box::new(b.fresh()?))),
            #[cfg(target_os = "macos")]
            Self::Metal(b) => Ok(Self::Metal(Box::new(b.fresh()?))),
        }
    }
    #[cfg_attr(
        all(target_os = "macos", not(feature = "test-backends")),
        expect(
            clippy::unnecessary_wraps,
            reason = "The shared interface supports the test reference executor, which has no per-operation execution statistics"
        )
    )]
    pub fn execution_stats(&self) -> Option<ExecutionStats> {
        match self {
            #[cfg(feature = "test-backends")]
            Self::Reference(_) => None,
            #[cfg(feature = "test-backends")]
            Self::Host(b) => Some(b.inspect()),
            #[cfg(target_os = "macos")]
            Self::Metal(b) => Some(b.inspect()),
        }
    }
    pub fn inspection(&self) -> serde_json::Value {
        match self {
            #[cfg(feature = "test-backends")]
            Self::Reference(_) => {
                serde_json::json!({"kind":"legacy-reference","physical_state":false})
            }
            #[cfg(feature = "test-backends")]
            Self::Host(b) => {
                serde_json::json!({"kind":"host-dataflow","state":b.inspect(),"capabilities":b.capabilities()})
            }
            #[cfg(target_os = "macos")]
            Self::Metal(b) => {
                serde_json::json!({"kind":"metal-dataflow","device":b.device_name(),"state":b.inspect(),"capabilities":b.capabilities(),"weight_load":b.load_plan()})
            }
        }
    }
    pub fn trace(&self) -> serde_json::Value {
        match self {
            #[cfg(feature = "test-backends")]
            Self::Reference(_) => serde_json::json!([]),
            #[cfg(feature = "test-backends")]
            Self::Host(b) => serde_json::json!(b.traces()),
            #[cfg(target_os = "macos")]
            Self::Metal(b) => serde_json::json!(b.traces()),
        }
    }
    pub fn drain_layer_probes(&mut self) -> Vec<LayerProbe> {
        match self {
            #[cfg(feature = "test-backends")]
            Self::Reference(_) => vec![],
            #[cfg(feature = "test-backends")]
            Self::Host(b) => b.drain_layer_probes(),
            #[cfg(target_os = "macos")]
            Self::Metal(b) => b.drain_layer_probes(),
        }
    }
    pub fn enable_layer_probes(&mut self, bytes: u64) -> Result<()> {
        match self {
            #[cfg(feature = "test-backends")]
            Self::Reference(_) if bytes == 0 => Ok(()),
            #[cfg(feature = "test-backends")]
            Self::Reference(_) => Err(Error::unsupported("layer probes require --package")),
            #[cfg(feature = "test-backends")]
            Self::Host(b) => b.enable_layer_probes(bytes),
            #[cfg(target_os = "macos")]
            Self::Metal(b) => b.enable_layer_probes(bytes),
        }
    }
    #[cfg_attr(
        feature = "test-backends",
        expect(
            clippy::cast_precision_loss,
            reason = "Floating-point statistics, normalization, and deterministic sampling intentionally convert bounded counts to floating-point values"
        )
    )]
    pub fn profile(&self) -> serde_json::Value {
        match self {
            #[cfg(feature = "test-backends")]
            Self::Reference(_) => {
                serde_json::json!({"scope":"legacy reference lacks per-op timing","traceEvents":[]})
            }
            #[cfg(feature = "test-backends")]
            Self::Host(b) => {
                serde_json::json!({"scope":"host dataflow wall-clock op timing","traceEvents":b.traces().iter().map(|t|serde_json::json!({
                "name":format!("op:{} kernel:{}",t.op,t.kernel),"cat":"host_op","ph":"X","ts":t.timestamp_ns as f64/1000.0,
                "dur":t.elapsed_ns as f64/1000.0,"pid":1,"tid":1,"args":t})).collect::<Vec<_>>(),"inspection":b.inspect()})
            }
            #[cfg(target_os = "macos")]
            Self::Metal(b) => b.profile(),
        }
    }
}
impl infer_agent::AgentBackend for SelectedBackend {
    fn fresh(&self) -> Result<Self> {
        Self::fresh(self)
    }
    fn registry(&self) -> Result<KernelRegistry> {
        Self::registry(self)
    }
    fn inspection(&self) -> serde_json::Value {
        Self::inspection(self)
    }
    fn traces(&self) -> Vec<OpTrace> {
        match self {
            #[cfg(feature = "test-backends")]
            Self::Reference(_) => Vec::new(),
            #[cfg(feature = "test-backends")]
            Self::Host(b) => b.traces().iter().cloned().collect(),
            #[cfg(target_os = "macos")]
            Self::Metal(b) => b.traces().iter().cloned().collect(),
        }
    }
    fn probes(&self) -> Vec<LayerProbe> {
        match self {
            #[cfg(feature = "test-backends")]
            Self::Reference(_) => Vec::new(),
            #[cfg(feature = "test-backends")]
            Self::Host(b) => b.layer_probes().iter().cloned().collect(),
            #[cfg(target_os = "macos")]
            Self::Metal(b) => b.layer_probes().iter().cloned().collect(),
        }
    }
    fn trace_timing_scope(&self) -> &'static str {
        match self {
            #[cfg(feature = "test-backends")]
            Self::Reference(_) => "unavailable",
            #[cfg(feature = "test-backends")]
            Self::Host(_) => "cpu_wall",
            #[cfg(target_os = "macos")]
            Self::Metal(_) => "cpu_encoding",
        }
    }
    fn profile(&self) -> serde_json::Value {
        Self::profile(self)
    }
    fn execution_stats(&self) -> Option<ExecutionStats> {
        Self::execution_stats(self)
    }
}
impl BackendProvider for SelectedBackend {
    type Ticket = SelectedTicket;
    fn state_recipe(&self) -> Option<&infer_ir::StateRecipe> {
        forward!(self, state_recipe)
    }
    fn begin_resource(
        &mut self,
        command: infer_spi::ResourceCommand,
    ) -> Result<infer_spi::ResourceTicket> {
        forward!(self, begin_resource, command)
    }
    fn recycle_output(&mut self, state: StateId, output: infer_ir::ModelOutput) -> Result<()> {
        forward!(self, recycle_output, state, output)
    }
    fn recycle_batch(&mut self, outputs: Vec<TaskOutput>) -> Result<()> {
        forward!(self, recycle_batch, outputs)
    }

    fn identity(&self) -> &str {
        forward!(self, identity)
    }
    fn weight_backed_dataflow(&self) -> bool {
        forward!(self, weight_backed_dataflow)
    }
    fn capabilities(&self) -> DeviceCapabilities {
        forward!(self, capabilities)
    }
    fn state_reservation_bytes(&self, capacity: usize) -> Result<Option<u64>> {
        forward!(self, state_reservation_bytes, capacity)
    }
    fn state_reservation_bytes_for(
        &self,
        capacity: usize,
        readout: infer_ir::OutputReadout,
    ) -> Result<Option<u64>> {
        forward!(self, state_reservation_bytes_for, capacity, readout)
    }
    fn free_state_bytes(&self) -> Result<Option<u64>> {
        forward!(self, free_state_bytes)
    }
    fn kv_cache(&self) -> Option<KvCacheInspection> {
        forward!(self, kv_cache)
    }
    fn state_page_growth(&self, s: StateId) -> Result<Option<PageGrowth>> {
        forward!(self, state_page_growth, s)
    }
    fn reusable_prefix(&self, t: &[u32], m: usize) -> usize {
        forward!(self, reusable_prefix, t, m)
    }
    fn reusable_prefix_for(&self, state: StateId, tokens: &[u32], maximum: usize) -> usize {
        forward!(self, reusable_prefix_for, state, tokens, maximum)
    }
    fn reusable_prefix_shared(
        &self,
        state: StateId,
        tokens: infer_ir::TokenBuffer,
        maximum: usize,
    ) -> usize {
        forward!(self, reusable_prefix_shared, state, tokens, maximum)
    }
    fn reuse_prefix_shared(
        &mut self,
        state: StateId,
        tokens: infer_ir::TokenBuffer,
        maximum: usize,
    ) -> Result<usize> {
        forward!(self, reuse_prefix_shared, state, tokens, maximum)
    }
    fn reuse_prefix(&mut self, s: StateId, t: &[u32], m: usize) -> Result<usize> {
        forward!(self, reuse_prefix, s, t, m)
    }
    fn supports_recompute_preemption(&self) -> bool {
        forward!(self, supports_recompute_preemption)
    }
    fn completion_timing(&self, t: &SelectedTicket) -> Option<ExecutionTiming> {
        match (self, t) {
            #[cfg(feature = "test-backends")]
            (Self::Reference(b), SelectedTicket::Reference(t)) => b.completion_timing(t),
            #[cfg(feature = "test-backends")]
            (Self::Host(b), SelectedTicket::Host(t)) => b.completion_timing(t),
            #[cfg(target_os = "macos")]
            (Self::Metal(b), SelectedTicket::Metal(t)) => b.completion_timing(t),
            #[cfg(any(not(target_os = "macos"), feature = "test-backends"))]
            _ => None,
        }
    }
    fn supports_control_checkpoint(&self) -> bool {
        forward!(self, supports_control_checkpoint)
    }
    fn validate_program(&self, m: &ModelIr, p: &ExecutionProgram) -> Result<()> {
        forward!(self, validate_program, m, p)
    }
    fn reserve_state(&mut self, id: StateId, n: usize) -> Result<()> {
        forward!(self, reserve_state, id, n)
    }
    fn reserve_state_for(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: infer_ir::OutputReadout,
    ) -> Result<()> {
        forward!(self, reserve_state_for, state, capacity, readout)
    }
    fn reset_state(&mut self, id: StateId) -> Result<()> {
        forward!(self, reset_state, id)
    }
    fn release_state(&mut self, id: StateId) -> Result<()> {
        forward!(self, release_state, id)
    }
    fn capture_execution_state(&self) -> Result<Option<Vec<u8>>> {
        forward!(self, capture_execution_state)
    }
    fn validate_state_ownership(&self, s: &[(StateId, usize, usize)]) -> Result<()> {
        forward!(self, validate_state_ownership, s)
    }
    fn restore_execution_state(&mut self, s: Option<&[u8]>) -> Result<()> {
        forward!(self, restore_execution_state, s)
    }
    fn submit(
        &mut self,
        p: &ExecutionProgram,
        s: &StepPlan,
        t: Vec<ExecutionTask>,
    ) -> Result<SelectedTicket> {
        match self {
            #[cfg(feature = "test-backends")]
            Self::Reference(b) => Ok(SelectedTicket::Reference(b.submit(p, s, t)?)),
            #[cfg(feature = "test-backends")]
            Self::Host(b) => Ok(SelectedTicket::Host(b.submit(p, s, t)?)),
            #[cfg(target_os = "macos")]
            Self::Metal(b) => Ok(SelectedTicket::Metal(b.submit(p, s, t)?)),
        }
    }
    fn submit_borrowed(
        &mut self,
        p: &ExecutionProgram,
        s: &StepPlan,
        t: &[ExecutionTask],
    ) -> Result<SelectedTicket> {
        match self {
            #[cfg(feature = "test-backends")]
            Self::Reference(b) => Ok(SelectedTicket::Reference(b.submit_borrowed(p, s, t)?)),
            #[cfg(feature = "test-backends")]
            Self::Host(b) => Ok(SelectedTicket::Host(b.submit_borrowed(p, s, t)?)),
            #[cfg(target_os = "macos")]
            Self::Metal(b) => Ok(SelectedTicket::Metal(b.submit_borrowed(p, s, t)?)),
        }
    }
    fn poll(&mut self, t: &mut SelectedTicket) -> Result<Option<Vec<TaskOutput>>> {
        match (self, t) {
            #[cfg(feature = "test-backends")]
            (Self::Reference(b), SelectedTicket::Reference(t)) => b.poll(t),
            #[cfg(feature = "test-backends")]
            (Self::Host(b), SelectedTicket::Host(t)) => b.poll(t),
            #[cfg(target_os = "macos")]
            (Self::Metal(b), SelectedTicket::Metal(t)) => b.poll(t),
            #[cfg(any(not(target_os = "macos"), feature = "test-backends"))]
            _ => Err(Error::invariant("ticket/backend mismatch")),
        }
    }
}
pub fn load(
    path: &std::path::Path,
    memory_mib: u64,
    selection: Selection,
) -> Result<SelectedBackend> {
    let choice = resolve(selection.kind, false, metal_available())?;
    let memory_bytes = memory_mib
        .checked_mul(1024 * 1024)
        .ok_or_else(|| Error::invalid("device budget overflow"))?;
    match choice {
        #[cfg(feature = "test-backends")]
        BackendChoice::TestCpu => testing::load(path, memory_bytes, selection),
        BackendChoice::Cuda => cuda::load(),
        #[cfg(target_os = "macos")]
        BackendChoice::Metal => metal::load(path, memory_bytes, selection),
        _ => Err(Error::unsupported(
            "Metal backend requires macOS and a supported Metal device",
        )),
    }
}

#[cfg(all(test, target_os = "macos"))]
mod readout_tests {
    use super::*;
    #[test]
    fn selected_backend_preserves_compact_generation_reservation() -> Result<()> {
        if !metal_available() {
            return Ok(());
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../examples/qwen-hybrid-tiny");
        let mut backend = metal::load(
            &root,
            512 << 20,
            Selection {
                kind: BackendChoice::Metal,
                kv_cache_blocks: None,
                page_tokens: None,
                prefill_chunk_tokens: None,
                upload_staging_mib: None,
            },
        )?;
        let capacity = backend.model_ir().max_sequence;
        let full = backend
            .state_reservation_bytes_for(capacity, infer_ir::OutputReadout::Full)?
            .ok_or_else(|| Error::invariant("missing full budget"))?;
        let compact = backend
            .state_reservation_bytes_for(capacity, infer_ir::OutputReadout::Logits)?
            .ok_or_else(|| Error::invariant("missing compact budget"))?;
        assert_eq!(
            full - compact,
            ((capacity - 1) * backend.model_ir().hidden_size * 4) as u64
        );
        backend.reserve_state_for(StateId::ONE, capacity, infer_ir::OutputReadout::Logits)?;
        assert_eq!(
            backend.execution_stats().map(|stats| stats.reserved_bytes),
            Some(compact)
        );
        backend.release_state(StateId::ONE)?;
        assert_eq!(
            backend.execution_stats().map(|stats| stats.reserved_bytes),
            Some(0)
        );
        Ok(())
    }
}
