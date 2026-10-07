#[cfg(target_os = "macos")]
use super::metal;
#[cfg(feature = "test-backends")]
use super::testing;
use super::{BackendChoice, Selection, cuda, cuda_available, metal_available, resolve};
#[cfg(all(target_os = "linux", feature = "cuda"))]
use infer_backend_cuda::{
    executor::{CudaBackend, CudaTicket},
    registry::CudaKernels,
};
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

/// Nanoseconds in one microsecond as `f64`, for Chrome trace timestamps in microseconds.
#[cfg(feature = "test-backends")]
const NANOS_PER_MICROSECOND_F64: f64 = 1000.0;

pub enum SelectedBackend {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    Cuda(Box<CudaBackend>),
    #[cfg(feature = "test-backends")]
    Reference(Box<ReferenceBackend>),
    #[cfg(feature = "test-backends")]
    Host(Box<HostBackend>),
    #[cfg(target_os = "macos")]
    Metal(Box<MetalBackend>),
}
pub enum SelectedTicket {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    Cuda(CudaTicket),
    #[cfg(feature = "test-backends")]
    Reference(ReferenceTicket),
    #[cfg(feature = "test-backends")]
    Host(HostTicket),
    #[cfg(target_os = "macos")]
    Metal(MetalTicket),
}
macro_rules! forward {
    ($s:expr,$method:ident $(,$arg:expr)*) => {match $s {
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        Self::Cuda(b)=>b.$method($($arg),*),
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
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Self::Cuda(b) => b.model(),
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
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Self::Cuda(_) => registry.register(&CudaKernels)?,
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
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Self::Cuda(_) => Err(Error::unsupported(
                "CUDA diagnostic fork requires independent state admission; not yet installed",
            )),
        }
    }
    #[cfg_attr(
        all(target_os = "macos", not(feature = "test-backends")),
        expect(
            clippy::unnecessary_wraps,
            reason = "The shared interface supports the test reference executor, which has no per-operation execution statistics"
        )
    )]
    #[cfg_attr(
        all(target_os = "linux", feature = "cuda", not(feature = "test-backends")),
        expect(
            clippy::missing_const_for_fn,
            reason = "The shared backend interface invokes non-const mutable Metal/host diagnostics in other builds"
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
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Self::Cuda(_) => None,
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
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Self::Cuda(b) => {
                serde_json::json!({"kind":"cuda-resident", "execution":"synchronous", "capabilities":b.capabilities(), "state_pool":b.pool_inspection()})
            }
        }
    }
    #[cfg_attr(
        all(target_os = "linux", feature = "cuda", not(feature = "test-backends")),
        expect(
            clippy::missing_const_for_fn,
            reason = "The shared backend interface invokes non-const mutable Metal/host diagnostics in other builds"
        )
    )]
    pub fn trace(&self) -> serde_json::Value {
        match self {
            #[cfg(feature = "test-backends")]
            Self::Reference(_) => serde_json::json!([]),
            #[cfg(feature = "test-backends")]
            Self::Host(b) => serde_json::json!(b.traces()),
            #[cfg(target_os = "macos")]
            Self::Metal(b) => serde_json::json!(b.traces()),
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Self::Cuda(_) => serde_json::json!([]),
        }
    }
    #[cfg_attr(
        all(target_os = "linux", feature = "cuda", not(feature = "test-backends")),
        expect(
            clippy::missing_const_for_fn,
            clippy::needless_pass_by_ref_mut,
            reason = "The shared backend interface invokes non-const mutable Metal/host diagnostics in other builds"
        )
    )]
    pub fn drain_layer_probes(&mut self) -> Vec<LayerProbe> {
        match self {
            #[cfg(feature = "test-backends")]
            Self::Reference(_) => vec![],
            #[cfg(feature = "test-backends")]
            Self::Host(b) => b.drain_layer_probes(),
            #[cfg(target_os = "macos")]
            Self::Metal(b) => b.drain_layer_probes(),
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Self::Cuda(_) => vec![],
        }
    }
    #[cfg_attr(
        all(target_os = "linux", feature = "cuda", not(feature = "test-backends")),
        expect(
            clippy::needless_pass_by_ref_mut,
            reason = "The shared backend interface invokes non-const mutable Metal/host diagnostics in other builds"
        )
    )]
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
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Self::Cuda(_) => {
                if bytes == 0 {
                    Ok(())
                } else {
                    Err(Error::unsupported("CUDA layer probes are not installed"))
                }
            }
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
                "name":format!("op:{} kernel:{}",t.op,t.kernel),"cat":"host_op","ph":"X","ts":t.timestamp_ns as f64/NANOS_PER_MICROSECOND_F64,
                "dur":t.elapsed_ns as f64/NANOS_PER_MICROSECOND_F64,"pid":1,"tid":1,"args":t})).collect::<Vec<_>>(),"inspection":b.inspect()})
            }
            #[cfg(target_os = "macos")]
            Self::Metal(b) => b.profile(),
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Self::Cuda(_) => {
                serde_json::json!({"scope":"CUDA per-operation profiling unavailable", "traceEvents":[]})
            }
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
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Self::Cuda(_) => vec![],
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
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Self::Cuda(_) => vec![],
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
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Self::Cuda(_) => "unavailable",
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
    fn control_ready(&self) -> bool {
        forward!(self, control_ready)
    }
    fn maintenance(&mut self) -> Result<()> {
        forward!(self, maintenance)
    }
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
    /// Wrappers must forward every defaulted trait method: the trait's default silently
    /// reports the conservative answer, which once disabled speculative decode unnoticed.
    ///
    /// `submit_shared`/`submit_shared_borrowed` are deliberately not implemented here: their
    /// defaults route through this wrapper's own `submit`, which already wraps the ticket.
    /// `launch_accepted` needs a per-variant ticket conversion, so it keeps the trait default.
    fn requires_async_checkpoint(&self) -> bool {
        forward!(self, requires_async_checkpoint)
    }
    fn pending_resource_releases(&self) -> bool {
        forward!(self, pending_resource_releases)
    }
    fn resource_epoch(&self) -> u64 {
        forward!(self, resource_epoch)
    }
    fn tracks_reservation_intent(&self) -> bool {
        forward!(self, tracks_reservation_intent)
    }
    fn set_waker(&mut self, wake: std::sync::Arc<dyn Fn() + Send + Sync>) {
        forward!(self, set_waker, wake);
    }

    /// Delegated explicitly: the trait default reports no speculation, which would silently
    /// disable the engine's speculative path even when the wrapped backend supports it.
    fn speculation_capability(&self) -> infer_ir::SpeculationCapability {
        forward!(self, speculation_capability)
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
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            (Self::Cuda(b), SelectedTicket::Cuda(t)) => b.completion_timing(t),
            #[cfg(feature = "test-backends")]
            _ => None,
        }
    }
    fn supports_control_checkpoint(&self) -> bool {
        forward!(self, supports_control_checkpoint)
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        forward!(self, execution_graph, model)
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
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Self::Cuda(b) => Ok(SelectedTicket::Cuda(b.submit(p, s, t)?)),
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
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            Self::Cuda(b) => Ok(SelectedTicket::Cuda(b.submit_borrowed(p, s, t)?)),
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
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            (Self::Cuda(b), SelectedTicket::Cuda(t)) => b.poll(t),
            #[cfg(feature = "test-backends")]
            _ => Err(Error::invariant("ticket/backend mismatch")),
        }
    }
}
pub fn load(
    path: &std::path::Path,
    memory_mib: u64,
    selection: &Selection,
) -> Result<SelectedBackend> {
    let choice = resolve(selection.kind, cuda_available(), metal_available())?;
    // Zero means "derive the budget from the device"; every backend resolves it locally.
    let memory_bytes = if memory_mib == 0 {
        0
    } else {
        memory_mib
            .checked_mul(crate::constants::MIB_U64)
            .ok_or_else(|| Error::invalid("device budget overflow"))?
    };
    match choice {
        #[cfg(feature = "test-backends")]
        BackendChoice::TestCpu => testing::load(path, memory_bytes, selection),
        BackendChoice::Cuda => cuda::load(path, memory_bytes, selection),
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
            // A zero budget derives the share from the Metal device; the CPU test-backend
            // default is compiled out without `test-backends`.
            0,
            &Selection {
                kind: BackendChoice::Metal,
                num_gpu_blocks_override: None,
                block_size: None,
                max_num_batched_tokens: None,
                upload_staging_mib: None,
                num_speculative_tokens: 0,
                gpu_memory_utilization: 0.0,
                autotune: false,
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
            ((capacity - 1) * backend.model_ir().hidden_size * size_of::<f32>()) as u64
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
