use super::{deadline, request, support};
use infer_core::{Error, ErrorCode, ModelId, Result, StateId};
use infer_ir::{
    DeviceCapabilities, ExecutionProgram, ExecutionTask, ModelIr, OutputReadout, PrecisionPlan,
    StepPlan, TaskOutput,
};
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, RuntimeConfig};
use infer_spi::{BackendProvider, ResourceCommand, ResourceReply};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread::{self, ThreadId},
    time::{Duration, Instant},
};

struct QuoteBackend {
    inner: support::ProtocolBackend,
    prepared_on: Arc<Mutex<Option<ThreadId>>>,
    quotes: Arc<AtomicUsize>,
    fail: bool,
}
impl BackendProvider for QuoteBackend {
    type Ticket = <support::ProtocolBackend as BackendProvider>::Ticket;
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.inner.capabilities()
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        self.inner.execution_graph(model)
    }
    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()> {
        self.inner.validate_program(model, program)
    }
    fn prepare_state_reservation(&mut self, _: usize, _: OutputReadout) -> Result<()> {
        if self.fail {
            return Err(Error::new(ErrorCode::Backend, "quote preparation failed"));
        }
        *self
            .prepared_on
            .lock()
            .map_err(|e| Error::invariant(e.to_string()))? = Some(thread::current().id());
        Ok(())
    }
    fn state_reservation_bytes_for(&self, _: usize, _: OutputReadout) -> Result<Option<u64>> {
        if self
            .prepared_on
            .lock()
            .map_err(|e| Error::invariant(e.to_string()))?
            .is_none()
        {
            return Err(Error::invariant("quote preceded preparation"));
        }
        self.quotes.fetch_add(1, Ordering::AcqRel);
        Ok(Some(1024))
    }
    fn free_state_bytes(&self) -> Result<Option<u64>> {
        let prepared = self
            .prepared_on
            .lock()
            .map_err(|e| Error::invariant(e.to_string()))?;
        Ok(Some(if prepared.is_some() { 2048 } else { 0 }))
    }
    fn reserve_state_for(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        self.inner.reserve_state_for(state, capacity, readout)
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        self.inner.release_state(state)
    }
    fn submit(
        &mut self,
        p: &ExecutionProgram,
        s: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<Self::Ticket> {
        self.inner.submit(p, s, tasks)
    }
    fn poll(&mut self, ticket: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>> {
        self.inner.poll(ticket)
    }
}
fn engine(fail: bool) -> Result<Engine<QuoteBackend>> {
    let model = support::model(ModelId::ONE);
    let backend = QuoteBackend {
        inner: support::ProtocolBackend::new(16, 8, &model)?,
        prepared_on: Arc::default(),
        quotes: Arc::default(),
        fail,
    };
    let mut registry = KernelRegistry::default();
    registry.register(&support::DeclaredKernels)?;
    Engine::new(
        backend,
        model,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig::default(),
    )
}

#[test]
fn direct_admission_prepares_before_sizing_and_capacity_check() -> Result<()> {
    let mut engine = engine(false)?;
    assert_eq!(engine.backend().free_state_bytes()?, Some(0));
    engine.submit(request(1)?)?;
    assert_eq!(engine.backend().quotes.load(Ordering::Acquire), 1);
    assert_eq!(
        *engine
            .backend()
            .prepared_on
            .lock()
            .map_err(|e| Error::invariant(e.to_string()))?,
        Some(thread::current().id())
    );
    Ok(())
}

#[test]
fn threaded_quote_publishes_reclaimed_capacity_before_acknowledgement() -> Result<()> {
    let direct = engine(false)?;
    let prepared_on = direct.backend().prepared_on.clone();
    let quotes = direct.backend().quotes.clone();
    let mut engine = direct.into_threaded(4, Duration::from_millis(20))?;
    assert_eq!(engine.backend().free_state_bytes()?, Some(0));
    let prepared = engine.request_preparer()?.prepare(request(1)?)?;
    let mut quote = engine.admission_quote(&prepared)?;
    let limit = Instant::now() + Duration::from_secs(3);
    let bytes = loop {
        if let Some(ResourceReply::ReservationBytes(bytes)) = quote.poll()? {
            break bytes;
        }
        deadline(limit)?;
    };
    // No maintenance/tick between acknowledgement and the snapshot read.
    assert_eq!(bytes, Some(1024));
    assert_eq!(engine.backend().free_state_bytes()?, Some(2048));
    assert_eq!(quotes.load(Ordering::Acquire), 1);
    assert_ne!(
        *prepared_on
            .lock()
            .map_err(|e| Error::invariant(e.to_string()))?,
        Some(thread::current().id())
    );
    engine.submit_quoted_with_trace(prepared, bytes, None)?;
    Ok(())
}

#[test]
fn failed_preparation_does_not_publish_a_reservation_price() -> Result<()> {
    let mut engine = engine(true)?;
    let mut quote = engine
        .backend_mut()
        .begin_resource(ResourceCommand::ReservationBytes {
            capacity: 16,
            readout: OutputReadout::Logits,
        })?;
    assert_eq!(quote.poll().err().map(|e| e.code), Some(ErrorCode::Backend));
    assert_eq!(engine.backend().quotes.load(Ordering::Acquire), 0);
    assert_eq!(engine.backend().free_state_bytes()?, Some(0));
    assert_eq!(
        engine.submit(request(1)?).err().map(|e| e.code),
        Some(ErrorCode::Backend)
    );
    assert_eq!(engine.backend().quotes.load(Ordering::Acquire), 0);
    Ok(())
}
