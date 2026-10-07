use infer_backend_reference::{
    ReferenceBackend, ReferenceKernels, ReferenceModel, ReferenceTicket,
};
use infer_core::{Error, FinishReason, ModelId, Result, StateId};
use infer_ir::{
    CanonicalRequest, DeviceCapabilities, ExecutionProgram, ExecutionTask, ModelIr, OutputReadout,
    PrecisionPlan, StepPlan, TaskOutput,
};
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, RuntimeConfig, runner::ThreadedBackend};
use infer_spi::{BackendProvider, ResourceCommand, ResourceReply};
use std::{
    sync::Arc, sync::atomic::AtomicBool, sync::atomic::AtomicUsize, sync::atomic::Ordering,
    sync::mpsc, time::Duration, time::Instant,
};

struct ReserveGate {
    inner: ReferenceBackend,
    draft_depth: usize,
    gate: Option<mpsc::Receiver<()>>,
    entered: mpsc::Sender<()>,
    owned: Arc<AtomicUsize>,
    reservation_limit: Arc<AtomicUsize>,
    reservation_attempts: Arc<AtomicUsize>,
    observed_work: Arc<AtomicUsize>,
    observed_sampling: Arc<AtomicUsize>,
    poll_gate: Arc<AtomicBool>,
}
impl BackendProvider for ReserveGate {
    type Ticket = ReferenceTicket;
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn speculation_capability(&self) -> infer_ir::SpeculationCapability {
        infer_ir::SpeculationCapability {
            draft_depth: self.draft_depth,
            greedy_only: true,
        }
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.inner.capabilities()
    }
    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()> {
        self.inner.validate_program(model, program)
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        self.inner.execution_graph(model)
    }
    fn reserve_state(&mut self, state: StateId, capacity: usize) -> Result<()> {
        self.reservation_attempts.fetch_add(1, Ordering::AcqRel);
        if self.owned.load(Ordering::Acquire) >= self.reservation_limit.load(Ordering::Acquire) {
            return Err(Error::new(
                infer_core::ErrorCode::Capacity,
                "resident budget busy",
            ));
        }
        self.inner.reserve_state(state, capacity)?;
        self.owned.fetch_add(1, Ordering::AcqRel);
        self.entered
            .send(())
            .map_err(|e| Error::invariant(e.to_string()))?;
        if let Some(gate) = self.gate.take() {
            gate.recv().map_err(|e| Error::invariant(e.to_string()))?;
        }
        Ok(())
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        self.inner.release_state(state)?;
        self.owned.fetch_sub(1, Ordering::AcqRel);
        Ok(())
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<ReferenceTicket> {
        self.observed_work
            .store(step.work.as_ptr() as usize, Ordering::Release);
        self.observed_sampling.fetch_add(
            tasks.iter().filter(|task| task.sampling.is_some()).count(),
            Ordering::Release,
        );
        self.inner.submit(program, step, tasks)
    }
    fn poll(&mut self, ticket: &mut ReferenceTicket) -> Result<Option<Vec<TaskOutput>>> {
        if self.poll_gate.load(Ordering::Acquire) {
            self.inner.poll(ticket)
        } else {
            Ok(None)
        }
    }
}
type TestEngine = Engine<ThreadedBackend<ReserveGate>>;
struct Fixture {
    engine: TestEngine,
    release: mpsc::Sender<()>,
    entered: mpsc::Receiver<()>,
    owned: Arc<AtomicUsize>,
    reservation_limit: Arc<AtomicUsize>,
    reservation_attempts: Arc<AtomicUsize>,
    observed_work: Arc<AtomicUsize>,
    observed_sampling: Arc<AtomicUsize>,
    poll_gate: Arc<AtomicBool>,
}
impl Fixture {
    fn new(gated: bool) -> Result<Self> {
        Self::with_draft(gated, 0)
    }
    fn with_draft(gated: bool, draft_depth: usize) -> Result<Self> {
        let model = ReferenceModel::fixture(ModelId::ONE, 7);
        let ir = model.ir.clone();
        let (release, gate) = mpsc::channel();
        let (entered, start) = mpsc::channel();
        let owned = Arc::new(AtomicUsize::new(0));
        let reservation_limit = Arc::new(AtomicUsize::new(usize::MAX));
        let reservation_attempts = Arc::new(AtomicUsize::new(0));
        let observed_work = Arc::new(AtomicUsize::new(0));
        let observed_sampling = Arc::new(AtomicUsize::new(0));
        let poll_gate = Arc::new(AtomicBool::new(true));
        let backend = ReserveGate {
            inner: ReferenceBackend::new(model)?,
            draft_depth,
            gate: gated.then_some(gate),
            entered,
            owned: owned.clone(),
            reservation_limit: reservation_limit.clone(),
            reservation_attempts: reservation_attempts.clone(),
            observed_work: observed_work.clone(),
            observed_sampling: observed_sampling.clone(),
            poll_gate: poll_gate.clone(),
        };
        let mut registry = KernelRegistry::default();
        registry.register(&ReferenceKernels)?;
        let engine = Engine::new(
            backend,
            ir,
            PrecisionPlan::f32(),
            &registry,
            RuntimeConfig {
                max_num_seqs: 1,
                ..RuntimeConfig::default()
            },
        )?
        .into_threaded(4, Duration::from_millis(20))?;
        Ok(Self {
            engine,
            release,
            entered: start,
            owned,
            reservation_limit,
            reservation_attempts,
            observed_work,
            observed_sampling,
            poll_gate,
        })
    }
    fn admit(&mut self, id: u64) -> Result<()> {
        let prepared = self.engine.request_preparer()?.prepare(request(id)?)?;
        let mut quote = self.engine.admission_quote(&prepared)?;
        let limit = Instant::now() + Duration::from_secs(3);
        let bytes = loop {
            if let Some(ResourceReply::ReservationBytes(bytes)) = quote.poll()? {
                break bytes;
            }
            deadline(limit)?;
        };
        self.engine.submit_quoted_with_trace(prepared, bytes, None)
    }
    fn drain(&mut self) -> Result<()> {
        let limit = Instant::now() + Duration::from_secs(3);
        while !self.engine.is_idle() {
            self.engine.tick(self.engine.now_us() + 1)?;
            deadline(limit)?;
        }
        self.engine.check_invariants()
    }
}
fn deadline(limit: Instant) -> Result<()> {
    if Instant::now() >= limit {
        return Err(Error::invariant("resource owner did not drain"));
    }
    std::thread::sleep(Duration::from_micros(100));
    Ok(())
}
fn request(id: u64) -> Result<CanonicalRequest> {
    serde_json::from_value(serde_json::json!({"id":id,"model":1,"input":{"Sequence":{"tokens":[1,2,3]}},"workload":{"Generate":{"max_new_tokens":2}}})).map_err(|e| Error::invalid(e.to_string()))
}
#[test]
fn resource_timeout_retains_ownership_until_ack_and_later_requests_progress() -> Result<()> {
    let mut fixture = Fixture::new(true)?;
    fixture.admit(1)?;
    fixture
        .entered
        .recv_timeout(Duration::from_secs(3))
        .map_err(|e| Error::invariant(e.to_string()))?;
    fixture.engine.tick(20_001)?;
    assert!(matches!(
        fixture
            .engine
            .pending_terminal(infer_core::RequestId::ONE)
            .map(|done| done.reason),
        Some(FinishReason::Failed(_))
    ));
    assert_eq!(fixture.owned.load(Ordering::Acquire), 1);
    assert_eq!(fixture.engine.inspect().state.sequence_count, 1);
    fixture.engine.check_invariants()?;
    fixture
        .release
        .send(())
        .map_err(|e| Error::invariant(e.to_string()))?;
    fixture.drain()?;
    assert_eq!(fixture.owned.load(Ordering::Acquire), 0);
    fixture.admit(2)?;
    fixture.drain()?;
    assert_eq!(
        fixture
            .engine
            .request(infer_core::RequestId::new(2)?)?
            .completed
            .as_ref()
            .map(|done| &done.reason),
        Some(&FinishReason::Length)
    );
    Ok(())
}
#[test]
fn reservation_cancel_is_idempotent_and_waits_for_real_ack() -> Result<()> {
    let mut fixture = Fixture::new(true)?;
    fixture.admit(1)?;
    fixture
        .entered
        .recv_timeout(Duration::from_secs(3))
        .map_err(|e| Error::invariant(e.to_string()))?;
    assert_eq!(fixture.engine.cancel(infer_core::RequestId::ONE)?, []);
    assert_eq!(fixture.engine.cancel(infer_core::RequestId::ONE)?, []);
    assert_eq!(fixture.owned.load(Ordering::Acquire), 1);
    fixture.engine.check_invariants()?;
    fixture
        .release
        .send(())
        .map_err(|e| Error::invariant(e.to_string()))?;
    fixture.drain()?;
    assert_eq!(fixture.owned.load(Ordering::Acquire), 0);
    Ok(())
}
#[test]
fn published_but_unconsumed_reservation_is_compensated_on_ticket_drop() -> Result<()> {
    let mut fixture = Fixture::new(false)?;
    let ticket = fixture
        .engine
        .backend_mut()
        .begin_resource(ResourceCommand::Reserve {
            state: StateId::ONE,
            capacity: 16,
            readout: OutputReadout::Full,
        })?;
    fixture
        .entered
        .recv_timeout(Duration::from_secs(3))
        .map_err(|e| Error::invariant(e.to_string()))?;
    // Sizing is ordered behind reservation publication; observing its reply makes the race deterministic.
    let mut barrier =
        fixture
            .engine
            .backend_mut()
            .begin_resource(ResourceCommand::ReservationBytes {
                capacity: 16,
                readout: OutputReadout::Full,
            })?;
    let limit = Instant::now() + Duration::from_secs(3);
    while barrier.poll()?.is_none() {
        deadline(limit)?;
    }
    assert_eq!(fixture.owned.load(Ordering::Acquire), 1);
    drop(ticket);
    while fixture.owned.load(Ordering::Acquire) > 0
        || fixture.engine.backend().pending_resource_releases()
    {
        deadline(limit)?;
    }
    Ok(())
}

#[test]
fn batch_metadata_is_shared_across_the_device_owner_handoff() -> Result<()> {
    let mut fixture = Fixture::new(false)?;
    let mut reserve = fixture
        .engine
        .backend_mut()
        .begin_resource(ResourceCommand::Reserve {
            state: StateId::ONE,
            capacity: 16,
            readout: OutputReadout::Full,
        })?;
    let limit = Instant::now() + Duration::from_secs(3);
    while reserve.poll()?.is_none() {
        deadline(limit)?;
    }
    let program = fixture.engine.program().clone();
    let step = Arc::new(StepPlan {
        id: infer_core::StepId::ONE,
        decision: infer_core::DecisionId::ONE,
        program: program.id,
        role: infer_ir::ExecutionRole::Prefill,
        work: vec![infer_ir::PlannedWork {
            request: infer_core::RequestId::ONE,
            state: StateId::ONE,
            token_count: 3,
            role: infer_ir::ExecutionRole::Prefill,
        }],
        cost: infer_ir::CostEstimate::default(),
        graph: None,
        quantum_overrun: false,
    });
    let mut ticket = fixture.engine.backend_mut().submit_shared(
        &program,
        step.clone(),
        vec![ExecutionTask {
            request: infer_core::RequestId::ONE,
            state: StateId::ONE,
            tokens: vec![1, 2, 3].into(),

            sampling: None,
        }],
    )?;
    while fixture.engine.backend_mut().poll(&mut ticket)?.is_none() {
        deadline(limit)?;
    }
    assert_eq!(
        fixture.observed_work.load(Ordering::Acquire),
        step.work.as_ptr() as usize
    );
    fixture.engine.backend_mut().release_state(StateId::ONE)?;
    while fixture.engine.backend().pending_resource_releases() {
        deadline(limit)?;
    }
    assert_eq!(fixture.owned.load(Ordering::Acquire), 0);
    Ok(())
}

#[test]
fn dropped_compute_ticket_waits_for_fence_then_allows_next_submission() -> Result<()> {
    let mut fixture = Fixture::new(false)?;
    let mut reserve = fixture
        .engine
        .backend_mut()
        .begin_resource(ResourceCommand::Reserve {
            state: StateId::ONE,
            capacity: 16,
            readout: OutputReadout::Full,
        })?;
    let limit = Instant::now() + Duration::from_secs(3);
    while reserve.poll()?.is_none() {
        deadline(limit)?;
    }
    fixture.poll_gate.store(false, Ordering::Release);
    let program = fixture.engine.program().clone();
    let step = Arc::new(StepPlan {
        id: infer_core::StepId::ONE,
        decision: infer_core::DecisionId::ONE,
        program: program.id,
        role: infer_ir::ExecutionRole::Prefill,
        work: vec![infer_ir::PlannedWork {
            request: infer_core::RequestId::ONE,
            state: StateId::ONE,
            token_count: 3,
            role: infer_ir::ExecutionRole::Prefill,
        }],
        cost: infer_ir::CostEstimate::default(),
        graph: None,
        quantum_overrun: false,
    });
    let tasks = vec![ExecutionTask {
        request: infer_core::RequestId::ONE,
        state: StateId::ONE,
        tokens: vec![1, 2, 3].into(),

        sampling: None,
    }];
    let ticket =
        fixture
            .engine
            .backend_mut()
            .submit_shared_borrowed(&program, step.clone(), &tasks)?;
    while fixture.observed_work.load(Ordering::Acquire) == 0 {
        deadline(limit)?;
    }
    drop(ticket);
    assert_eq!(
        fixture
            .engine
            .backend_mut()
            .submit_shared_borrowed(&program, step.clone(), &tasks)
            .err()
            .map(|e| e.code),
        Some(infer_core::ErrorCode::Conflict)
    );
    fixture.poll_gate.store(true, Ordering::Release);
    let mut next = loop {
        match fixture
            .engine
            .backend_mut()
            .submit_shared_borrowed(&program, step.clone(), &tasks)
        {
            Ok(ticket) => break ticket,
            Err(error)
                if matches!(
                    error.code,
                    infer_core::ErrorCode::Capacity | infer_core::ErrorCode::Conflict
                ) =>
            {
                deadline(limit)?;
            }
            Err(error) => return Err(error),
        }
    };
    while fixture.engine.backend_mut().poll(&mut next)?.is_none() {
        deadline(limit)?;
    }
    fixture.engine.backend_mut().release_state(StateId::ONE)?;
    while fixture.engine.backend().pending_resource_releases() {
        deadline(limit)?;
    }
    assert_eq!(fixture.owned.load(Ordering::Acquire), 0);
    Ok(())
}

#[test]
fn independent_ready_work_is_preplanned_and_new_admission_invalidates_it() -> Result<()> {
    let mut fixture = Fixture::new(false)?;
    for id in 1..=3 {
        fixture.admit(id)?;
    }
    fixture.poll_gate.store(false, Ordering::Release);
    let limit = Instant::now() + Duration::from_secs(3);
    while !fixture.engine.inspect().cpu.prepared_next {
        fixture.engine.tick(fixture.engine.now_us() + 1)?;
        deadline(limit)?;
    }
    assert!(fixture.engine.inspect().inflight_step.is_some());
    let prepared = fixture.engine.request_preparer()?.prepare(request(4)?)?;
    fixture
        .engine
        .submit_quoted_with_trace(prepared, None, None)?;
    assert!(!fixture.engine.inspect().cpu.prepared_next);
    fixture.engine.cancel(infer_core::RequestId::new(2)?)?;
    fixture.poll_gate.store(true, Ordering::Release);
    fixture.drain()?;
    assert_eq!(fixture.owned.load(Ordering::Acquire), 0);
    assert_eq!(
        fixture
            .engine
            .request(infer_core::RequestId::new(2)?)?
            .completed
            .as_ref()
            .map(|done| &done.reason),
        Some(&FinishReason::Cancelled)
    );
    for id in [1, 3, 4] {
        assert_eq!(
            fixture
                .engine
                .request(infer_core::RequestId::new(id)?)?
                .completed
                .as_ref()
                .map(|done| &done.reason),
            Some(&FinishReason::Length)
        );
    }
    Ok(())
}

#[test]
fn threaded_dispatch_preserves_speculation_and_supplies_decode_sampling() -> Result<()> {
    for depth in [0, 2] {
        let mut fixture = Fixture::with_draft(false, depth)?;
        assert_eq!(
            fixture
                .engine
                .backend()
                .speculation_capability()
                .draft_depth,
            depth
        );
        fixture.admit(1)?;
        fixture.drain()?;
        assert_eq!(
            fixture.observed_sampling.load(Ordering::Acquire) > 0,
            depth > 0,
            "a loaded draft must receive sampling through the threaded service path"
        );
    }
    Ok(())
}

#[test]
fn reservation_capacity_waits_for_progress_and_retries_without_losing_tokens() -> Result<()> {
    let mut fixture = Fixture::new(false)?;
    fixture.reservation_limit.store(1, Ordering::Release);
    fixture.poll_gate.store(false, Ordering::Release);
    fixture.admit(1)?;
    fixture
        .entered
        .recv_timeout(Duration::from_secs(3))
        .map_err(|e| Error::invariant(e.to_string()))?;
    fixture.admit(2)?;
    let limit = Instant::now() + Duration::from_secs(3);
    while fixture.reservation_attempts.load(Ordering::Acquire) < 2 {
        deadline(limit)?;
    }
    // Consume the failed acknowledgement and stabilize the snapshot. The busy
    // backend must not receive a new reservation on every scheduler poll.
    for _ in 0..100 {
        fixture.engine.tick(fixture.engine.now_us() + 1)?;
        deadline(limit)?;
    }
    let attempts = fixture.reservation_attempts.load(Ordering::Acquire);
    for _ in 0..100 {
        fixture.engine.tick(fixture.engine.now_us() + 1)?;
        deadline(limit)?;
    }
    assert_eq!(
        fixture.reservation_attempts.load(Ordering::Acquire),
        attempts
    );
    assert!(
        fixture
            .engine
            .request(infer_core::RequestId::new(2)?)?
            .completed
            .is_none()
    );
    fixture.engine.check_invariants()?;
    fixture.poll_gate.store(true, Ordering::Release);
    fixture.drain()?;
    for id in [1, 2] {
        let record = fixture.engine.request(infer_core::RequestId::new(id)?)?;
        assert_eq!(
            record.completed.as_ref().map(|done| &done.reason),
            Some(&FinishReason::Length)
        );
        assert_eq!(record.generated.len(), 2);
    }
    assert_eq!(fixture.owned.load(Ordering::Acquire), 0);
    Ok(())
}

#[test]
fn deferred_reservation_can_time_out_or_cancel_without_a_backend_state() -> Result<()> {
    for cancel in [false, true] {
        let mut fixture = Fixture::new(false)?;
        fixture.reservation_limit.store(0, Ordering::Release);
        fixture.admit(1)?;
        let limit = Instant::now() + Duration::from_secs(3);
        for _ in 0..100 {
            fixture.engine.tick(fixture.engine.now_us() + 1)?;
            deadline(limit)?;
        }
        assert_eq!(fixture.reservation_attempts.load(Ordering::Acquire), 1);
        fixture.engine.check_invariants()?;
        if cancel {
            fixture.engine.cancel(infer_core::RequestId::ONE)?;
        } else {
            fixture.engine.tick(20_001)?;
        }
        fixture.drain()?;
        let reason = &fixture
            .engine
            .request(infer_core::RequestId::ONE)?
            .completed
            .as_ref()
            .ok_or_else(|| Error::invariant("deferred reservation did not terminate"))?
            .reason;
        if cancel {
            assert_eq!(reason, &FinishReason::Cancelled);
        } else {
            assert!(matches!(reason, FinishReason::Failed(_)));
        }
        assert_eq!(fixture.owned.load(Ordering::Acquire), 0);
    }
    Ok(())
}

/// A direct backend that settles resource commands inline but owns reservation intent, so a
/// busy `Reserve` is always observed on the engine's very first poll instead of a later one.
struct InlineReservationIntent {
    inner: ReferenceBackend,
    attempts: Arc<AtomicUsize>,
}
impl BackendProvider for InlineReservationIntent {
    type Ticket = ReferenceTicket;
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.inner.capabilities()
    }
    fn tracks_reservation_intent(&self) -> bool {
        true
    }
    fn state_reservation_bytes(&self, capacity: usize) -> Result<Option<u64>> {
        self.inner.state_reservation_bytes(capacity)
    }
    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()> {
        self.inner.validate_program(model, program)
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        self.inner.execution_graph(model)
    }
    fn reserve_state(&mut self, _state: StateId, _capacity: usize) -> Result<()> {
        self.attempts.fetch_add(1, Ordering::AcqRel);
        Err(Error::new(
            infer_core::ErrorCode::Capacity,
            "resident budget busy",
        ))
    }
    fn release_state(&mut self, _state: StateId) -> Result<()> {
        // The rejected reservation never created backend state.
        Ok(())
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<ReferenceTicket> {
        self.inner.submit(program, step, tasks)
    }
    fn poll(&mut self, ticket: &mut ReferenceTicket) -> Result<Option<Vec<TaskOutput>>> {
        self.inner.poll(ticket)
    }
}

#[test]
fn inline_busy_reservation_defers_instead_of_failing_admission() -> Result<()> {
    // A direct backend settles `begin_resource` inline, so the busy rejection is always
    // observed on the engine's first poll. When the backend owns reservation intent the
    // rejection must be deferred exactly like a later acknowledgement: keep the request
    // and its host state, and publish no retry until resource ownership progresses.
    let model = ReferenceModel::fixture(ModelId::ONE, 7);
    let ir = model.ir.clone();
    let attempts = Arc::new(AtomicUsize::new(0));
    let backend = InlineReservationIntent {
        inner: ReferenceBackend::new(model)?,
        attempts: attempts.clone(),
    };
    let mut registry = KernelRegistry::default();
    registry.register(&ReferenceKernels)?;
    let mut engine = Engine::new(
        backend,
        ir,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig {
            max_num_seqs: 1,
            ..RuntimeConfig::default()
        },
    )?;
    let prepared = engine.request_preparer()?.prepare(request(1)?)?;
    let mut quote = engine.admission_quote(&prepared)?;
    let bytes = loop {
        if let Some(ResourceReply::ReservationBytes(bytes)) = quote.poll()? {
            break bytes;
        }
    };
    engine.submit_quoted_with_trace(prepared, bytes, None)?;
    assert_eq!(attempts.load(Ordering::Acquire), 1);
    engine.check_invariants()?;
    for _ in 0..100 {
        engine.tick(engine.now_us() + 1)?;
    }
    assert_eq!(attempts.load(Ordering::Acquire), 1);
    assert!(
        engine
            .request(infer_core::RequestId::ONE)?
            .completed
            .is_none()
    );
    engine.cancel(infer_core::RequestId::ONE)?;
    let limit = Instant::now() + Duration::from_secs(3);
    while !engine.is_idle() {
        engine.tick(engine.now_us() + 1)?;
        deadline(limit)?;
    }
    engine.check_invariants()?;
    Ok(())
}
