//! Restore responsibilities.
use super::{Engine, ReplayAction, RequestRecord, RuntimeSnapshot};
use infer_core::{Error, Result};
use infer_kernel_api::KernelRegistry;
use infer_scheduler::{CalibratedCosts, CostAwarePolicy, ResourceAdmission};
use infer_spi::{AdmissionPolicy, BackendProvider, CostModelProvider, WorkloadProvider};
use infer_workloads::NativeWorkloads;

impl<B: BackendProvider> Engine<B, CostAwarePolicy> {
    ///
    /// # Errors
    /// Returns a validation, backend, or invariant error for incompatible checkpoint versions, identities, providers, programs, or state ownership.
    pub fn restore(
        backend: B,
        registry: &KernelRegistry,
        snapshot: RuntimeSnapshot,
    ) -> Result<Self> {
        Self::restore_with_workloads(backend, registry, snapshot, NativeWorkloads)
    }
    ///
    /// # Errors
    /// Returns a validation, backend, or invariant error if the checkpoint or workload provider is incompatible.
    pub fn restore_with_workloads(
        backend: B,
        registry: &KernelRegistry,
        snapshot: RuntimeSnapshot,
        workloads: impl WorkloadProvider + Send + Sync + 'static,
    ) -> Result<Self> {
        let costs = CalibratedCosts::new(
            format!("{}:{}", backend.identity(), snapshot.program.id),
            snapshot.config.cost_model.clone(),
        )?;
        Self::restore_with_cost_model(backend, registry, snapshot, workloads, costs)
    }
    ///
    /// # Errors
    /// Returns a validation, backend, or invariant error if the checkpoint, workload provider, or calibration is incompatible.
    pub fn restore_with_cost_model(
        backend: B,
        registry: &KernelRegistry,
        snapshot: RuntimeSnapshot,
        workloads: impl WorkloadProvider + Send + Sync + 'static,
        costs: impl CostModelProvider + Send + Sync + 'static,
    ) -> Result<Self> {
        let admission = ResourceAdmission::new(snapshot.config.admission.clone())?;
        Self::restore_with_planning(backend, registry, snapshot, workloads, costs, admission)
    }
    ///
    /// # Errors
    /// Returns a validation, backend, or invariant error if checkpoint state or planning-provider identities are incompatible.
    pub fn restore_with_planning(
        backend: B,
        registry: &KernelRegistry,
        snapshot: RuntimeSnapshot,
        workloads: impl WorkloadProvider + Send + Sync + 'static,
        costs: impl CostModelProvider + Send + Sync + 'static,
        admission: impl AdmissionPolicy + Send + Sync + 'static,
    ) -> Result<Self> {
        snapshot.validate_compatibility(&backend, &workloads, &costs, &admission)?;
        let mut engine = Self::new(
            backend,
            snapshot.model.clone(),
            snapshot.program.precision.clone(),
            registry,
            snapshot.config.clone(),
        )?
        .with_workloads(workloads)?
        .with_cost_model(costs)?
        .with_admission_policy(admission)?;
        engine.costs.restore_state(snapshot.cost_state.as_deref())?;
        if snapshot.program != engine.program {
            return Err(Error::invalid(
                "snapshot program differs from compiler output",
            ));
        }
        engine
            .backend
            .restore_execution_state(snapshot.execution_state.as_deref())?;
        snapshot.validate_capacities()?;
        for (id, record) in &snapshot.requests {
            engine.validate_checkpoint_record(*id, record, &snapshot)?;
        }
        engine.state = snapshot.state;
        engine.state.prepare_storage()?;
        for (id, record) in snapshot.requests {
            let record = engine.restore_host_record(record)?;
            engine.host.requests.insert(id, record)?;
        }
        engine.host.cost_epoch = snapshot.cost_epoch;
        engine.validate_backend_state()?;
        engine.seen_requests = snapshot.seen_requests;
        engine.retired_request_floor = snapshot.retired_request_floor;
        engine.tenants = snapshot.tenants;
        engine.ids = snapshot.ids;
        engine.global_progress_epoch = snapshot.global_progress_epoch;
        engine.now_us = snapshot.now_us;
        engine.observations.unix_origin_us = engine
            .observations
            .unix_origin_us
            .saturating_sub(snapshot.now_us);
        if snapshot.preemption_focus.is_some_and(|id| {
            engine
                .host
                .requests
                .get(id)
                .is_none_or(|r| r.status.terminal())
        }) {
            return Err(Error::invalid("checkpoint preemption focus is not active"));
        }
        engine.preemption_focus = snapshot.preemption_focus;
        engine.host.queues = snapshot.queues;
        engine.resource_epoch = snapshot.resource_epoch;
        engine.guard = snapshot.guard;
        engine.decisions = snapshot.decisions;
        engine
            .host
            .history
            .prepare_archive(&mut engine.decisions, engine.config.history_capacity)?;
        engine.host.history_bytes = snapshot.actions.iter().fold(0usize, |bytes, action| {
            bytes.saturating_add(action.retained_bytes())
        });
        if engine.host.history_bytes > engine.config.max_history_bytes {
            return Err(Error::invalid("checkpoint replay bytes exceed host budget"));
        }
        engine.host.history_tokens = snapshot
            .actions
            .iter()
            .map(ReplayAction::retained_tokens)
            .sum();
        engine.actions = snapshot.actions;
        engine.actions.reserve(
            engine
                .config
                .history_capacity
                .saturating_sub(engine.actions.len()),
        );
        engine.dropped_actions = snapshot.dropped_actions;
        for observation in snapshot.pending_cost_observations {
            engine.enqueue_cost_observation(observation, false)?;
        }
        engine.check_invariants()?;
        Ok(engine)
    }
    pub(super) fn restore_host_record(&self, mut record: RequestRecord) -> Result<RequestRecord> {
        let prompt = record
            .plan
            .units
            .get(record.unit.min(record.plan.units.len() - 1))
            .ok_or_else(|| Error::invalid("checkpoint prompt owner missing"))?;
        if &record.context.prompt != prompt {
            return Err(Error::invalid(
                "checkpoint context differs from planned unit",
            ));
        }
        record.context.prompt = prompt.clone();
        let capacity = crate::preparation::generation_capacity(&record.request);
        let credit = self
            .host
            .credits
            .reserve(crate::preparation::retained_tokens(&record.plan, capacity)?)?;
        let mut generated = Vec::with_capacity(capacity);
        generated.extend_from_slice(&record.generated);
        record.generated = generated.into();
        record.generated.attach_credit(credit.clone());
        record.context.prompt.attach_credit(credit.clone());
        record.host_lease = Some(credit);
        let bytes = crate::preparation::retained_bytes(
            &record.request,
            &record.plan,
            &self.model,
            self.config.block_size,
        )?;
        let byte_credit = self.host.bytes.reserve(bytes)?;
        record.generated.attach_byte_credit(byte_credit.clone());
        record
            .context
            .prompt
            .attach_byte_credit(byte_credit.clone());
        for unit in &mut record.plan.units {
            unit.attach_byte_credit(byte_credit.clone());
        }
        if let Some(output) = record
            .completed
            .as_mut()
            .and_then(|done| done.output.as_mut())
        {
            output.attach_credit(byte_credit.clone());
        }
        record.byte_lease = Some(byte_credit);
        if let Some(infer_ir::WorkloadOutput::Tokens(tokens)) = record
            .completed
            .as_mut()
            .and_then(|done| done.output.as_mut())
        {
            if *tokens != record.generated {
                return Err(Error::invalid(
                    "checkpoint completed tokens differ from generated tail",
                ));
            }
            *tokens = record.generated.clone();
        }
        Ok(record)
    }
}
