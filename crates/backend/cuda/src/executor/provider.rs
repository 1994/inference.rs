use super::{CudaBackend, CudaTicket};
use infer_core::{Result, StateId};
use infer_ir::{
    DeviceCapabilities, ExecutionProgram, ExecutionTask, ModelIr, OutputReadout, StepPlan,
    TaskOutput,
};
use infer_spi::BackendProvider;
impl BackendProvider for CudaBackend {
    type Ticket = CudaTicket;
    fn identity(&self) -> &str {
        &self.identity
    }
    fn weight_backed_dataflow(&self) -> bool {
        true
    }
    fn execution_profile(&self) -> Option<infer_ir::ExecutionProfileInspection> {
        Some(self.loaded.execution_profile())
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.capabilities.clone()
    }
    fn speculation_capability(&self) -> infer_ir::SpeculationCapability {
        self.capabilities.speculation
    }
    fn control_ready(&self) -> bool {
        !self.busy && self.fatal.is_none()
    }
    fn maintenance(&mut self) -> Result<()> {
        self.fatal
            .as_ref()
            .map_or(Ok(()), |error| Err(error.clone()))
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        if model != self.model() {
            return Err(infer_core::Error::invalid("execution graph model mismatch"));
        }
        Ok(self.loaded.graph().clone())
    }
    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()> {
        self.check_program(model, program)
    }
    fn state_reservation_bytes(&self, capacity: usize) -> Result<Option<u64>> {
        self.state_reservation_bytes_for(capacity, OutputReadout::Full)
    }
    fn state_reservation_bytes_for(
        &self,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<Option<u64>> {
        // Quote the same price reserve() will charge, or engine admission and backend
        // accounting disagree on how many sequences fit.
        let private = self.private_verification_for(capacity, readout);
        self.loaded
            .sequence_budget(capacity, readout, private)
            .map(Some)
    }
    fn free_state_bytes(&self) -> Result<Option<u64>> {
        self.available().map(Some)
    }
    fn reserve_state(&mut self, state: StateId, capacity: usize) -> Result<()> {
        self.reserve(state, capacity, OutputReadout::Full)
    }
    fn reserve_state_for(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        self.reserve(state, capacity, readout)
    }
    fn reset_state(&mut self, state: StateId) -> Result<()> {
        self.reset(state)
    }
    /// Restore a cached prompt prefix into `state` and report how many tokens it covered.
    ///
    /// Returns zero whenever the snapshot cannot be used verbatim (no match, non-logits readout,
    /// unknown state), which makes the engine fall back to a full prefill.
    fn reuse_prefix_shared(
        &mut self,
        state: StateId,
        tokens: infer_ir::TokenBuffer,
        maximum: usize,
    ) -> Result<usize> {
        let Some(entry) = self.prefix.take_match(&tokens, maximum) else {
            return Ok(0);
        };
        let Some(sequence) = self.states.get_mut(&state) else {
            self.prefix.insert(entry);
            return Ok(0);
        };
        // A slot-leased sequence decodes through the shared pool graph; restoring a prefix
        // into its private program state would desync the slot, so decline to reuse.
        if sequence.readout != OutputReadout::Logits || sequence.slot.is_some() {
            self.prefix.insert(entry);
            return Ok(0);
        }
        let matched = entry.tokens.len();
        if matched > maximum
            || matched > sequence.capacity
            || sequence.speculation.is_some() != entry.draft.is_some()
        {
            self.prefix.insert(entry);
            return Ok(0);
        }
        let device = self.loaded.device();
        let restored = (|| {
            entry
                .target
                .restore(device, &mut sequence.program, matched)?;
            if let (Some(spec), Some(draft)) = (&mut sequence.speculation, &entry.draft) {
                draft.program.restore(
                    device,
                    &mut spec.program,
                    matched.saturating_sub(crate::constants::MTP_KV_OFFSET),
                )?;
                spec.last_hidden.clone_from(&draft.hidden);
                spec.prompt_len = None;
            }
            Ok(())
        })();
        self.prefix.insert(entry);
        if let Err(error) = restored {
            sequence.poisoned = true;
            return Err(error);
        }
        sequence.history.clear();
        sequence.history.extend_from_slice(&tokens[..matched]);
        Ok(matched)
    }

    fn release_state(&mut self, state: StateId) -> Result<()> {
        self.release(state)
    }
    fn validate_state_ownership(&self, states: &[(StateId, usize, usize)]) -> Result<()> {
        self.ownership(states)
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<CudaTicket> {
        self.execute(program, step, &tasks)
    }
    fn submit_borrowed(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: &[ExecutionTask],
    ) -> Result<CudaTicket> {
        self.execute(program, step, tasks)
    }
    fn poll(&mut self, ticket: &mut CudaTicket) -> Result<Option<Vec<TaskOutput>>> {
        self.complete(ticket)
    }
}
