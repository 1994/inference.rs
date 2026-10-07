//! `BackendProvider` adaptation; lifecycle helpers live alongside their owning subsystem.
use super::{Checkpoint, HostBackend, HostTicket};
use infer_core::{Error, Result, StateId};
use infer_ir::{
    DeviceCapabilities, ExecutionProgram, ExecutionTask, ExecutionTiming, ModelIr, StepPlan,
    TaskOutput,
};
use infer_spi::BackendProvider;

impl BackendProvider for HostBackend {
    type Ticket = HostTicket;
    fn weight_backed_dataflow(&self) -> bool {
        true
    }
    fn identity(&self) -> &str {
        &self.identity
    }
    fn capabilities(&self) -> DeviceCapabilities {
        let mut c = DeviceCapabilities::reference();
        c.memory_bytes = self.config.memory_bytes;
        c
    }
    fn state_reservation_bytes(&self, capacity: usize) -> Result<Option<u64>> {
        Ok(Some(self.required_state_bytes(capacity)?))
    }
    fn free_state_bytes(&self) -> Result<Option<u64>> {
        let used = self
            .weight_bytes()
            .checked_add(self.graph.scratch_elements as u64 * crate::constants::F32_BYTES_U64)
            .and_then(|n| n.checked_add(self.inspect().reserved_bytes))
            .and_then(|n| n.checked_add(self.config.prefix_cache_bytes))
            .and_then(|n| n.checked_add(self.config.probe_bytes))
            .ok_or_else(|| Error::invalid("host state budget overflow"))?;
        Ok(Some(self.config.memory_bytes.saturating_sub(used)))
    }
    fn completion_timing(&self, ticket: &HostTicket) -> Option<ExecutionTiming> {
        ticket.result.is_none().then_some(ticket.timing)
    }
    fn supports_control_checkpoint(&self) -> bool {
        true
    }
    fn supports_recompute_preemption(&self) -> bool {
        true
    }
    fn reusable_prefix(&self, tokens: &[u32], maximum: usize) -> usize {
        self.prefixes.matched_tokens(tokens, maximum)
    }
    fn reuse_prefix(&mut self, id: StateId, tokens: &[u32], maximum: usize) -> Result<usize> {
        Ok(self.provider_reuse_prefix(id, tokens, maximum))
    }

    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        if model != &self.model {
            return Err(Error::invalid("execution graph model mismatch"));
        }
        Ok(self.graph.clone())
    }
    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()> {
        self.provider_validate_program(model, program)
    }

    fn reserve_state(&mut self, id: StateId, capacity: usize) -> Result<()> {
        self.provider_reserve_state(id, capacity)
    }

    fn reset_state(&mut self, id: StateId) -> Result<()> {
        let capacity = self
            .sequences
            .get(&id)
            .ok_or_else(|| Error::invalid("reset unknown physical state"))?
            .capacity;
        self.sequences.insert(id, self.empty_sequence(capacity)?);
        Ok(())
    }
    fn release_state(&mut self, id: StateId) -> Result<()> {
        self.sequences
            .remove(&id)
            .ok_or_else(|| Error::invalid("release unknown physical state"))?;
        Ok(())
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<HostTicket> {
        self.provider_submit(program, step, &tasks)
    }

    fn poll(&mut self, ticket: &mut HostTicket) -> Result<Option<Vec<TaskOutput>>> {
        ticket
            .result
            .take()
            .map(Some)
            .ok_or_else(|| Error::invariant("host ticket completed twice"))
    }
    fn capture_execution_state(&self) -> Result<Option<Vec<u8>>> {
        let checkpoint = Checkpoint {
            identity: self.identity.clone(),
            sequences: self.sequences.clone(),
            tokens_executed: self.tokens_executed,
            prefix_hits: self.prefix_hits,
        };
        Ok(Some(
            serde_json::to_vec(&checkpoint).map_err(|e| Error::invalid(e.to_string()))?,
        ))
    }
    fn validate_state_ownership(&self, states: &[(StateId, usize, usize)]) -> Result<()> {
        if states.len() != self.sequences.len()
            || states.iter().any(|(id, capacity, position)| {
                !self
                    .sequences
                    .get(id)
                    .is_some_and(|s| s.capacity == *capacity && s.tokens.len() == *position)
            })
        {
            return Err(Error::invalid(
                "physical state ownership/cursor differs from runtime",
            ));
        }
        Ok(())
    }

    fn restore_execution_state(&mut self, payload: Option<&[u8]>) -> Result<()> {
        self.provider_restore_execution_state(payload)
    }
}
