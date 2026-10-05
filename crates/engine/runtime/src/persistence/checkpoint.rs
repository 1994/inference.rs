//! Quiescent checkpoints use the same nonblocking owner protocol as resource commands.
use crate::{Engine, RuntimeSnapshot};
use infer_core::{Error, ErrorCode, Result};
use infer_spi::{
    BackendProvider, ResourceCommand, ResourceReply, ResourceTicket, SchedulingPolicy,
};

pub struct CheckpointFlight {
    ticket: ResourceTicket,
    progress: u64,
    resource: u64,
    cost: u64,
}
impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    pub(crate) fn poll_checkpoint(&mut self) -> Result<Option<RuntimeSnapshot>> {
        if !self.backend.supports_control_checkpoint() {
            return Err(Error::unsupported("backend does not support checkpoints"));
        }
        let Some(mut flight) = self.host.checkpoint.take() else {
            if self.backend.pending_resource_releases() {
                return Ok(None);
            }
            self.check_invariants()?;
            let states = self.checkpoint_states()?;
            let ticket = match self
                .backend
                .begin_resource(ResourceCommand::Checkpoint { states })
            {
                Ok(ticket) => ticket,
                Err(error) if error.code == ErrorCode::Capacity => return Ok(None),
                Err(error) => return Err(error),
            };
            self.host.checkpoint = Some(CheckpointFlight {
                ticket,
                progress: self.global_progress_epoch,
                resource: self.resource_epoch,
                cost: self.host.cost_epoch,
            });
            return Ok(None);
        };
        let Some(reply) = flight.ticket.poll()? else {
            self.host.checkpoint = Some(flight);
            return Ok(None);
        };
        if flight.progress != self.global_progress_epoch
            || flight.resource != self.resource_epoch
            || flight.cost != self.host.cost_epoch
        {
            // Admission/cancel/deadline or resource changes invalidate a captured device frontier.
            return Ok(None);
        }
        let ResourceReply::Checkpoint(execution) = reply else {
            return Err(Error::invariant(
                "checkpoint acknowledgement has wrong phase",
            ));
        };
        self.check_invariants()?;
        let mut snapshot = self.snapshot_for_diagnostic();
        snapshot.execution_state = execution;
        snapshot.cost_state = self.costs.capture_state()?;
        Ok(Some(snapshot))
    }
    pub(crate) fn checkpoint_states(&self) -> Result<Vec<(infer_core::StateId, usize, usize)>> {
        self.host
            .requests
            .values()
            .filter(|record| !record.status.terminal())
            .map(|record| {
                let id = record
                    .state
                    .ok_or_else(|| Error::invariant("checkpoint owner missing state"))?;
                let state = self.state.get(id)?;
                Ok((id, state.capacity_tokens, state.committed_tokens))
            })
            .collect()
    }
}
