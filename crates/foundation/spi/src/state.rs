//! State extension contract.
use infer_core::{AllocationId, Error, Result, StateId};
use infer_ir::{ModelIr, StateRequirement};

pub trait SequenceStateProvider {
    ///
    /// # Errors
    /// Returns an invalid-input or unsupported error if the model has no valid state layout.
    fn requirements(&self, model: &ModelIr) -> Result<Vec<StateRequirement>>;
}
pub trait StateStorageProvider {
    ///
    /// # Errors
    /// Returns an invalid-input or capacity error if device storage cannot be reserved.
    fn reserve(&mut self, bytes: u64) -> Result<AllocationId>;
    ///
    /// # Errors
    /// Returns a not-found or invariant error for unknown allocations or inconsistent ownership.
    fn release(&mut self, allocation: AllocationId) -> Result<()>;
}
#[derive(Debug, Clone)]
pub struct TransferTicket {
    pub state: StateId,
    pub endpoint: String,
    pub opaque: Vec<u8>,
}
pub trait StateTransferProvider {
    ///
    /// # Errors
    /// Returns an unsupported, not-found, or backend error if the state cannot be exported to this endpoint.
    fn export(&self, state: StateId, endpoint: &str) -> Result<TransferTicket>;
    ///
    /// # Errors
    /// Returns an unsupported, invalid-input, or backend error if the transfer ticket cannot be imported.
    fn import(&self, ticket: &TransferTicket) -> Result<StateId>;
}
pub struct LocalOnlyTransfer;
impl StateTransferProvider for LocalOnlyTransfer {
    fn export(&self, state: StateId, endpoint: &str) -> Result<TransferTicket> {
        if endpoint != "local" {
            return Err(Error::unsupported("remote state transfer is not installed"));
        }
        Ok(TransferTicket {
            state,
            endpoint: endpoint.into(),
            opaque: Vec::new(),
        })
    }
    fn import(&self, ticket: &TransferTicket) -> Result<StateId> {
        if ticket.endpoint != "local" || !ticket.opaque.is_empty() {
            return Err(Error::unsupported("not a local state ticket"));
        }
        Ok(ticket.state)
    }
}
