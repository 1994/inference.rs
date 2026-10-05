//! Protocol extension contract.
use infer_core::Result;
use infer_ir::{CanonicalRequest, WorkloadOutput};

pub trait ProtocolAdapter {
    ///
    /// # Errors
    /// Returns an invalid-input or unsupported error for malformed request bytes or an unsupported protocol.
    fn normalize(&self, bytes: &[u8]) -> Result<CanonicalRequest>;
    ///
    /// # Errors
    /// Returns an invalid-input or serialization error if workload output cannot be encoded by this protocol.
    fn encode(&self, output: &WorkloadOutput) -> Result<Vec<u8>>;
}
