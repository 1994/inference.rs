//! Protocol HTTP adapter.
use infer_core::{Error, Result};
use infer_ir::{CanonicalRequest, WorkloadOutput};
use infer_spi::ProtocolAdapter;

pub struct NativeJsonAdapter;
impl ProtocolAdapter for NativeJsonAdapter {
    fn normalize(&self, bytes: &[u8]) -> Result<CanonicalRequest> {
        let request: CanonicalRequest =
            serde_json::from_slice(bytes).map_err(|e| Error::invalid(e.to_string()))?;
        request.validate()?;
        Ok(request)
    }
    fn encode(&self, output: &WorkloadOutput) -> Result<Vec<u8>> {
        serde_json::to_vec(output).map_err(|e| Error::invalid(e.to_string()))
    }
}
