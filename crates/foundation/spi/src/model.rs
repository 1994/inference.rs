//! Model extension contract.
use super::ProviderMetadata;
use infer_core::{ModelId, Result};
use infer_ir::ModelIr;

pub trait ModelProvider {
    fn metadata(&self) -> ProviderMetadata;
    ///
    /// # Errors
    /// Returns an invalid-input or unsupported error for malformed configuration or an unsupported model.
    fn import(&self, id: ModelId, config: &[u8]) -> Result<ModelIr>;
}
