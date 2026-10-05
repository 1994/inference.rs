//! Provider extension contract.
use super::SPI_VERSION;
use infer_core::{Error, ProviderId, Result};

#[derive(Debug, Clone)]
pub struct ProviderMetadata {
    pub id: ProviderId,
    pub name: String,
    pub spi_version: u32,
}
impl ProviderMetadata {
    ///
    /// # Errors
    /// Returns an unsupported error for a different SPI version or an invalid-input error for an empty provider name.
    pub fn validate(&self) -> Result<()> {
        if self.spi_version != SPI_VERSION {
            return Err(Error::unsupported("incompatible SPI version"));
        }
        if self.name.is_empty() {
            return Err(Error::invalid("empty provider name"));
        }
        Ok(())
    }
}
