//! Cost extension contract.
use infer_core::{Error, Result};
use infer_ir::{CostEstimate, CostModelInspection, CostObservation, CostQuery};

/// Fixed batch features for providers explicitly supporting exact summary estimation.
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchCostSummary {
    pub fallback: CostEstimate,
    pub tokens: usize,
    pub context_tokens: usize,
    pub batch: usize,
    /// Prefill/decode/forward membership bits are 1/2/4 respectively.
    pub roles: u8,
}
/// Estimates a complete candidate batch; device execution stays outside the planner.
///
/// Estimates must remain deterministic during one planning call. Arbitrary providers
/// retain complete queries; the summary interface requires explicit opt-in and exact parity.
pub trait CostModelProvider {
    fn identity(&self) -> &str;
    ///
    /// # Errors
    /// Returns an invalid-input or unsupported error if queries are inconsistent or cannot be estimated.
    fn estimate(&self, work: &[CostQuery]) -> Result<CostEstimate>;
    /// Opt in only when these batch features preserve the complete estimate exactly.
    fn supports_batch_summary(&self) -> bool {
        false
    }
    /// # Errors
    /// Rejects invalid summaries or estimation failures; default providers require full queries.
    fn estimate_summary(&self, _summary: &BatchCostSummary) -> Result<Option<CostEstimate>> {
        Ok(None)
    }
    ///
    /// # Errors
    /// Implementations may reject incompatible programs, backends, or timing sources; the default accepts observations.
    fn observe(&mut self, _observation: &CostObservation) -> Result<()> {
        Ok(())
    }
    ///
    /// # Errors
    /// Implementations may return a serialization or backend error when calibration cannot be captured.
    fn capture_state(&self) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }
    ///
    /// # Errors
    /// Returns an unsupported or invalid-input error if the captured state cannot be restored by this provider.
    fn restore_state(&mut self, data: Option<&[u8]>) -> Result<()> {
        if data.is_some() {
            return Err(Error::unsupported(
                "cost provider cannot restore calibration",
            ));
        }
        Ok(())
    }
    fn inspect(&self) -> CostModelInspection {
        CostModelInspection {
            provider: self.identity().into(),
            adaptive: false,
            profiles: 0,
            observations: 0,
            evictions: 0,
            last_source: None,
        }
    }
}
