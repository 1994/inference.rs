//! Setup responsibilities.
use super::{MetalBackend, loading};
use infer_core::{Error, ErrorCode, Result};

impl MetalBackend {
    ///
    /// # Errors
    /// Returns a capacity or backend error if fresh execution state cannot be allocated.
    pub fn fresh(&self) -> Result<Self> {
        if self.busy.is_some() {
            return Err(Error::new(
                ErrorCode::Conflict,
                "fork Metal executor after completion",
            ));
        }
        Self::assemble(
            self.model.clone(),
            self.graph.clone(),
            self.gpu.clone(),
            loading::ResidentWeights {
                buffers: self.weights.clone(),
                formats: self.weight_formats.clone(),
                plan: self.load_plan.clone(),
                identity: self.identity.clone(),
            },
            self.config.clone(),
        )
    }
}
