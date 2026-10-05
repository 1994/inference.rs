//! Extensions extension contract.
use infer_core::{AllocationId, Result};
use infer_ir::{
    CanonicalRequest, DeviceCapabilities, FeaturePlan, MediaInput, Modality, ModelIr,
    MultimodalGraph, PrecisionPlan, SpeculationPlan,
};

pub struct PreparedMedia {
    pub input: MediaInput,
    pub tokens: usize,
    pub embedding: AllocationId,
}
pub trait ModalityProvider {
    fn supports(&self, modality: &Modality) -> bool;
    ///
    /// # Errors
    /// Returns an invalid-input or unsupported error if the media cannot be represented by this provider.
    fn graph(&self, input: &MediaInput) -> Result<MultimodalGraph>;
}
pub trait FusionProvider {
    ///
    /// # Errors
    /// Returns an invalid-input or unsupported error if the request, model, or backend cannot produce a valid plan.
    fn plan(&self, media: &[PreparedMedia], model: &ModelIr) -> Result<MultimodalGraph>;
}
pub trait FeatureProvider {
    ///
    /// # Errors
    /// Returns an invalid-input or unsupported error if the request, model, or backend cannot produce a valid plan.
    fn plan(
        &self,
        request: &CanonicalRequest,
        model: &ModelIr,
        features: &mut FeaturePlan,
    ) -> Result<()>;
}
pub trait PrecisionProvider {
    ///
    /// # Errors
    /// Returns an invalid-input or unsupported error if the request, model, or backend cannot produce a valid plan.
    fn plan(&self, model: &ModelIr, capabilities: &DeviceCapabilities) -> Result<PrecisionPlan>;
}
pub trait SpeculationProvider {
    ///
    /// # Errors
    /// Returns an invalid-input or unsupported error if the request, model, or backend cannot produce a valid plan.
    fn plan(&self, model: &ModelIr, capabilities: &DeviceCapabilities) -> Result<SpeculationPlan>;
}
