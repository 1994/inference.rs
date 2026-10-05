//! Kernel extension contract.
use infer_core::KernelId;
use infer_ir::{BackendKind, CapabilityRequirements, Operation, PrecisionPlan};

pub trait KernelProvider {
    fn kernels(&self) -> Vec<KernelRegistration>;
}
#[derive(Debug, Clone)]
pub struct KernelRegistration {
    pub backend: BackendKind,
    pub id: KernelId,
    pub operation: Operation,
    pub precision: PrecisionPlan,
    pub requirements: CapabilityRequirements,
    pub max_shape_elements: usize,
    pub workspace_bytes: u64,
    pub priority: i32,
    pub estimated_ns: u64,
    pub source: SourceLocation,
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SourceLocation {
    pub crate_name: String,
    pub file: String,
    pub function: String,
}
