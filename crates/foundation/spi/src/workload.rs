//! Workload extension contract.
use infer_core::Result;
use infer_ir::{CanonicalRequest, ModelIr, ModelOutput, Workload, WorkloadOutput, WorkloadPlan};

pub trait WorkloadProvider {
    fn fork(&self) -> Option<Box<dyn WorkloadProvider + Send + Sync>> {
        None
    }
    fn identity(&self) -> &str {
        std::any::type_name::<Self>()
    }
    fn supports(&self, workload: &Workload) -> bool;
    ///
    /// # Errors
    /// Returns an invalid-input or unsupported error if the request, model, or backend cannot produce a valid plan.
    fn plan(
        &self,
        request: &CanonicalRequest,
        model: &ModelIr,
        program: infer_core::ProgramId,
    ) -> Result<WorkloadPlan>;
    ///
    /// # Errors
    /// Returns an invalid-input or unsupported error if outputs are incomplete, incompatible, or cannot be mapped to the requested workload.
    fn postprocess(
        &self,
        request: &CanonicalRequest,
        outputs: &[ModelOutput],
    ) -> Result<WorkloadOutput>;
}
impl WorkloadProvider for Box<dyn WorkloadProvider + Send + Sync> {
    fn identity(&self) -> &str {
        self.as_ref().identity()
    }
    fn fork(&self) -> Option<Box<dyn WorkloadProvider + Send + Sync>> {
        self.as_ref().fork()
    }
    fn supports(&self, w: &Workload) -> bool {
        self.as_ref().supports(w)
    }
    fn plan(
        &self,
        r: &CanonicalRequest,
        m: &ModelIr,
        p: infer_core::ProgramId,
    ) -> Result<WorkloadPlan> {
        self.as_ref().plan(r, m, p)
    }
    fn postprocess(&self, r: &CanonicalRequest, o: &[ModelOutput]) -> Result<WorkloadOutput> {
        self.as_ref().postprocess(r, o)
    }
}
