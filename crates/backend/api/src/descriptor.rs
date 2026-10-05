//! Descriptor contract.
use super::{MAX_SUBMISSION_BATCH, SUBMISSION_ABI_VERSION};
use infer_core::{Error, Result};
use infer_ir::StepPlan;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct WorkDescriptor {
    pub request_id: u64,
    pub state_id: u64,
    pub token_count: u32,
    pub computed_frontier: u32,
    pub role: u32,
    pub readout: u32,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct SubmissionDescriptor {
    pub abi_version: u32,
    pub descriptor_bytes: u32,
    pub owner: u64,
    pub generation: u32,
    pub reserved: u32,
    pub step_id: u64,
    pub program_id: u64,
    pub metadata_slot: u32,
    pub work_count: u32,
    pub work: [WorkDescriptor; MAX_SUBMISSION_BATCH],
}
impl SubmissionDescriptor {
    ///
    /// # Errors
    /// Returns an invalid-input error if token counts or descriptor fields exceed the device ABI ranges.
    pub fn from_plan(plan: &StepPlan, metadata_slot: u32) -> Result<Self> {
        if plan.work.is_empty() || plan.work.len() > MAX_SUBMISSION_BATCH {
            return Err(Error::invalid(
                "submission batch exceeds fixed descriptor capacity",
            ));
        }
        let mut descriptor = Self {
            abi_version: SUBMISSION_ABI_VERSION,
            descriptor_bytes: u32::try_from(size_of::<Self>())
                .map_err(|_| Error::invariant("descriptor layout too large"))?,
            owner: 0,
            generation: 0,
            reserved: 0,
            step_id: plan.id.get(),
            program_id: plan.program.get(),
            metadata_slot,
            work_count: u32::try_from(plan.work.len())
                .map_err(|_| Error::invalid("batch count exceeds device ABI"))?,
            work: [WorkDescriptor::default(); MAX_SUBMISSION_BATCH],
        };
        for (slot, work) in descriptor.work.iter_mut().zip(&plan.work) {
            if work.token_count == 0 {
                return Err(Error::invalid("empty submission work"));
            }
            *slot = WorkDescriptor {
                request_id: work.request.get(),
                state_id: work.state.get(),
                token_count: u32::try_from(work.token_count)
                    .map_err(|_| Error::invalid("token count exceeds device ABI"))?,
                computed_frontier: 0,
                role: match work.role {
                    infer_ir::ExecutionRole::Prefill => 0,
                    infer_ir::ExecutionRole::Decode => 1,
                    infer_ir::ExecutionRole::Forward => 2,
                    infer_ir::ExecutionRole::Mixed => 3,
                },
                readout: 0,
            };
        }
        Ok(descriptor)
    }
}
