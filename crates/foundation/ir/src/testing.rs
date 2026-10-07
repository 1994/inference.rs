use crate::{
    DType,
    hardware::{DeviceBackend, DeviceCapabilities, SpeculationCapability},
};
use infer_core::DeviceId;

#[must_use]
pub fn reference_capabilities() -> DeviceCapabilities {
    DeviceCapabilities {
        backend: DeviceBackend::TestCpu,
        device: DeviceId::ONE,
        compute_dtypes: vec![DType::F32],
        memory_bytes: 0,
        unified_memory: true,
        profiling: false,
        speculation: SpeculationCapability::default(),
    }
}
