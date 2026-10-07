use crate::device::CudaDevice;
use cuda_core::sys::{
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as MAJOR,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as MINOR,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as SMS,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_WARP_SIZE as WARP,
};
use infer_core::{DeviceId, Result};
use infer_ir::{
    DeviceBackend, DeviceCapabilities, NvidiaArchitecture, NvidiaCapabilities,
    SpeculationCapability,
};

/// Compute capability major of Hopper, the first architecture with fourth-generation tensor cores.
const HOPPER_SM_MAJOR: u32 = 9;
/// Compute capability major of consumer Blackwell, with fifth-generation tensor cores.
const BLACKWELL_SM_MAJOR: u32 = 12;
/// Tensor core generation reported for Hopper.
const HOPPER_TENSOR_CORE_GENERATION: u16 = 4;
/// Tensor core generation reported for Blackwell.
const BLACKWELL_TENSOR_CORE_GENERATION: u16 = 5;

pub(super) fn query(device: &CudaDevice) -> Result<DeviceCapabilities> {
    let major = device.attribute(MAJOR)?;
    let minor = device.attribute(MINOR)?;
    Ok(DeviceCapabilities {
        backend: DeviceBackend::Cuda(NvidiaCapabilities {
            architecture: NvidiaArchitecture {
                compute_major: major,
                compute_minor: minor,
                multiprocessors: device.attribute(SMS)?,
            },
            tensor_core_generation: match (major, minor) {
                (HOPPER_SM_MAJOR, 0) => Some(HOPPER_TENSOR_CORE_GENERATION),
                (BLACKWELL_SM_MAJOR, 0) => Some(BLACKWELL_TENSOR_CORE_GENERATION),
                _ => None,
            },
            warp_size: device.attribute(WARP)?,
            graphs: true,
            // Advertise only features exposed by this provider, not untested hardware paths.
            tma: false,
            clusters: false,
            pinned_transfer: false,
            cuda_ipc: false,
            nvlink: false,
            gpu_direct: false,
        }),
        device: DeviceId::new(device.stream.device().ordinal() as u64 + 1)?,
        compute_dtypes: device.target().compute_dtypes().to_vec(),
        memory_bytes: device.memory_info()?.1,
        unified_memory: false,
        profiling: false,

        speculation: SpeculationCapability::default(),
    })
}
