//! Safe CUDA memory and kernel launch boundary.
use cutile::prelude::*;
use infer_core::{Error as InferError, ErrorCode, Result};
use serde::Serialize;
use std::sync::{Arc, OnceLock};

/// One MiB in bytes, for human-readable identities.
const MIB: u64 = 1024 * 1024;
/// Device memory reserved for driver and module bookkeeping, as a fraction of total memory.
const HEADROOM_DIVISOR: u64 = 32;
/// Lower bound of that reservation, for small devices.
const MIN_DEVICE_HEADROOM: u64 = 256 * MIB;
/// Upper bound of that reservation, so huge devices keep state space usable.
const MAX_DEVICE_HEADROOM: u64 = 1024 * MIB;
/// Hertz per kilohertz, for the driver's kHz clock attributes.
const HERTZ_PER_KILOHERTZ: u64 = 1000;
/// Bits per byte, for the driver's bit-width attributes.
const BITS_PER_BYTE: u32 = 8;
/// Double data rate: two transfers per memory clock.
const TRANSFERS_PER_CLOCK: u64 = 2;

/// Hardware facts queried from the driver at open time, plus the policy values derived from them.
///
/// Nothing here is maintained per GPU model: every field is read from the device, so a new card
/// needs no code or table change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceProfile {
    /// Driver-reported device name.
    pub name: String,
    /// Compute capability as `sm_<major><minor>`.
    pub architecture: String,
    /// Streaming multiprocessors.
    pub multiprocessors: u32,
    /// Total device memory.
    pub total_memory_bytes: u64,
    /// Memory clock in kHz.
    pub memory_clock_khz: u32,
    /// Global memory bus width in bits.
    pub memory_bus_width_bits: u32,
    /// L2 cache size in bytes.
    pub l2_cache_bytes: u32,
}
impl DeviceProfile {
    /// Stable identity for machine-local artifacts such as the kernel tuning table. Two
    /// different devices cannot share name, architecture and memory size at once.
    #[must_use]
    pub fn identity(&self) -> String {
        format!(
            "{}|{}|{}MiB",
            self.name,
            self.architecture,
            self.total_memory_bytes / MIB
        )
    }
    /// Theoretical DRAM bandwidth, the roofline denominator for bandwidth-bound decode steps.
    #[must_use]
    pub fn memory_bandwidth_bytes_per_second(&self) -> u64 {
        memory_bandwidth(self.memory_clock_khz, self.memory_bus_width_bits)
    }
    /// Driver and module slack kept out of the request-state budget.
    #[must_use]
    pub fn device_headroom_bytes(&self) -> u64 {
        device_headroom(self.total_memory_bytes)
    }
    /// Graph capture and allocator slack counted inside a sequence budget.
    #[must_use]
    pub fn graph_headroom_bytes(&self) -> u64 {
        graph_headroom(self.total_memory_bytes)
    }
    /// Per-request activation arena and projection workspace budget.
    #[must_use]
    pub fn arena_budget_bytes(&self) -> u64 {
        arena_budget(self.total_memory_bytes)
    }
    /// Verification checkpoint budget retained across speculation lanes.
    #[must_use]
    pub fn checkpoint_budget_bytes(&self) -> u64 {
        checkpoint_budget(self.total_memory_bytes)
    }
    /// Resident sequences this device is expected to admit before the byte budget rejects more.
    #[must_use]
    pub fn resident_states(&self) -> usize {
        resident_states(self.total_memory_bytes)
    }
}

/// Double-data-rate bandwidth from the memory clock and bus width.
fn memory_bandwidth(clock_khz: u32, bus_width_bits: u32) -> u64 {
    u64::from(clock_khz)
        * HERTZ_PER_KILOHERTZ
        * u64::from(bus_width_bits / BITS_PER_BYTE)
        * TRANSFERS_PER_CLOCK
}

/// Fractional headroom clamped so small devices keep a floor and large ones stay bounded.
fn device_headroom(total_memory_bytes: u64) -> u64 {
    (total_memory_bytes / HEADROOM_DIVISOR).clamp(MIN_DEVICE_HEADROOM, MAX_DEVICE_HEADROOM)
}

/// Graph and allocator slack reserved inside every sequence budget.
const GRAPH_HEADROOM_DIVISOR: u64 = 128;
/// Lower bound of that reservation.
const MIN_GRAPH_HEADROOM: u64 = 64 * MIB;
/// Upper bound of that reservation.
const MAX_GRAPH_HEADROOM: u64 = 512 * MIB;
/// Per-request activation arena budget.
const ARENA_DIVISOR: u64 = 64;
/// Lower bound of the activation arena budget.
const MIN_ARENA_BUDGET: u64 = 128 * MIB;
/// Upper bound of the activation arena budget.
const MAX_ARENA_BUDGET: u64 = 1024 * MIB;
/// Retained verification checkpoints across all speculation lanes.
const CHECKPOINT_DIVISOR: u64 = 16;
/// Lower bound of the checkpoint budget.
const MIN_CHECKPOINT_BUDGET: u64 = 512 * MIB;
/// Upper bound of the checkpoint budget.
const MAX_CHECKPOINT_BUDGET: u64 = 4 * 1024 * MIB;
/// Device memory charged per reserved sequence when bounding concurrency.
const STATE_BYTES_PER_SLOT: u64 = 512 * MIB;
/// Concurrency bounds applied after dividing device memory by the per-slot charge.
const MIN_RESIDENT_STATES: usize = 4;
/// Upper bound on resident sequences regardless of device size.
const MAX_RESIDENT_STATES: usize = 64;

/// Fraction of device memory reserved for graph capture and allocator bookkeeping.
fn graph_headroom(total_memory_bytes: u64) -> u64 {
    (total_memory_bytes / GRAPH_HEADROOM_DIVISOR).clamp(MIN_GRAPH_HEADROOM, MAX_GRAPH_HEADROOM)
}

/// Per-request activation arena and projection workspace budget.
fn arena_budget(total_memory_bytes: u64) -> u64 {
    (total_memory_bytes / ARENA_DIVISOR).clamp(MIN_ARENA_BUDGET, MAX_ARENA_BUDGET)
}

/// Retained verification checkpoint budget.
fn checkpoint_budget(total_memory_bytes: u64) -> u64 {
    (total_memory_bytes / CHECKPOINT_DIVISOR).clamp(MIN_CHECKPOINT_BUDGET, MAX_CHECKPOINT_BUDGET)
}

/// Resident state slots a device of this size can be expected to admit.
fn resident_states(total_memory_bytes: u64) -> usize {
    usize::try_from(total_memory_bytes / STATE_BYTES_PER_SLOT)
        .unwrap_or(MAX_RESIDENT_STATES)
        .clamp(MIN_RESIDENT_STATES, MAX_RESIDENT_STATES)
}
mod readback;
// Exercised by its own device test today; the projection path adopts it in a following change.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "adopted by the projection path in a following change"
    )
)]
pub(crate) mod cublaslt;
pub(crate) use readback::{ReadbackSource, Readbacks};

#[derive(Clone)]
pub struct CudaDevice {
    pub(crate) stream: Arc<cuda_core::Stream>,
    target: crate::target::CudaTarget,
    profile: Arc<OnceLock<DeviceProfile>>,
}

pub(crate) fn device_error(error: impl std::fmt::Display) -> InferError {
    InferError::new(ErrorCode::Backend, error.to_string())
}

impl CudaDevice {
    /// Enable cuTile's persistent compiled-kernel cache for subsequent captures.
    /// # Errors
    /// Returns cache initialization errors.
    pub fn enable_kernel_cache() -> Result<()> {
        cutile::jit_cache::enable_default().map_err(device_error)
    }
    /// # Errors
    /// Fails when the requested CUDA device or stream is unavailable.
    pub fn new(ordinal: usize) -> Result<Self> {
        let device = Device::new(ordinal).map_err(device_error)?;
        let target = query_target(&device)?;
        Ok(Self {
            stream: device.new_stream().map_err(device_error)?,
            target,
            profile: Arc::new(OnceLock::new()),
        })
    }

    /// # Errors
    /// Returns a CUDA error when the device name cannot be queried.
    pub fn name(&self) -> Result<String> {
        self.stream.device().name().map_err(device_error)
    }

    #[must_use]
    pub const fn target(&self) -> &crate::target::CudaTarget {
        &self.target
    }

    /// # Errors
    /// Returns a CUDA error if upload or reshape fails.
    pub fn upload<T: DType>(&self, values: Vec<T>, shape: &[usize]) -> Result<Arc<Tensor<T>>> {
        api::copy_host_vec_to_device(&Arc::new(values))
            .sync_on(&self.stream)
            .map_err(device_error)?
            .reshape(shape)
            .map(Arc::new)
            .map_err(device_error)
    }

    /// # Errors
    /// Returns a shape or device error for an invalid matrix-vector product.
    pub fn matvec<T: DType>(
        &self,
        x: Arc<Tensor<f32>>,
        w: Arc<Tensor<T>>,
    ) -> Result<Arc<Tensor<f32>>> {
        self.matvec_tiled(
            x,
            w,
            crate::strategy::LinearTiling::new(
                crate::constants::MATVEC_TILE_ROWS,
                crate::constants::MATVEC_TILE_COLUMNS,
            )?,
        )
    }

    /// # Errors
    /// Returns a shape or device error for an invalid projection.
    pub fn matvec_tiled<T: DType>(
        &self,
        x: Arc<Tensor<f32>>,
        w: Arc<Tensor<T>>,
        tiling: crate::strategy::LinearTiling,
    ) -> Result<Arc<Tensor<f32>>> {
        let (rows, columns) = dimensions(&x, &w)?;
        let block = columns.min(tiling.columns()).next_power_of_two();
        let out = api::zeros::<f32>(&[rows]).partition([tiling.rows()]);
        crate::kernels::linear::dense(out, x, w)
            .generics(vec![
                T::DTYPE.as_str().into(),
                tiling.rows().to_string(),
                block.to_string(),
                columns.to_string(),
            ])
            .first()
            .unpartition()
            .sync_on(&self.stream)
            .map(Arc::new)
            .map_err(device_error)
    }

    /// # Errors
    /// Returns a CUDA transfer error.
    pub fn read<T: DType>(&self, tensor: &Arc<Tensor<T>>) -> Result<Vec<T>> {
        tensor
            .to_host_vec()
            .sync_on(&self.stream)
            .map_err(device_error)
    }

    /// Copy an f32 tensor back without taking ownership, so a borrowed snapshot can be inspected.
    ///
    /// # Errors
    /// Returns a backend error when the device is unbound or the copy fails.
    #[expect(
        unsafe_code,
        reason = "Audited device-to-host copy from a live tensor of known length into a live host slice"
    )]
    pub fn read_borrowed(&self, tensor: &Tensor<f32>) -> Result<Vec<f32>> {
        self.reclaim_barrier()?;
        let mut host = vec![0.0_f32; tensor.size()];
        let bytes = tensor
            .size()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| device_error("readback size overflow"))?;
        // SAFETY: the device context is current, the tensor owns `size()` f32 elements and the
        // destination is a live host slice of the same length.
        let status = unsafe {
            cuda_core::sys::cuMemcpyDtoH_v2(
                host.as_mut_ptr().cast(),
                tensor.device_pointer().cu_deviceptr(),
                bytes,
            )
        };
        driver_status(status)?;
        Ok(host)
    }
}

#[expect(
    unsafe_code,
    reason = "Audited read-only CUDA attribute query; the live owned Device supplies a valid driver handle"
)]
fn query_target(device: &Device) -> Result<crate::target::CudaTarget> {
    // SAFETY: Device::new initialized CUDA and owns this live device handle.
    let name =
        unsafe { cuda_core::get_device_sm_name(device.cu_device()) }.map_err(device_error)?;
    Ok(crate::target::CudaTarget::from_sm_name(&name))
}

fn dimensions<T: DType>(x: &Tensor<f32>, w: &Tensor<T>) -> Result<(usize, usize)> {
    if w.shape().len() != 2 || x.shape().len() != 1 || x.shape()[0] != w.shape()[1] {
        return Err(InferError::invalid("CUDA matrix-vector dimensions"));
    }
    let rows = usize::try_from(w.shape()[0]).map_err(|_| InferError::invalid("CUDA rows"))?;
    let columns = usize::try_from(w.shape()[1]).map_err(|_| InferError::invalid("CUDA columns"))?;
    if rows == 0 || columns == 0 {
        return Err(InferError::invalid("empty CUDA matrix"));
    }
    Ok((rows, columns))
}

#[expect(
    unsafe_code,
    reason = "Audited driver queries and stream drain with an owned device and a bound context"
)]
impl CudaDevice {
    pub(crate) fn attribute(&self, attribute: cuda_core::sys::CUdevice_attribute) -> Result<u32> {
        let mut value = 0;
        // SAFETY: the owned stream retains this device; value is a live output pointer.
        let status = unsafe {
            cuda_core::sys::cuDeviceGetAttribute(
                &raw mut value,
                attribute,
                self.stream.device().cu_device(),
            )
        };
        driver_status(status)?;
        u32::try_from(value).map_err(device_error)
    }

    /// Cached hardware facts and derived policy for this device.
    /// # Errors
    /// Returns a driver error if any device attribute or memory size cannot be queried.
    pub fn profile(&self) -> Result<&DeviceProfile> {
        if let Some(profile) = self.profile.get() {
            return Ok(profile);
        }
        let queried = self.query_profile()?;
        let _ = self.profile.set(queried);
        self.profile
            .get()
            .ok_or_else(|| device_error("device profile cache"))
    }

    /// Query the driver for hardware facts and derived policy.
    /// # Errors
    /// Returns a driver error if any device attribute or memory size cannot be queried.
    fn query_profile(&self) -> Result<DeviceProfile> {
        use cuda_core::sys::{
            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as MAJOR,
            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as MINOR,
            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH as BUS,
            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE as L2,
            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE as MEMORY_CLOCK,
            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as SMS,
        };
        Ok(DeviceProfile {
            name: self.name()?,
            architecture: format!("sm_{}{}", self.attribute(MAJOR)?, self.attribute(MINOR)?),
            multiprocessors: self.attribute(SMS)?,
            total_memory_bytes: self.memory_info()?.1,
            memory_clock_khz: self.attribute(MEMORY_CLOCK)?,
            memory_bus_width_bits: self.attribute(BUS)?,
            l2_cache_bytes: self.attribute(L2)?,
        })
    }

    /// Copy `elements` values between resident tensors of one dtype on this device's stream.
    ///
    /// Prefix caching snapshots and restores KV state with this: one stream-ordered device copy,
    /// no host round trip.
    /// # Errors
    /// Returns a driver error when the copy cannot be enqueued.
    pub fn copy_d2d<T: DType>(
        &self,
        dst: &mut Tensor<T>,
        src: &Tensor<T>,
        elements: usize,
    ) -> Result<()> {
        self.copy_d2d_at(dst, src, 0, 0, elements)
    }

    /// Ranged variant: snapshots copy only the token rows a prefix actually covers.
    /// # Errors
    /// Rejects ranges outside either tensor or a driver failure when the copy cannot be enqueued.
    pub fn copy_d2d_at<T: DType>(
        &self,
        dst: &mut Tensor<T>,
        src: &Tensor<T>,
        dst_offset: usize,
        src_offset: usize,
        elements: usize,
    ) -> Result<()> {
        let fits = |tensor: &Tensor<T>, offset: usize| {
            offset
                .checked_add(elements)
                .is_some_and(|end| end <= tensor.size())
        };
        if !fits(dst, dst_offset) || !fits(src, src_offset) {
            return Err(InferError::invalid("device copy range outside tensor"));
        }
        let bytes = elements
            .checked_mul(size_of::<T>())
            .ok_or_else(|| InferError::invalid("device copy size overflow"))?;
        let offset_bytes = |offset: usize| {
            offset
                .checked_mul(size_of::<T>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| InferError::invalid("device copy offset overflow"))
        };
        let dst_bytes = offset_bytes(dst_offset)?;
        let src_bytes = offset_bytes(src_offset)?;
        let stream = self.stream.cu_stream();
        // SAFETY: the checked ranges keep both accesses inside live device allocations, and the
        // stream belongs to this device; callers drain it before reusing the buffers.
        let status = unsafe {
            cuda_core::sys::cuMemcpyDtoDAsync_v2(
                dst.device_pointer().cu_deviceptr() + dst_bytes,
                src.device_pointer().cu_deviceptr() + src_bytes,
                bytes,
                stream,
            )
        };
        driver_status(status)
    }

    /// # Errors
    /// Returns a driver error if available device memory cannot be queried.
    pub fn memory_info(&self) -> Result<(u64, u64)> {
        self.stream
            .device()
            .bind_to_thread()
            .map_err(device_error)?;
        let (mut free, mut total) = (0, 0);
        // SAFETY: the owned device's context is current and both outputs are live.
        let status = unsafe { cuda_core::sys::cuMemGetInfo_v2(&raw mut free, &raw mut total) };
        driver_status(status)?;
        Ok((free as u64, total as u64))
    }

    /// Device bytes the allocator can hand out again without asking the driver.
    ///
    /// Tensors are allocated with `cuMemAllocAsync` from the device's memory pool, and a freed
    /// block stays reserved in that pool. `cuMemGetInfo` counts reserved-but-unused blocks as
    /// used, so a pre-flight admission check against it under-reports what a retry can actually
    /// obtain; this is the difference.
    pub(crate) fn pool_reclaimable_bytes(&self) -> Result<u64> {
        self.stream
            .device()
            .bind_to_thread()
            .map_err(device_error)?;
        let pool = self
            .stream
            .device()
            .default_mem_pool()
            .map_err(device_error)?;
        let stats = pool.mem_stats().map_err(device_error)?;
        Ok(stats.reserved_current.saturating_sub(stats.used_current))
    }

    // cuda-async frees tensors on a separate deallocator stream. Join it before
    // using driver free-memory queries to admit allocations after cache eviction.
    pub(crate) fn reclaim_barrier(&self) -> Result<()> {
        self.stream
            .device()
            .bind_to_thread()
            .map_err(device_error)?;
        // Tensors come from the device memory pool, and a block returned to the pool keeps
        // counting as used in `cuMemGetInfo` until the driver trims it. Neither raising the
        // release threshold nor `cuMemPoolTrimTo` (unbound in cuda-bindings 0.4) gets the memory
        // back in time, so a failed slot-pool attempt is only visible through
        // `pool_reclaimable_bytes`; nothing here pretends to make the query honest.
        // SAFETY: the owned device's context is current; this only waits for its work.
        unsafe { self.stream.device().synchronize() }.map_err(device_error)
    }

    pub(crate) fn drain(&self) -> Result<()> {
        self.stream
            .device()
            .bind_to_thread()
            .map_err(device_error)?;
        // SAFETY: the owned stream remains live and its parent context is current.
        unsafe { self.stream.synchronize() }.map_err(device_error)
    }
}

fn driver_status(status: cuda_core::sys::CUresult) -> Result<()> {
    if status == cuda_core::sys::cudaError_enum_CUDA_SUCCESS {
        Ok(())
    } else {
        Err(device_error(format!("CUDA driver status {status}")))
    }
}

/// Discard only the host return value of a graph-safe operation. Resource leases
/// remain owned by its execution context until the enclosing replay completes.
pub(crate) struct GraphUpdate<N>(pub N);

#[expect(
    unsafe_code,
    reason = "Audited forwarding of the caller's live execution context to a graph-safe operation"
)]
impl<N: GraphNode> DeviceOp for GraphUpdate<N> {
    type Output = ();

    unsafe fn execute(self, context: &ExecutionContext) -> std::result::Result<(), DeviceError> {
        // SAFETY: preserve the caller's context and submission lifetime unchanged.
        unsafe { self.0.execute(context) }.map(|_| ())
    }
}

impl<N: GraphNode> GraphNode for GraphUpdate<N> {}

impl<N: GraphNode> IntoFuture for GraphUpdate<N> {
    type Output = std::result::Result<(), DeviceError>;
    type IntoFuture = cutile::cuda_async::device_future::DeviceFuture<(), Self>;

    fn into_future(self) -> Self::IntoFuture {
        match cutile::cuda_async::device_context::with_default_device_policy(|policy| {
            policy.next_stream()
        }) {
            Ok(Ok(stream)) => Self::IntoFuture::scheduled(self, ExecutionContext::new(stream)),
            Ok(Err(error)) | Err(error) => Self::IntoFuture::failed(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(total_memory_bytes: u64) -> DeviceProfile {
        DeviceProfile {
            name: "NVIDIA GeForce RTX 5090".to_owned(),
            architecture: "sm_120".to_owned(),
            multiprocessors: 170,
            total_memory_bytes,
            memory_clock_khz: 14_001_000,
            memory_bus_width_bits: 512,
            l2_cache_bytes: 128 * 1024 * 1024,
        }
    }

    #[test]
    fn profile_derives_bandwidth_headroom_and_identity() {
        let device = profile(32 * 1024 * MIB);
        // 14.001 GHz memory clock, double data rate, 512-bit bus: 1.792 TB/s.
        assert_eq!(
            device.memory_bandwidth_bytes_per_second(),
            1_792_128_000_000
        );
        // A 32 GiB device keeps exactly 1 GiB of driver headroom.
        assert_eq!(device.device_headroom_bytes(), 1024 * MIB);
        // Small devices keep the floor, huge ones stay bounded.
        assert_eq!(device_headroom(4 * 1024 * MIB), MIN_DEVICE_HEADROOM);
        assert_eq!(device_headroom(1024 * 1024 * MIB), MAX_DEVICE_HEADROOM);
        assert_eq!(device.identity(), "NVIDIA GeForce RTX 5090|sm_120|32768MiB");
        // Identities of different devices never collide.
        assert_ne!(device.identity(), profile(24 * 1024 * MIB).identity());
    }
}
