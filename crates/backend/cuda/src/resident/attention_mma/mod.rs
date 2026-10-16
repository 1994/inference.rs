//! AOT FA4 prefill: GPU page gathering, architecture-specific MMA, then F32 readout.
//! This BF16 profile is distinct from the canonical F32 exact attention profile.
pub mod catalog;
mod host;
mod recorded;
mod workspace;
pub use super::attention_oxide::Bindings;
use crate::device::{CudaDevice, device_error};
use catalog::{DIM, HEADS, KV_HEADS, Variant};
use cuda_core::{Function, LaunchConfig};
use cutile::{cuda_async::launch::AsyncKernelLaunch, prelude::*};
use infer_core::Result;
use std::collections::BTreeMap;
pub use workspace::{Workspace, eligible};

const PTX: &str = include_str!("../../../kernels/fa4-attention/bridge.ptx");
const THREADS: u32 = 256;
const RESIDENT_BLOCKS_PER_SM: u32 = 8;

pub struct Support {
    prepare: Arc<Function>,
    finish: Arc<Function>,
    apis: BTreeMap<Variant, Arc<host::Api>>,
    scratch: workspace::Scratch,
    stream: usize,
    grid: u32,
    capacity: usize,
    lanes: usize,
}
impl Support {
    pub fn load(device: &CudaDevice, capacity: usize, lanes: usize) -> Result<Self> {
        if capacity == 0
            || !capacity.is_multiple_of(crate::constants::KV_BLOCK_TOKENS)
            || i32::try_from(capacity).is_err()
            || i32::try_from(lanes).is_err()
        {
            return Err(infer_core::Error::invalid("MMA scratch dimensions"));
        }
        catalog::workspace_bytes(capacity, lanes)?;
        let module = device
            .stream
            .device()
            .load_module_from_ptx_src(PTX)
            .map_err(device_error)?;
        let sms = device.profile()?.multiprocessors;
        Ok(Self {
            prepare: Arc::new(module.load_function("prepare_fa4").map_err(device_error)?),
            finish: Arc::new(module.load_function("finish_fa4").map_err(device_error)?),
            apis: BTreeMap::new(),
            scratch: workspace::Scratch::new(device, capacity, lanes)?,
            stream: device.stream.cu_stream() as usize,
            grid: sms.saturating_mul(RESIDENT_BLOCKS_PER_SM).max(1),
            capacity,
            lanes,
        })
    }
    pub fn prepare_variant(&mut self, device: &CudaDevice, variant: Variant) -> Result<()> {
        if !Variant::candidates(&device.profile()?.architecture).contains(&variant) {
            return Err(infer_core::Error::invalid(
                "MMA variant architecture mismatch",
            ));
        }
        let ordinal = i32::try_from(device.stream.device().ordinal()).map_err(device_error)?;
        self.apis
            .insert(variant, host::Api::load(variant, ordinal)?);
        Ok(())
    }
    fn accepts<E: DType>(
        &self,
        b: &Bindings<'_, E>,
        (heads, kv_heads, dim, lanes, capacity): (usize, usize, usize, usize, usize),
        window: i32,
        scales: [f32; 2],
    ) -> std::result::Result<bool, DeviceError> {
        Ok(catalog::geometry(
            heads,
            kv_heads,
            dim,
            lanes,
            usize::try_from(window).unwrap_or(usize::MAX),
        ) && lanes <= self.lanes
            && capacity == self.capacity
            && E::DTYPE == cuda_core::f8e4m3fn::DTYPE
            && size_of::<E>() == 1
            && scales.iter().all(|v| v.is_finite() && *v > 0.0)
            && b.query.is_contiguous()
            && b.query.size() == lanes * HEADS * DIM
            && b.keys.is_contiguous()
            && b.values.is_contiguous()
            && b.keys.shape()
                == [
                    i32::try_from(KV_HEADS).map_err(super::capture::error)?,
                    i32::try_from(capacity).map_err(super::capture::error)?,
                    i32::try_from(DIM).map_err(super::capture::error)?,
                ]
            && b.values.shape() == b.keys.shape()
            && b.output.is_contiguous()
            && b.output.size() == lanes * HEADS * DIM
            && b.metadata.is_contiguous()
            && b.metadata.size() >= catalog::METADATA_FIELDS
            && b.table.is_contiguous()
            && b.table.size() >= capacity / crate::constants::KV_BLOCK_TOKENS)
    }
    #[expect(
        unsafe_code,
        reason = "checked contiguous graph-owned operands are marshalled into the embedded bridge; recorded execution retains all owners"
    )]
    pub fn record<E: DType>(
        &mut self,
        scope: &Scope,
        b: Bindings<'_, E>,
        shape: (usize, usize, usize, usize, usize),
        window: i32,
        scales: [f32; 2],
        variant: Variant,
    ) -> std::result::Result<bool, DeviceError> {
        let (_, _, _, lanes, capacity) = shape;
        if !self.accepts(&b, shape, window, scales)? {
            return Ok(false);
        }
        let api = self
            .apis
            .get(&variant)
            .ok_or_else(|| super::capture::error("MMA variant not prepared before capture"))?;
        let ptr = |tensor: &Tensor<cutile::half::bf16>| tensor.device_pointer().cu_deviceptr();
        let pointers = [
            ptr(&self.scratch.query),
            ptr(&self.scratch.keys),
            ptr(&self.scratch.values),
            ptr(&self.scratch.output),
            self.scratch.used.device_pointer().cu_deviceptr(),
            self.scratch.used.device_pointer().cu_deviceptr() + size_of::<i32>() as u64,
        ];
        let [
            q_pointer,
            k_pointer,
            v_pointer,
            o_pointer,
            used_q_pointer,
            _,
        ] = pointers;
        let mut prepare = AsyncKernelLaunch::new(Arc::clone(&self.prepare));
        for pointer in [
            b.query.device_pointer().cu_deviceptr(),
            b.keys.device_pointer().cu_deviceptr(),
            b.values.device_pointer().cu_deviceptr(),
            b.metadata.device_pointer().cu_deviceptr(),
            b.table.device_pointer().cu_deviceptr(),
            q_pointer,
            k_pointer,
            v_pointer,
            used_q_pointer,
        ] {
            // SAFETY: the checked operands and scratch are retained by Recorded.
            unsafe {
                prepare.push_device_ptr(pointer);
            }
        }
        prepare.push_arg(u32::try_from(capacity).map_err(super::capture::error)?);
        prepare.push_arg(u32::try_from(lanes).map_err(super::capture::error)?);
        prepare.push_arg(scales[0]);
        prepare.push_arg(scales[1]);
        prepare.set_launch_config(LaunchConfig {
            grid_dim: (self.grid, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        });
        let mut finish = AsyncKernelLaunch::new(Arc::clone(&self.finish));
        for pointer in [
            o_pointer,
            b.output.device_pointer().cu_deviceptr(),
            b.metadata.device_pointer().cu_deviceptr(),
        ] {
            // SAFETY: these are the same retained, checked outputs and metadata.
            unsafe {
                finish.push_device_ptr(pointer);
            }
        }
        finish.push_arg(u32::try_from(lanes).map_err(super::capture::error)?);
        finish.set_launch_config(LaunchConfig {
            grid_dim: (
                u32::try_from((lanes * HEADS * DIM).div_ceil(THREADS as usize))
                    .map_err(super::capture::error)?,
                1,
                1,
            ),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        });
        scope.record(recorded::Recorded {
            prepare,
            finish,
            api: Arc::clone(api),
            pointers,
            stream: self.stream,
            lanes,
            capacity,
            bindings: b,
            scratch: &mut self.scratch,
        })?;
        Ok(true)
    }
}
pub fn artifact_identity() -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(PTX);
    hash.update(include_bytes!("../../../kernels/fa4-attention/bridge.cu"));
    hash.update(include_bytes!(
        "../../../kernels/fa4-attention/manifest.json"
    ));
    hash.update(catalog::PROFILE);
    for arch in ["sm_90", "sm_100", "sm_103", "sm_120"] {
        for variant in Variant::candidates(arch) {
            hash.update(variant.bytes());
        }
    }
    format!("{:x}", hash.finalize())
}
