//! Owned pinned destinations; host slices are exposed only after submission completion.
use super::{CudaDevice, device_error};
use cuda_core::{CudaContext, PinnedHostBuffer};
use cutile::{prelude::*, tensor::KernelInputStored};
use infer_core::{Error, ErrorCode, Result};
use std::sync::{Arc, Mutex};

/// Total pinned bytes admitted across one batch of readback copies.
const MAX_READBACK_STAGING_BYTES: usize = 64 * crate::constants::MIB;

#[cfg(test)]
#[path = "../../tests/unit/device_readback_check.rs"]
mod check;

type Buffer = Arc<Mutex<PinnedHostBuffer<f32>>>;

/// A half-open element range `[skip, skip + len)` of a device tensor to copy out.
#[derive(Clone)]
pub struct ReadbackSource {
    pub tensor: Arc<Tensor<f32>>,
    pub skip: usize,
    pub len: usize,
}

impl ReadbackSource {
    #[must_use]
    pub fn whole(tensor: Arc<Tensor<f32>>) -> Self {
        let len = tensor.size();
        Self {
            tensor,
            skip: 0,
            len,
        }
    }
}

#[derive(Default)]
pub struct Readbacks {
    context: Option<Arc<CudaContext>>,
    buffers: Vec<Option<Buffer>>,
    poisoned: bool,
}

impl Readbacks {
    pub(crate) fn run(
        &mut self,
        device: &CudaDevice,
        graph: &CudaGraph<()>,
        sources: &[Option<ReadbackSource>],
    ) -> Result<Vec<Vec<f32>>> {
        if self.poisoned || !Arc::ptr_eq(graph.stream(), &device.stream) {
            return Err(Error::invalid(
                "poisoned readback or mismatched CUDA graph stream",
            ));
        }
        self.prepare(device, sources)?;
        let mut copies = DeviceOpVec::with_capacity(sources.len());
        for (source, buffer) in sources.iter().zip(&self.buffers) {
            if let Some(source) = source {
                copies.push(PinnedCopy {
                    source: Arc::clone(&source.tensor),
                    skip: source.skip,
                    len: source.len,
                    destination: Arc::clone(
                        buffer
                            .as_ref()
                            .ok_or_else(|| Error::invariant("pinned slot"))?,
                    ),
                });
            }
        }
        self.poisoned = true;
        device
            .stream
            .device()
            .bind_to_thread()
            .map_err(device_error)?;
        graph
            .launch()
            .then(move |()| copies)
            .sync_on(&device.stream)
            .map_err(device_error)?;
        // sync_on succeeded: every queued D2H finished. No GPU work can touch
        // these private buffers until another exclusive &mut self call.
        let results = sources
            .iter()
            .zip(&self.buffers)
            .map(|(source, buffer)| {
                if source.is_none() {
                    return Ok(Vec::new());
                }
                let buffer = buffer
                    .as_ref()
                    .ok_or_else(|| Error::invariant("pinned result"))?;
                Ok(buffer.lock().map_err(device_error)?.as_slice().to_vec())
            })
            .collect::<Result<Vec<_>>>()?;
        self.poisoned = false;
        Ok(results)
    }

    fn prepare(&mut self, device: &CudaDevice, sources: &[Option<ReadbackSource>]) -> Result<()> {
        let mut bytes = 0usize;
        for (index, source) in sources.iter().enumerate() {
            if let Some(source) = source {
                if source.tensor.device_id() != device.stream.device().ordinal() {
                    return Err(Error::invalid("pinned readback device mismatch"));
                }
                if source
                    .skip
                    .checked_add(source.len)
                    .is_none_or(|end| end > source.tensor.size())
                {
                    return Err(Error::invalid("pinned readback range out of bounds"));
                }
                bytes = bytes
                    .checked_add(
                        source
                            .len
                            .checked_mul(crate::constants::F32_BYTES)
                            .ok_or_else(|| Error::invalid("pinned size overflow"))?,
                    )
                    .ok_or_else(|| Error::invalid("pinned size overflow"))?;
            } else if let Some(Some(buffer)) = self.buffers.get(index) {
                bytes = bytes
                    .checked_add(buffer.lock().map_err(device_error)?.num_bytes())
                    .ok_or_else(|| Error::invalid("pinned size overflow"))?;
            }
        }
        if bytes > MAX_READBACK_STAGING_BYTES {
            return Err(Error::new(
                ErrorCode::Capacity,
                "pinned readbacks exceed 64 MiB",
            ));
        }
        self.buffers.resize_with(sources.len(), || None);
        for (source, slot) in sources.iter().zip(&mut self.buffers) {
            let Some(source) = source else {
                continue;
            };
            if let Some(buffer) = slot
                && buffer.lock().map_err(device_error)?.len() == source.len
            {
                continue;
            }
            if self.context.is_none() {
                self.context =
                    Some(CudaContext::new(device.stream.device().ordinal()).map_err(device_error)?);
            }
            let context = self
                .context
                .as_ref()
                .ok_or_else(|| Error::invariant("pinned context"))?;
            *slot = Some(Arc::new(Mutex::new(
                PinnedHostBuffer::zeroed(context, source.len).map_err(device_error)?,
            )));
        }
        Ok(())
    }
}

struct PinnedCopy {
    source: Arc<Tensor<f32>>,
    skip: usize,
    len: usize,
    destination: Buffer,
}

#[expect(
    unsafe_code,
    reason = "Audited asynchronous D2H: the [skip, skip + len) element range is bounds-checked \
              against the source before enqueue; source access lease and pinned destination are \
              retained by the live submission"
)]
impl DeviceOp for PinnedCopy {
    type Output = ();

    unsafe fn execute(self, context: &ExecutionContext) -> std::result::Result<(), DeviceError> {
        self.source.retain(context)?;
        context.retain(Arc::clone(&self.destination))?;
        if self
            .skip
            .checked_add(self.len)
            .is_none_or(|end| end > self.source.size())
            || self.len.checked_mul(crate::constants::F32_BYTES).is_none()
        {
            return Err(DeviceError::Internal(
                "pinned copy range out of bounds".into(),
            ));
        }
        let mut destination = self
            .destination
            .lock()
            .map_err(|e| DeviceError::Internal(e.to_string()))?;
        if destination.len() != self.len {
            return Err(DeviceError::Internal("pinned copy size mismatch".into()));
        }
        if self.len == 0 {
            return Ok(());
        }
        let bytes = self.len * crate::constants::F32_BYTES;
        let offset = self
            .skip
            .checked_mul(crate::constants::F32_BYTES)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| DeviceError::Internal("pinned copy offset overflow".into()))?;
        // SAFETY: owned, initialized destination has exactly self.len f32s; the
        // checked source range keeps the offset read inside the source tensor.
        // Both allocations outlive the submission. Host reads happen only after
        // sync_on; failures poison the owner and submission cleanup retains work.
        let status = unsafe {
            cuda_core::sys::cuMemcpyDtoHAsync_v2(
                destination.as_mut_ptr().cast(),
                self.source.device_pointer().cu_deviceptr() + offset,
                bytes,
                context.get_cuda_stream().cu_stream(),
            )
        };
        if status != cuda_core::sys::cudaError_enum_CUDA_SUCCESS {
            return Err(DeviceError::Internal(format!(
                "pinned D2H CUDA status {status}"
            )));
        }
        Ok(())
    }
}

impl IntoFuture for PinnedCopy {
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
