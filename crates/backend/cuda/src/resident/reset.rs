//! In-place state clearing recorded as native driver memset nodes.
use cutile::{prelude::*, tensor::KernelOutputStored};

pub(super) struct Clear<'a>(pub &'a mut Tensor<f32>);

#[expect(
    unsafe_code,
    reason = "The live tensor bounds the memset extent; its write lease is retained by the capture context and replay submission"
)]
impl DeviceOp for Clear<'_> {
    type Output = ();
    unsafe fn execute(self, context: &ExecutionContext) -> Result<(), DeviceError> {
        let pointer = self.0.device_pointer().cu_deviceptr();
        let bytes = self.0.num_bytes();
        KernelOutputStored::retain(&self.0.partition([256]), context)?;
        // SAFETY: the tensor owns this exact byte range on the captured stream's device.
        let status = unsafe {
            cuda_core::sys::cuMemsetD8Async(
                pointer,
                0,
                bytes,
                context.get_cuda_stream().cu_stream(),
            )
        };
        if status == cuda_core::sys::cudaError_enum_CUDA_SUCCESS {
            Ok(())
        } else {
            Err(DeviceError::Launch(format!("state clear: {status:?}")))
        }
    }
}
impl GraphNode for Clear<'_> {}

impl IntoFuture for Clear<'_> {
    type Output = Result<(), DeviceError>;
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
#[path = "../../tests/unit/resident_reset.rs"]
mod tests;
