//! Captured execution owns every input, scratch tensor, module and host ABI library.
use super::{Bindings, host, workspace::Scratch};
use cuda_core::sys::CUdeviceptr;
use cutile::{
    cuda_async::{
        device_context::with_default_device_policy, device_future::DeviceFuture,
        launch::AsyncKernelLaunch,
    },
    prelude::*,
};
pub(super) struct Recorded<'a, E: DType> {
    pub prepare: AsyncKernelLaunch,
    pub finish: AsyncKernelLaunch,
    pub api: Arc<host::Api>,
    pub pointers: [CUdeviceptr; super::catalog::OPERANDS],
    pub stream: usize,
    pub lanes: usize,
    pub capacity: usize,
    pub bindings: Bindings<'a, E>,
    pub scratch: &'a mut Scratch,
}
impl<E: DType> GraphNode for Recorded<'_, E> {}
#[expect(
    unsafe_code,
    reason = "the exact stream and all graph-owned tensor extents are checked before the three ordered launches; retained owners outlive graph destruction"
)]
impl<E: DType> DeviceOp for Recorded<'_, E> {
    type Output = ();
    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        if ctx.get_cuda_stream().cu_stream() as usize != self.stream {
            return Err(DeviceError::Internal(
                "MMA attention stream mismatch".into(),
            ));
        }
        KernelInputStored::retain(&self.bindings.query, ctx)?;
        KernelInputStored::retain(&self.bindings.keys, ctx)?;
        KernelInputStored::retain(&self.bindings.values, ctx)?;
        KernelInputStored::retain(&self.bindings.metadata, ctx)?;
        KernelInputStored::retain(&self.bindings.table, ctx)?;
        KernelOutputStored::retain(
            &(&mut *self.bindings.output).partition([1, super::DIM]),
            ctx,
        )?;
        self.scratch.retain(ctx)?;
        ctx.retain(Arc::clone(&self.prepare.func))?;
        ctx.retain(Arc::clone(&self.finish.func))?;
        ctx.retain(Arc::clone(&self.api))?;
        // SAFETY: one checked stream orders conversion, AOT attention and readout.
        unsafe {
            self.prepare.execute(ctx)?;
        }
        self.api
            .launch(
                self.pointers,
                self.lanes,
                self.capacity,
                ctx.get_cuda_stream().cu_stream(),
            )
            .map_err(super::super::capture::error)?;
        // SAFETY: the BF16 output and final F32 output are both retained above.
        unsafe { self.finish.execute(ctx) }
    }
}
impl<E: DType> IntoFuture for Recorded<'_, E> {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), Self>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(error)) | Err(error) => DeviceFuture::failed(error),
        }
    }
}
