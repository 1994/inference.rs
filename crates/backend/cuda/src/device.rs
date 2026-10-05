//! Safe CUDA memory and kernel launch boundary.
use cutile::prelude::*;
use infer_core::{Error as InferError, ErrorCode, Result};
use std::sync::Arc;

#[derive(Clone)]
pub struct CudaDevice {
    pub(crate) stream: Arc<cuda_core::Stream>,
    target: crate::target::CudaTarget,
}

pub(crate) fn device_error(error: impl std::fmt::Display) -> InferError {
    InferError::new(ErrorCode::Backend, error.to_string())
}

impl CudaDevice {
    /// # Errors
    /// Fails when the requested CUDA device or stream is unavailable.
    pub fn new(ordinal: usize) -> Result<Self> {
        let device = Device::new(ordinal).map_err(device_error)?;
        let target = query_target(&device)?;
        Ok(Self {
            stream: device.new_stream().map_err(device_error)?,
            target,
        })
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
        self.matvec_tiled(x, w, crate::strategy::LinearTiling::new(4, 128)?)
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
