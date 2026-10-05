use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::{half::bf16, prelude::*};
use infer_backend_cuda::{device::CudaDevice, strategy::LinearTiling};
use infer_core::{Error, Result};
use infer_models::{
    DeviceWeight, QuantizedPackage, WeightEncoding, WeightSource, convert_float_bytes,
};
use std::sync::Arc;

const STAGING: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Clone)]
pub enum Projection {
    Dense(Arc<Tensor<bf16>>),
    Fp8(Arc<Tensor<f8e4m3fn>>, Arc<Tensor<f32>>),
    Fp4(Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>, f32),
}

pub fn floats(package: &mut QuantizedPackage, source: &WeightSource) -> Result<Vec<f32>> {
    let bytes = package.read(source, STAGING)?;
    let converted = convert_float_bytes(&bytes, source.dtype)?;
    Ok(converted
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

impl Projection {
    pub fn load(
        device: &CudaDevice,
        package: &mut QuantizedPackage,
        weight: &DeviceWeight,
    ) -> Result<Self> {
        let scale = || {
            weight
                .scale
                .as_ref()
                .ok_or_else(|| Error::invalid("missing projection scale"))
        };
        match weight.encoding {
            WeightEncoding::Float => {
                let values = floats(package, &weight.data)?
                    .into_iter()
                    .map(bf16::from_f32)
                    .collect();
                Ok(Self::Dense(device.upload(values, &weight.shape)?))
            }
            WeightEncoding::Fp8Channel => {
                let values = package
                    .read(&weight.data, STAGING)?
                    .into_iter()
                    .map(f8e4m3fn)
                    .collect();
                let scaling = floats(package, scale()?)?;
                Ok(Self::Fp8(
                    device.upload(values, &weight.shape)?,
                    device.upload(scaling, &[weight.shape[0]])?,
                ))
            }
            WeightEncoding::Nvfp4 => {
                let values = package
                    .read(&weight.data, STAGING)?
                    .into_iter()
                    .map(f4e2m1fnx2::from_bits)
                    .collect();
                let scaling = package
                    .read(scale()?, STAGING)?
                    .into_iter()
                    .map(f8e4m3fn)
                    .collect();
                let global = weight
                    .global_scale
                    .as_ref()
                    .ok_or_else(|| Error::invalid("missing NVFP4 global scale"))?;
                let global = floats(package, global)?;
                Ok(Self::Fp4(
                    device.upload(values, &weight.data.shape)?,
                    device.upload(scaling, &scale()?.shape)?,
                    global[0],
                ))
            }
        }
    }

    pub fn apply(&self, device: &CudaDevice, input: &[f32]) -> Result<Vec<f32>> {
        let vector = device.upload(input.to_vec(), &[input.len()])?;
        let tile = LinearTiling::new(16, 256)?;
        let output = match self {
            Self::Dense(weights) => device.matvec_tiled(vector, weights.clone(), tile)?,
            Self::Fp8(weights, scales) => {
                device.fp8_matvec(vector, weights.clone(), scales.clone(), tile)?
            }
            Self::Fp4(weights, scales, global) => {
                device.nvfp4_matvec(vector, weights.clone(), scales.clone(), *global, tile)?
            }
        };
        device.read(&output)
    }
}
