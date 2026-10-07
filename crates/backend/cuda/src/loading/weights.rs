use crate::{device::CudaDevice, strategy::LinearTiling};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::{half::bf16, prelude::*};
use infer_core::{Error, Result};
use infer_ir::DType;
use infer_models::{
    DeviceWeight, QuantizedPackage, WeightEncoding, WeightSource, convert_float_bytes,
};
use std::sync::Arc;

const STAGING: u64 = 2 * 1024 * 1024 * 1024;
/// Decoded BF16 byte budget for one NVFP4 projection falling back to host decoding.
const NVFP4_DECODE_BUDGET: usize = crate::constants::GIB;

#[derive(Clone)]
pub enum Projection {
    Dense(Arc<Tensor<bf16>>),
    Fp8(Arc<Tensor<f8e4m3fn>>, Arc<Tensor<f32>>),
    Fp4(Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>, f32),
}

/// Read floating weights under a 2 GiB encoded staging limit.
/// # Errors
/// Rejects invalid storage, unsupported floating formats or failed reads.
pub fn floats(package: &mut QuantizedPackage, source: &WeightSource) -> Result<Vec<f32>> {
    let bytes = package.read_view(source, STAGING)?;
    let converted = convert_float_bytes(bytes, source.dtype)?;
    Ok(converted
        .as_chunks::<{ crate::constants::F32_BYTES }>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

impl Projection {
    #[must_use]
    pub fn resident(&self) -> crate::mlp::ProjectionWeight {
        use crate::mlp::ProjectionWeight;
        match self {
            Self::Dense(w) => ProjectionWeight::Dense(w.clone()),
            Self::Fp8(w, s) => ProjectionWeight::Fp8(w.clone(), s.clone()),
            Self::Fp4(w, s, g) => ProjectionWeight::Fp4(w.clone(), s.clone(), *g),
        }
    }
    /// # Errors
    /// Rejects malformed quantization, staging limits or failed CUDA allocations.
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
                    .read_view(&weight.data, STAGING)?
                    .iter()
                    .copied()
                    .map(f8e4m3fn)
                    .collect();
                let scaling = floats(package, scale()?)?;
                Ok(Self::Fp8(
                    device.upload(values, &weight.shape)?,
                    device.upload(scaling, &[weight.shape[0]])?,
                ))
            }
            WeightEncoding::Nvfp4 => Self::load_nvfp4(device, package, weight),
        }
    }

    fn load_nvfp4(
        device: &CudaDevice,
        package: &mut QuantizedPackage,
        weight: &DeviceWeight,
    ) -> Result<Self> {
        // The model provider declares its storage preference; the device answers what it can
        // compute. No architecture enum is consulted here.
        let native = package
            .imported
            .precision
            .resolve(|dtype| device.target().supports_compute(dtype))
            .storage
            == DType::Fp4E2M1;
        let scale = weight
            .scale
            .as_ref()
            .ok_or_else(|| Error::invalid("missing NVFP4 block scale"))?;
        let global = weight
            .global_scale
            .as_ref()
            .ok_or_else(|| Error::invalid("missing NVFP4 global scale"))?;
        let global = floats(package, global)?;
        let [global] = global.as_slice() else {
            return Err(Error::invalid("NVFP4 global scale must be scalar"));
        };
        if !global.is_finite() || *global <= 0.0 {
            return Err(Error::invalid("invalid NVFP4 global scale"));
        }
        let values = package.read_view(&weight.data, STAGING)?;
        let scaling = package.read_view(scale, STAGING)?;
        if native {
            Ok(Self::Fp4(
                device.upload(
                    values.iter().copied().map(f4e2m1fnx2::from_bits).collect(),
                    &weight.data.shape,
                )?,
                device.upload(
                    scaling.iter().copied().map(f8e4m3fn).collect(),
                    &scale.shape,
                )?,
                *global,
            ))
        } else {
            let [rows, columns] = weight.shape.as_slice() else {
                return Err(Error::invalid("NVFP4 projection must be a matrix"));
            };
            let decoded = crate::nvfp4::to_bf16(
                values,
                scaling,
                *global,
                *rows,
                *columns,
                NVFP4_DECODE_BUDGET,
            )?;
            Ok(Self::Dense(device.upload(
                decoded.into_iter().map(bf16::from_bits).collect(),
                &weight.shape,
            )?))
        }
    }

    /// # Errors
    /// Rejects incompatible projection dimensions or failed CUDA execution.
    pub fn apply(&self, device: &CudaDevice, input: &[f32]) -> Result<Vec<f32>> {
        let vector = device.upload(input.to_vec(), &[input.len()])?;
        let tile = LinearTiling::new(
            crate::constants::DEFAULT_TILE_ROWS,
            crate::constants::DEFAULT_TILE_COLUMNS,
        )?;
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
