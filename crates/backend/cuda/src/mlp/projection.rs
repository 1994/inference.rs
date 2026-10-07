//! Storage strategy shared by graph capture and model-specific binding adapters.
use crate::{kernels::linear, strategy::LinearTiling};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::{half::bf16, prelude::*};
use infer_core::{Error, Result};
use std::sync::Arc;

#[derive(Clone)]
pub enum ProjectionWeight {
    Dense(Arc<Tensor<bf16>>),
    Fp8(Arc<Tensor<f8e4m3fn>>, Arc<Tensor<f32>>),
    Fp4(Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>, f32),
}

impl ProjectionWeight {
    pub(crate) fn validate(&self, rows: usize, columns: usize) -> Result<()> {
        let matches = |shape: &[i32], expected: &[usize]| {
            shape.len() == expected.len()
                && shape
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| usize::try_from(*a).ok() == Some(*b))
        };
        let valid = match self {
            Self::Dense(w) => matches(w.shape(), &[rows, columns]),
            Self::Fp8(w, s) => matches(w.shape(), &[rows, columns]) && matches(s.shape(), &[rows]),
            Self::Fp4(w, s, g) => {
                columns.is_multiple_of(crate::constants::NVFP4_GROUP_SIZE)
                    && matches(w.shape(), &[rows, columns / 2])
                    && matches(
                        s.shape(),
                        &[rows, columns / crate::constants::NVFP4_GROUP_SIZE],
                    )
                    && g.is_finite()
                    && *g > 0.0
            }
        };
        if valid {
            Ok(())
        } else {
            Err(Error::invalid("MLP projection dimensions/scales"))
        }
    }

    pub(crate) fn record(
        &self,
        scope: &Scope,
        out: &mut Tensor<f32>,
        input: &Tensor<f32>,
        columns: usize,
        tile: LinearTiling,
    ) -> std::result::Result<(), DeviceError> {
        let mut generics = vec![
            tile.rows().to_string(),
            tile.columns().to_string(),
            columns.to_string(),
        ];
        match self {
            Self::Dense(w) => {
                generics.insert(0, bf16::DTYPE.as_str().into());
                scope.record(
                    linear::dense(out.partition([tile.rows()]), input, w).generics(generics),
                )?;
            }
            Self::Fp8(w, s) => {
                scope.record(
                    linear::fp8(out.partition([tile.rows()]), input, w, s).generics(generics),
                )?;
            }
            Self::Fp4(w, s, g) => {
                generics.extend([
                    (tile.columns() / 2).to_string(),
                    (tile.columns() / crate::constants::NVFP4_GROUP_SIZE).to_string(),
                ]);
                scope.record(
                    linear::nvfp4(out.partition([tile.rows()]), input, w, s, g.recip())
                        .generics(generics),
                )?;
            }
        }
        Ok(())
    }
}
