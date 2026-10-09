//! Per-program activation quantization buffers, retained as long as captured graphs.
use super::kernels;
use crate::{
    device::{CudaDevice, device_error},
    mlp::ProjectionWeight,
    resident::ProgramWeights,
};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_ir::{DataflowGraph, TensorOp};
use std::collections::BTreeMap;

type Quantized = (Tensor<f4e2m1fnx2>, Tensor<f8e4m3fn>);

pub struct Workspace {
    buffers: BTreeMap<(usize, usize), Quantized>,
    /// Keyed by `(rows, columns, scale_columns)`: per-channel and block-scaled FP8 share an
    /// operand geometry but need different activation-scale extents, so geometry alone would let
    /// whichever projection is visited first decide the layout for the other.
    fp8: BTreeMap<(usize, usize, usize), (Tensor<f8e4m3fn>, Tensor<f32>)>,
}

impl Workspace {
    pub(crate) fn program(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
    ) -> Result<Self> {
        let mut widths = vec![1, weights.batch_width];
        widths.extend(weights.prompt_widths());
        Self::new(device, graph, weights, &widths)
    }

    pub(crate) fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        widths: &[usize],
    ) -> Result<Self> {
        let mut buffers = BTreeMap::new();
        let mut fp8 = BTreeMap::new();
        for node in &graph.nodes {
            if node.op != TensorOp::Linear
                || (!weights.input_scales.contains_key(&node.inputs[1])
                    && !weights.fp8_inputs.contains(&node.inputs[1]))
            {
                continue;
            }
            let columns = graph
                .tensors
                .iter()
                .find(|t| t.id == node.inputs[0])
                .ok_or_else(|| Error::invariant("FP4 input tensor"))?
                .elements()?;
            if weights.fp8_inputs.contains(&node.inputs[1]) {
                let block = matches!(
                    weights.projections.get(&node.inputs[1]),
                    Some(ProjectionWeight::Fp8Block(..))
                );
                let scale_columns = if block {
                    columns / crate::constants::FP8_BLOCK_COLUMNS
                } else {
                    1
                };
                for &rows in widths {
                    if rows > 0 && !fp8.contains_key(&(rows, columns, scale_columns)) {
                        fp8.insert(
                            (rows, columns, scale_columns),
                            (
                                api::zeros::<f8e4m3fn>(&[rows, columns])
                                    .sync_on(&device.stream)
                                    .map_err(device_error)?,
                                api::zeros::<f32>(&[rows, scale_columns])
                                    .sync_on(&device.stream)
                                    .map_err(device_error)?,
                            ),
                        );
                    }
                }
                continue;
            }
            for &rows in widths {
                if rows == 0 || buffers.contains_key(&(rows, columns)) {
                    continue;
                }
                buffers.insert(
                    (rows, columns),
                    (
                        api::zeros::<f4e2m1fnx2>(&[rows, columns / 2])
                            .sync_on(&device.stream)
                            .map_err(device_error)?,
                        api::zeros::<f8e4m3fn>(&[
                            rows,
                            columns / crate::constants::NVFP4_GROUP_SIZE,
                        ])
                        .sync_on(&device.stream)
                        .map_err(device_error)?,
                    ),
                );
            }
        }
        Ok(Self { buffers, fp8 })
    }

    pub(super) fn record(
        &mut self,
        scope: &Scope,
        weight: &ProjectionWeight,
        scale: f32,
        input: &TensorView<'_, f32>,
        output: &mut Tensor<f32>,
        columns: usize,
    ) -> std::result::Result<(), DeviceError> {
        let ProjectionWeight::Fp4(w, s, global) = weight else {
            return Err(super::super::capture::error("FP4 workspace weight"));
        };
        let rows = output.size()
            / usize::try_from(output.shape()[1]).map_err(super::super::capture::error)?;
        let (q, qs) = self
            .buffers
            .get_mut(&(rows, columns))
            .ok_or_else(|| super::super::capture::error("FP4 quantization workspace missing"))?;
        scope.record(
            kernels::quantize(
                (&mut *q).partition(crate::constants::NVFP4_QUANT_CODES_TILE),
                (&mut *qs).partition(crate::constants::NVFP4_QUANT_SCALES_TILE),
                input,
                scale,
            )
            .generics(vec![columns.to_string()]),
        )?;
        let tile = crate::constants::quant_gemm_tile(rows);
        scope.record(
            kernels::packed(
                output.partition(tile),
                &*q,
                &*qs,
                w,
                s,
                1.0 / (global * scale),
            )
            .generics(vec![
                columns.to_string(),
                tile[0].to_string(),
                tile[1].to_string(),
            ]),
        )?;
        Ok(())
    }
    pub(super) fn record_fp8(
        &mut self,
        scope: &Scope,
        weight: &ProjectionWeight,
        input: &TensorView<'_, f32>,
        output: &mut Tensor<f32>,
        columns: usize,
    ) -> std::result::Result<(), DeviceError> {
        use super::super::{capture::error, fp8_gemm::kernels};
        if !matches!(
            weight,
            ProjectionWeight::Fp8(..) | ProjectionWeight::Fp8Block(..)
        ) {
            return Err(error("FP8 workspace weight"));
        }
        let rows = usize::try_from(output.shape()[0]).map_err(error)?;
        // The scales layout follows the weight's quantization mode, and the key has to say which.
        let scale_columns = match weight {
            ProjectionWeight::Fp8Block(..) => columns / crate::constants::FP8_BLOCK_COLUMNS,
            _ => 1,
        };
        let (q, qs) = self
            .fp8
            .get_mut(&(rows, columns, scale_columns))
            .ok_or_else(|| error("FP8 quantization workspace missing"))?;
        let tile = crate::constants::quant_gemm_tile(rows);
        let generics = vec![
            columns.to_string(),
            tile[0].to_string(),
            tile[1].to_string(),
        ];
        match weight {
            ProjectionWeight::Fp8Block(w, block_scale) => {
                scope.record(
                    kernels::quantize_block(
                        (&mut *q).partition([1, crate::constants::FP8_BLOCK_COLUMNS]),
                        (&mut *qs).partition([1, 1]),
                        input,
                    )
                    .generics(vec![
                        columns.to_string(),
                        crate::constants::FP8_BLOCK_COLUMNS.to_string(),
                    ]),
                )?;
                scope.record(
                    kernels::matmul_block(output.partition(tile), &*q, w, &*qs, block_scale)
                        .generics(generics),
                )?;
            }
            ProjectionWeight::Fp8(w, channel_scale) => {
                scope.record(
                    kernels::quantize(
                        (&mut *q).partition([1, columns.next_power_of_two()]),
                        (&mut *qs).partition([1, 1]),
                        input,
                    )
                    .generics(vec![
                        columns.to_string(),
                        columns.next_power_of_two().to_string(),
                    ]),
                )?;
                scope.record(
                    kernels::matmul(output.partition(tile), &*q, w, &*qs, channel_scale)
                        .generics(generics),
                )?;
            }
            _ => return Err(error("FP8 workspace weight")),
        }
        Ok(())
    }
}
