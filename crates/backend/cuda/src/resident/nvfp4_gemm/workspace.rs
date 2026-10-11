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
/// Keyed by `(rows, columns, activation-scale columns)`.
type Fp8Key = (usize, usize, usize);
/// Quantized activation and its per-token scales.
type Fp8Buffers = (Tensor<f8e4m3fn>, Tensor<f32>);

pub struct Workspace {
    strategies: BTreeMap<
        (crate::tuning::resident_gemm::WeightKey, usize, usize, usize),
        crate::tuning::resident_gemm::GemmConfig,
    >,
    buffers: BTreeMap<(usize, usize), Quantized>,
    /// Keyed by `(rows, columns, scale_columns)`: per-channel and block-scaled FP8 share an
    /// operand geometry but need different activation-scale extents, so geometry alone would let
    /// whichever projection is visited first decide the layout for the other.
    fp8: BTreeMap<Fp8Key, Fp8Buffers>,
}

impl Workspace {
    pub(crate) fn program(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        verification: bool,
        wide_prompt: bool,
    ) -> Result<Self> {
        let widths = Self::captured_widths(
            weights.batch_width,
            weights.prefill_width,
            weights.narrow_prefill_width,
            verification,
            wide_prompt,
        );
        Self::new(device, graph, weights, &widths)
    }

    pub(crate) fn captured_widths(
        batch: usize,
        wide: usize,
        narrow: usize,
        verification: bool,
        wide_prompt: bool,
    ) -> Vec<usize> {
        captured_widths(batch, wide, narrow, verification, wide_prompt)
    }

    pub(crate) fn ensure_width(
        &mut self,
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        width: usize,
    ) -> Result<()> {
        let mut extra = Self::new(device, graph, weights, &[width])?;
        self.strategies.append(&mut extra.strategies);
        self.buffers.append(&mut extra.buffers);
        self.fp8.append(&mut extra.fp8);
        Ok(())
    }

    pub(crate) fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        widths: &[usize],
    ) -> Result<Self> {
        let mut buffers = BTreeMap::new();
        let mut fp8 = BTreeMap::new();
        let mut strategies = BTreeMap::new();
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
            if let Some(weight @ ProjectionWeight::Fp4(..)) =
                weights.projections.get(&node.inputs[1])
                && let Some(&scale) = weights.input_scales.get(&node.inputs[1])
            {
                let outputs = graph
                    .tensors
                    .iter()
                    .find(|tensor| tensor.id == node.outputs[0])
                    .ok_or_else(|| Error::invariant("FP4 output tensor"))?
                    .elements()?;
                let weight_key = crate::tuning::resident_gemm::WeightKey::new(weight, scale)?;
                for &rows in widths {
                    if rows > 0 {
                        strategies
                            .entry((weight_key, rows, outputs, columns))
                            .or_insert_with(|| {
                                crate::tuning::resident_gemm::select(
                                    device,
                                    weight,
                                    scale,
                                    [rows, outputs, columns],
                                    weights.autotune,
                                    &weights.resident_tuning,
                                )
                            });
                    }
                }
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
        Ok(Self {
            strategies,
            buffers,
            fp8,
        })
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
        let ProjectionWeight::Fp4(..) = weight else {
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
        let outputs = usize::try_from(output.shape()[1]).map_err(super::super::capture::error)?;
        let mut config = self
            .strategies
            .get(&(
                crate::tuning::resident_gemm::WeightKey::new(weight, scale)
                    .map_err(super::super::capture::error)?,
                rows,
                outputs,
                columns,
            ))
            .copied()
            .unwrap_or_else(|| crate::tuning::resident_gemm::GemmConfig::baseline(rows));
        // An explicit experimental override never enters the measured table or memo.
        if std::env::var_os("INFER_CUDA_QUANT_TILE").is_some() {
            let tile = crate::constants::quant_gemm_tile(rows);
            config.tile_m = tile[0];
            config.tile_n = tile[1];
            config.main_loop = if rows > crate::constants::QUANT_GEMM_TILE[0]
                && tile[1] > crate::constants::QUANT_GEMM_TILE[1]
            {
                crate::tuning::resident_gemm::MainLoop::Prefetch8
            } else {
                crate::tuning::resident_gemm::MainLoop::Plain
            };
        }
        config.record(scope, weight, scale, &*q, &*qs, output, columns)?;
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

/// Mirrors `BatchBuilder`'s capture decisions; allocation and tuning share this exact list.
pub fn captured_widths(
    batch: usize,
    wide: usize,
    narrow: usize,
    verification: bool,
    wide_prompt: bool,
) -> Vec<usize> {
    let mut widths = vec![1];
    if verification && batch > 1 {
        widths.push(batch);
    }
    if narrow >= crate::constants::PREFILL_LANES && (narrow < wide || !wide_prompt) {
        widths.push(narrow);
    }
    if wide_prompt && wide >= crate::constants::PREFILL_LANES && !widths.contains(&wide) {
        widths.push(wide);
    }
    widths
}
