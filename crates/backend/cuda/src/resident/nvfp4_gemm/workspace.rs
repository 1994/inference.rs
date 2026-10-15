//! Per-program activation quantization buffers, retained as long as captured graphs.
use super::{
    kernels,
    quant_reuse::{CaptureQuantReuse, QuantKey, QuantKind, QuantPairRole},
};
use crate::{
    device::{CudaDevice, device_error, vendor_fp8},
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
    #[cfg(test)]
    quant_occupancy: i32,
    quant_reuse: CaptureQuantReuse,
    strategies: BTreeMap<
        (crate::tuning::resident_gemm::WeightKey, usize, usize, usize),
        crate::tuning::resident_gemm::GemmConfig,
    >,
    buffers: BTreeMap<(usize, usize), Quantized>,
    quant_strategies: BTreeMap<(usize, usize), i32>,
    fp8_strategies: BTreeMap<
        (crate::tuning::fp8_gemm::WeightKey, usize, usize, usize),
        crate::tuning::fp8_gemm::Choice,
    >,
    /// Lt handle, descriptors and original weight allocations outlive every program capture.
    vendor_fp8: vendor_fp8::Support,
    /// Explicit diagnostic override; production automatic choices still require exact gates.
    vendor_diagnostic: bool,
    /// Keyed by `(rows, columns, scale_columns)`: per-channel and block-scaled FP8 share an
    /// operand geometry but need different activation-scale extents, so geometry alone would let
    /// whichever projection is visited first decide the layout for the other.
    fp8: BTreeMap<Fp8Key, Fp8Buffers>,
}

impl Workspace {
    #[cfg(test)]
    pub(crate) const fn set_quant_occupancy(&mut self, value: i32) {
        self.quant_occupancy = value;
    }
    /// Start a fresh graph-local permission epoch; no device memory is changed.
    pub(crate) const fn begin_quant_capture(&mut self) {
        self.quant_reuse.begin();
    }
    pub(crate) fn enter_quant_node(&mut self, index: usize, role: Option<QuantPairRole>) {
        self.quant_reuse.enter(index, role);
    }
    pub(crate) const fn end_quant_capture(&mut self) {
        self.quant_reuse.end();
    }
    pub(crate) const fn clear_quant_reuse(&mut self) {
        self.quant_reuse.clear();
    }
    #[cfg(test)]
    pub(crate) const fn set_quant_reuse_enabled(&mut self, enabled: bool) {
        self.quant_reuse.set_enabled(enabled);
    }
    #[cfg(test)]
    pub(crate) const fn quant_reuse_stats(&self) -> (usize, usize) {
        let stats = self.quant_reuse.stats();
        (stats.recorded, stats.reused)
    }
    #[cfg(test)]
    pub(crate) const fn quant_reuse_totals(&self) -> (usize, usize) {
        let stats = self.quant_reuse.total();
        (stats.recorded, stats.reused)
    }
    #[cfg(test)]
    pub(crate) fn quantized_fp8(
        &self,
        rows: usize,
        columns: usize,
    ) -> Option<(&Tensor<f8e4m3fn>, &Tensor<f32>)> {
        self.fp8
            .get(&(rows, columns, 1))
            .map(|(codes, scales)| (codes, scales))
    }
    #[cfg(test)]
    pub(crate) fn quantized_fp4(
        &self,
        rows: usize,
        columns: usize,
    ) -> Option<(&Tensor<f4e2m1fnx2>, &Tensor<f8e4m3fn>)> {
        self.buffers
            .get(&(rows, columns))
            .map(|(codes, scales)| (codes, scales))
    }

    /// Test evidence for prepared strategies, grouped by shape/config; this is not a replay trace.
    #[cfg(test)]
    pub(crate) fn prepared_configs(&self) -> serde_json::Value {
        let mut groups = BTreeMap::<(&str, usize, usize, usize, String), usize>::new();
        for ((_, rows, outputs, columns), config) in &self.strategies {
            *groups
                .entry(("nvfp4", *rows, *outputs, *columns, format!("{config:?}")))
                .or_default() += 1;
        }
        for ((_, rows, outputs, columns), config) in &self.fp8_strategies {
            *groups
                .entry((
                    "fp8-channel",
                    *rows,
                    *outputs,
                    *columns,
                    format!("{config:?}"),
                ))
                .or_default() += 1;
        }
        serde_json::json!(
            groups
                .into_iter()
                .map(|((mode, rows, outputs, columns, config), count)| {
                    serde_json::json!({"mode":mode, "m":rows, "n":outputs, "k":columns,
                "prepared_config":config, "weight_count":count})
                })
                .collect::<Vec<_>>()
        )
    }

    pub(crate) fn program(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        geometry: crate::resident::program::ProgramGeometry,
    ) -> Result<Self> {
        let widths = geometry.widths();
        Self::new(device, graph, weights, &widths)
    }

    pub(crate) fn ensure_width(
        &mut self,
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        width: usize,
    ) -> Result<()> {
        self.extend(device, graph, weights, &[width])
    }

    pub(crate) fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        widths: &[usize],
    ) -> Result<Self> {
        let mut workspace = Self {
            #[cfg(test)]
            quant_occupancy: -1,
            quant_reuse: CaptureQuantReuse::default(),
            strategies: BTreeMap::new(),
            buffers: BTreeMap::new(),
            quant_strategies: BTreeMap::new(),
            fp8: BTreeMap::new(),
            fp8_strategies: BTreeMap::new(),
            vendor_fp8: vendor_fp8::Support::new_with(
                device,
                crate::tuning::fp8_gemm::vendor_autotune_enabled() || vendor_diagnostic_enabled(),
            ),
            vendor_diagnostic: vendor_diagnostic_enabled(),
        };
        workspace.extend(device, graph, weights, widths)?;
        Ok(workspace)
    }

    fn extend(
        &mut self,
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        widths: &[usize],
    ) -> Result<()> {
        let _probe_inputs = crate::tuning::fp8_gemm::probe_inputs::Scope::new(device)?;
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
                self.allocate_fp8(
                    device,
                    weights.projections.get(&node.inputs[1]),
                    columns,
                    widths,
                )?;
                self.prepare_channel(
                    device,
                    graph,
                    weights,
                    node.inputs[1],
                    node.outputs[0],
                    columns,
                    widths,
                )?;
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
                        self.strategies
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
                if rows == 0 || self.buffers.contains_key(&(rows, columns)) {
                    continue;
                }
                self.quant_strategies.insert(
                    (rows, columns),
                    crate::tuning::nvfp4_quant::select(
                        device,
                        rows,
                        columns,
                        weights.autotune,
                        &weights.resident_tuning,
                    ),
                );
                self.buffers.insert(
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
        Ok(())
    }

    fn allocate_fp8(
        &mut self,
        device: &CudaDevice,
        weight: Option<&ProjectionWeight>,
        columns: usize,
        widths: &[usize],
    ) -> Result<()> {
        let block = matches!(weight, Some(ProjectionWeight::Fp8Block(..)));
        let scale_columns = if block {
            columns / crate::constants::FP8_BLOCK_COLUMNS
        } else {
            1
        };
        for &rows in widths {
            if rows > 0 && !self.fp8.contains_key(&(rows, columns, scale_columns)) {
                self.fp8.insert(
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
        Ok(())
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "Preparation carries graph weight/output identities and the captured widths before any CUDA scope starts"
    )]
    fn prepare_channel(
        &mut self,
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        weight_id: infer_core::TensorId,
        output_id: infer_core::TensorId,
        columns: usize,
        widths: &[usize],
    ) -> Result<()> {
        if let Some(weight @ ProjectionWeight::Fp8(codes, scales)) =
            weights.projections.get(&weight_id)
        {
            self.vendor_fp8.retain_channels(codes, scales);
            let outputs = graph
                .tensors
                .iter()
                .find(|tensor| tensor.id == output_id)
                .ok_or_else(|| Error::invariant("FP8 output tensor"))?
                .elements()?;
            let weight_key = crate::tuning::fp8_gemm::WeightKey::new(weight)?;
            for &rows in widths {
                // Prompt captures omit the batched vocabulary head and use a single final lane.
                if rows == 0
                    || (rows >= crate::constants::PREFILL_LANES && Some(output_id) == graph.logits)
                {
                    continue;
                }
                let key = (weight_key, rows, outputs, columns);
                if self.fp8_strategies.contains_key(&key) {
                    continue;
                }
                let choice = if self.vendor_diagnostic {
                    let q = &self
                        .fp8
                        .get(&(rows, columns, 1))
                        .ok_or_else(|| Error::invariant("FP8 diagnostic activation"))?
                        .0;
                    let out = api::zeros::<f32>(&[rows, outputs])
                        .sync_on(&device.stream)
                        .map_err(device_error)?;
                    if let Err(error) = self.vendor_fp8.prepare(q, codes, &out) {
                        tracing::warn!(target: "infer::tuning", %error, "FP8 diagnostic preparation rejected: native fallback");
                    }
                    crate::tuning::fp8_gemm::Choice::baseline(rows)
                } else {
                    crate::tuning::fp8_gemm::select(
                        device,
                        &mut self.vendor_fp8,
                        weight,
                        [rows, outputs, columns],
                        weights.autotune,
                        &weights.resident_tuning,
                    )
                };
                self.fp8_strategies.insert(key, choice);
            }
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Quantization records the exact view pointer alongside its original projection contract"
    )]
    pub(super) fn record_input(
        &mut self,
        scope: &Scope,
        weight: &ProjectionWeight,
        scale: f32,
        input: &TensorView<'_, f32>,
        output: &mut Tensor<f32>,
        columns: usize,
        input_pointer: Option<u64>,
    ) -> std::result::Result<(), DeviceError> {
        let ProjectionWeight::Fp4(..) = weight else {
            return Err(super::super::capture::error("FP4 workspace weight"));
        };
        let rows = output.size()
            / usize::try_from(output.shape()[1]).map_err(super::super::capture::error)?;
        let occupancy = self
            .quant_strategies
            .get(&(rows, columns))
            .copied()
            .unwrap_or(0);
        #[cfg(test)]
        let occupancy = if self.quant_occupancy >= 0 {
            self.quant_occupancy
        } else {
            occupancy
        };
        #[cfg(test)]
        let occupancy = super::quant_discovery::model_occupancy(occupancy);
        let options = crate::tuning::nvfp4_quant::compile_options(occupancy);
        let (q, qs) = self
            .buffers
            .get_mut(&(rows, columns))
            .ok_or_else(|| super::super::capture::error("FP4 quantization workspace missing"))?;
        let key = quant_key(
            input_pointer,
            input,
            output,
            rows,
            columns,
            QuantKind::Fp4(scale.to_bits()),
            (&*q, &*qs),
        );
        let (reuse, ticket) = self.quant_reuse.prepare(key);
        let record = || {
            if !reuse {
                scope.record(
                    kernels::quantize(
                        (&mut *q).partition(crate::constants::NVFP4_QUANT_CODES_TILE),
                        (&mut *qs).partition(crate::constants::NVFP4_QUANT_SCALES_TILE),
                        input,
                        scale,
                    )
                    .generics(vec![columns.to_string()])
                    .compile_options(options),
                )?;
            }
            let outputs =
                usize::try_from(output.shape()[1]).map_err(super::super::capture::error)?;
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
        };
        let result = record();
        self.quant_reuse.finish(ticket, result.is_ok());
        result
    }
    pub(super) fn record_fp8_input(
        &mut self,
        scope: &Scope,
        weight: &ProjectionWeight,
        input: &TensorView<'_, f32>,
        output: &mut Tensor<f32>,
        columns: usize,
        input_pointer: Option<u64>,
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
        let scale_columns = fp8_scale_columns(weight, columns);
        let (q, qs) = self
            .fp8
            .get_mut(&(rows, columns, scale_columns))
            .ok_or_else(|| error("FP8 quantization workspace missing"))?;
        let kind = matches!(weight, ProjectionWeight::Fp8(..)).then_some(QuantKind::Fp8Token);
        let key = kind.and_then(|kind| {
            quant_key(
                input_pointer,
                input,
                output,
                rows,
                columns,
                kind,
                (&*q, &*qs),
            )
        });
        let (reuse, ticket) = self.quant_reuse.prepare(key);
        let mut record = || {
            let tile = crate::constants::quant_gemm_tile(rows);
            let generics = [columns, tile[0], tile[1]]
                .map(|value| value.to_string())
                .to_vec();
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
                    if !reuse {
                        super::super::fp8_quant_reduction::record(scope, q, qs, input, columns)?;
                    }
                    if self.vendor_diagnostic
                        && self
                            .vendor_fp8
                            .record(scope, &*q, w, output, &*qs, channel_scale)?
                    {
                        return Ok(());
                    }
                    let outputs = usize::try_from(output.shape()[1]).map_err(error)?;
                    let choice = self
                        .fp8_strategies
                        .get(&(
                            crate::tuning::fp8_gemm::WeightKey::new(weight).map_err(error)?,
                            rows,
                            outputs,
                            columns,
                        ))
                        .copied()
                        .unwrap_or_else(|| crate::tuning::fp8_gemm::Choice::baseline(rows));
                    choice.record(
                        scope,
                        &self.vendor_fp8,
                        &*q,
                        w,
                        output,
                        &*qs,
                        channel_scale,
                        columns,
                    )?;
                }
                _ => return Err(error("FP8 workspace weight")),
            }
            Ok(())
        };
        let result = record();
        self.quant_reuse.finish(ticket, result.is_ok());
        result
    }
}

const fn fp8_scale_columns(weight: &ProjectionWeight, columns: usize) -> usize {
    match weight {
        ProjectionWeight::Fp8Block(..) => columns / crate::constants::FP8_BLOCK_COLUMNS,
        _ => 1,
    }
}

/// Only the certified whole-matrix paths supply a pointer; sliced/legacy views stay separate.
fn quant_key<Q: DType, S: DType>(
    input_pointer: Option<u64>,
    input: &TensorView<'_, f32>,
    output: &Tensor<f32>,
    rows: usize,
    columns: usize,
    kind: QuantKind,
    buffers: (&Tensor<Q>, &Tensor<S>),
) -> Option<QuantKey> {
    let input_pointer = input_pointer.filter(|pointer| *pointer != 0)?;
    let shape: [i32; 2] = input.shape().try_into().ok()?;
    let strides: [i32; 2] = input.strides().try_into().ok()?;
    if shape != [i32::try_from(rows).ok()?, i32::try_from(columns).ok()?]
        || strides != [i32::try_from(columns).ok()?, 1]
    {
        return None;
    }
    let input_end = input_pointer
        .checked_add(u64::try_from(input.size().checked_mul(size_of::<f32>())?).ok()?)?;
    let output_pointer = output.device_pointer().cu_deviceptr();
    let output_end = output_pointer
        .checked_add(u64::try_from(output.size().checked_mul(size_of::<f32>())?).ok()?)?;
    if input_pointer < output_end && output_pointer < input_end {
        return None;
    }
    Some(QuantKey {
        input_pointer,
        rows,
        columns,
        shape,
        strides,
        kind,
        codes_pointer: buffers.0.device_pointer().cu_deviceptr(),
        scales_pointer: buffers.1.device_pointer().cu_deviceptr(),
    })
}

/// Independent Lt diagnostics are confined to test executables, never production serving.
#[cfg(not(test))]
const fn vendor_diagnostic_enabled() -> bool {
    false
}
#[cfg(test)]
fn vendor_diagnostic_enabled() -> bool {
    std::env::var("INFER_CUDA_VENDOR_FP8").is_ok_and(|value| value == "1" || value == "true")
}
