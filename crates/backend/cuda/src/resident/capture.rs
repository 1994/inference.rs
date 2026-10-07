use super::{ActivationArena, ProgramWeights, attention::attention, kernels::aux};
use crate::strategy::LinearTiling;
use cutile::prelude::*;
use infer_core::TensorId;
use infer_ir::{TensorNode, TensorOp};
use std::collections::BTreeMap;
use std::sync::Arc;

/// How activation slots are bound for one captured node.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CaptureMode {
    /// Width 1 and verification widths: every slot is one flat `[size]` tensor.
    Flat,
    /// Prefill width 32: every slot is one `[32, size]` tensor used as a whole.
    Batched,
    /// Contiguous prompt rows with [position, active rows, state offset, unused] metadata.
    Prefill,
    /// Prefill width 32, one lane at a time: slots stay `[32, size]`, kernels
    /// with per-lane metadata or ordered state read and write single rows.
    Row(usize),
}

pub(super) struct Capture<'a> {
    pub scope: &'a Scope,
    pub arena: &'a mut ActivationArena,
    pub weights: &'a ProgramWeights,
    pub nvfp4: &'a mut super::nvfp4_gemm::Workspace,
    pub attention: &'a mut super::attention_decode::Workspace,
    pub fp8_states: &'a mut super::fp8_cache::Fp8Caches,
    pub states: &'a mut BTreeMap<TensorId, Vec<Tensor<f32>>>,
    pub metadata: &'a Tensor<i32>,
    pub external: &'a Tensor<f32>,
    pub capacity: usize,
    pub fusion: &'a mut Option<super::program::FusionWorkspace>,
    pub mode: CaptureMode,
}

pub(super) fn error(message: impl std::fmt::Display) -> DeviceError {
    DeviceError::Launch(message.to_string())
}

fn identity(tensor: &Tensor<f32>) -> Result<TensorView<'_, f32>, DeviceError> {
    let shape = tensor
        .shape()
        .iter()
        .map(|&dim| usize::try_from(dim).map_err(error))
        .collect::<Result<Vec<_>, _>>()?;
    tensor.view(&shape).map_err(error)
}

impl Capture<'_> {
    /// Activation inputs resolve per mode; weight constants keep their own shape.
    pub fn input(&self, id: TensorId) -> Result<TensorView<'_, f32>, DeviceError> {
        if let Some(value) = self.weights.constants.get(&id) {
            return identity(value);
        }
        let tensor = self.arena.get(id).map_err(error)?;
        match self.mode {
            CaptureMode::Flat => identity(tensor),
            CaptureMode::Batched | CaptureMode::Prefill => {
                tensor.view(&[tensor.size()]).map_err(error)
            }
            CaptureMode::Row(lane) => self.arena.row(id, lane).map_err(error),
        }
    }

    pub fn record(&mut self, node: &TensorNode) -> Result<(), DeviceError> {
        for (part, id) in node.outputs.iter().enumerate() {
            let index = self.arena.slot(*id).map_err(error)?;
            let output = self.arena.buffers[index]
                .take()
                .ok_or_else(|| error("missing output buffer"))?;
            let restore = output
                .shape()
                .iter()
                .map(|&dim| usize::try_from(dim).map_err(error))
                .collect::<Result<Vec<_>, _>>()?;
            let size = output.size();
            let output = self.operation(node, output.reshape(&[size])?, part)?;
            self.arena.buffers[index] = Some(output.reshape(&restore)?);
        }
        Ok(())
    }

    /// Row-mode twin of [`Self::record`]: kernels write one lane of each batched
    /// slot through an owned row tensor; slots return unaltered.
    pub fn record_row(&mut self, node: &TensorNode, lane: usize) -> Result<(), DeviceError> {
        if self.mode != CaptureMode::Row(lane) {
            return Err(error("row capture mode mismatch"));
        }
        for (part, id) in node.outputs.iter().enumerate() {
            let index = self.arena.slot(*id).map_err(error)?;
            let slot = self.arena.buffers[index]
                .take()
                .ok_or_else(|| error("missing output buffer"))?;
            let (row, shared) =
                super::arena::row_tensor(slot, self.arena.lanes(), lane).map_err(error)?;
            let _ = self.operation(node, row, part)?;
            let slot = Arc::try_unwrap(shared)
                .map_err(|_| error("activation row tensor still aliased"))?;
            self.arena.buffers[index] = Some(slot);
        }
        Ok(())
    }

    fn operation(
        &mut self,
        node: &TensorNode,
        mut output: Tensor<f32>,
        part: usize,
    ) -> Result<Tensor<f32>, DeviceError> {
        match &node.op {
            TensorOp::Embedding => {
                self.embedding(node, &mut output)?;
            }
            TensorOp::Linear => {
                output = self.linear(node, output)?;
            }
            TensorOp::Silu | TensorOp::Sigmoid => {
                self.scope.record(
                    aux::unary(
                        (&mut output).partition([crate::constants::AUX_KERNEL_TILE]),
                        &self.input(node.inputs[0])?,
                    )
                    .generics(vec![i32::from(node.op == TensorOp::Sigmoid).to_string()]),
                )?;
            }
            TensorOp::Add | TensorOp::Multiply => {
                self.scope.record(
                    aux::binary(
                        (&mut output).partition([crate::constants::AUX_KERNEL_TILE]),
                        &self.input(node.inputs[0])?,
                        &self.input(node.inputs[1])?,
                    )
                    .generics(vec![i32::from(node.op == TensorOp::Multiply).to_string()]),
                )?;
            }
            TensorOp::Norm {
                head_dim,
                epsilon,
                offset,
            } => {
                output = self.norm(node, output, *head_dim, *epsilon, *offset, false)?;
            }
            TensorOp::GatedNorm { head_dim, epsilon } => {
                output = self.norm(node, output, *head_dim, *epsilon, 0.0, true)?;
            }
            TensorOp::Split { widths, .. } => {
                let width = widths[part];
                let offset: usize = widths[..part].iter().sum();
                if !width.is_power_of_two() || !offset.is_multiple_of(width) {
                    return Err(error("unsupported device split alignment"));
                }
                let stride: usize = widths.iter().sum();
                let heads = output.size() / width;
                output = output.reshape(&[heads, width])?;
                self.scope.record(
                    aux::split(
                        (&mut output).partition([1, width]),
                        &self.input(node.inputs[0])?.view(&[heads, stride])?,
                        i32::try_from(offset / width).map_err(error)?,
                    )
                    .generics(vec![stride.to_string(), width.to_string()]),
                )?;
            }
            TensorOp::Rope { .. } => {
                output = self.record_rope(node, output)?;
            }
            TensorOp::Conv { .. } | TensorOp::Delta { .. } | TensorOp::Attention { .. } => {
                output = self.record_state(node, output)?;
            }
        }
        Ok(output)
    }

    /// Width-1 and verification-width projection: per-lane input tensor.
    /// Prefill width 32 linear nodes never reach here; see `prefill_projection`.
    fn linear(
        &mut self,
        node: &TensorNode,
        mut output: Tensor<f32>,
    ) -> Result<Tensor<f32>, DeviceError> {
        let input = self.weights.constants.get(&node.inputs[0]).map_or_else(
            || self.arena.get(node.inputs[0]).map_err(error),
            |value| Ok(value.as_ref()),
        )?;
        let weights = self
            .weights
            .projections
            .get(&node.inputs[1])
            .ok_or_else(|| error("missing projection"))?;
        if let Some(scale) = self.weights.activation_quantization(node.inputs[1]) {
            let size = output.size();
            let mut matrix = output.reshape(&[1, size])?;
            if !super::nvfp4_gemm::record(
                self.scope,
                self.nvfp4,
                weights,
                Some(scale),
                &input.view(&[1, input.size()]).map_err(error)?,
                &mut matrix,
                input.size(),
            )? {
                return Err(error(
                    "activation quantization and weight encoding disagree",
                ));
            }
            return Ok(matrix.reshape(&[size])?);
        }
        weights.record(
            self.scope,
            &mut output,
            input,
            input.size(),
            self.weights.tiling.get(&node.inputs[1]).copied().unwrap_or(
                LinearTiling::new(
                    crate::constants::DEFAULT_TILE_ROWS,
                    crate::constants::DEFAULT_TILE_COLUMNS,
                )
                .map_err(error)?,
            ),
        )?;
        Ok(output)
    }

    fn embedding(
        &mut self,
        node: &TensorNode,
        output: &mut Tensor<f32>,
    ) -> Result<(), DeviceError> {
        let weight = self
            .weights
            .embeddings
            .get(&node.inputs[0])
            .ok_or_else(|| error("missing device embedding"))?;
        let dimension = output.size();
        if let Some(fusion) = &self.weights.fusion {
            let work = self
                .fusion
                .as_mut()
                .ok_or_else(|| error("missing fusion workspace"))?;
            self.scope.record(
                attention::embedding(
                    (&mut work.embedding).partition([crate::constants::AUX_KERNEL_TILE]),
                    weight,
                    self.external,
                    self.metadata,
                )
                .generics(vec![dimension.to_string(), "0".into()]),
            )?;
            let mut normalized = work
                .normalized
                .take()
                .ok_or_else(|| error("fusion buffer"))?
                .reshape(&[2, dimension])?;
            self.scope.record(
                aux::fusion_norm(
                    (&mut normalized).partition([1, dimension.next_power_of_two()]),
                    &work.embedding,
                    self.external,
                    &fusion.norms,
                    fusion.epsilon,
                    fusion.offset,
                )
                .generics(vec![
                    dimension.to_string(),
                    dimension.next_power_of_two().to_string(),
                ]),
            )?;
            let normalized = normalized.reshape(&[2 * dimension])?;
            fusion.projection.record(
                self.scope,
                output,
                &normalized,
                2 * dimension,
                fusion.tiling,
            )?;
            work.normalized = Some(normalized);
        } else {
            self.scope.record(
                attention::embedding(
                    output.partition([crate::constants::AUX_KERNEL_TILE]),
                    weight,
                    self.external,
                    self.metadata,
                )
                .generics(vec![dimension.to_string(), "1".into()]),
            )?;
        }
        Ok(())
    }

    fn norm(
        &self,
        node: &TensorNode,
        output: Tensor<f32>,
        dimension: usize,
        epsilon: f32,
        offset: f32,
        gated: bool,
    ) -> Result<Tensor<f32>, DeviceError> {
        if dimension == 0 || !output.size().is_multiple_of(dimension) {
            return Err(error("norm dimensions"));
        }
        let heads = output.size() / dimension;
        let mut output = output.reshape(&[heads, dimension])?;
        let x = self.input(node.inputs[0])?;
        let input = x.view(&[heads, dimension])?;
        let g = self.input(node.inputs[usize::from(gated)])?;
        let gate = g.view(&[heads, dimension])?;
        let weight = self.input(node.inputs[if gated { 2 } else { 1 }])?;
        self.scope.record(
            aux::norm(
                (&mut output).partition([1, dimension.next_power_of_two()]),
                &input,
                &weight,
                &gate,
                epsilon,
                offset,
            )
            .generics(vec![
                dimension.to_string(),
                dimension.next_power_of_two().to_string(),
                i32::from(gated).to_string(),
            ]),
        )?;
        Ok(output)
    }
}
