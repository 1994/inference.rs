//! Capture-time split-KV planning and reusable scratch, independent of model identity.
use super::kernels;
use crate::device::{CudaDevice, device_error};
use cutile::prelude::*;
use infer_core::{Result, TensorId};
use infer_ir::{DataflowGraph, TensorOp};
use std::collections::BTreeMap;

#[cfg(test)]
#[path = "../../../tests/unit/resident_attention_prefill_taps.rs"]
pub mod taps;

struct Buffers {
    parts: usize,
    numerator: Tensor<f32>,
    maxima: Tensor<f32>,
    sums: Tensor<f32>,
}

pub struct Workspace {
    pub(in crate::resident) oxide: Option<super::super::attention_oxide::Support>,
    #[cfg(test)]
    pub taps: taps::Taps,
    buffers: BTreeMap<(usize, usize), Buffers>,
    prefill: BTreeMap<(TensorId, usize), crate::tuning::attention::Choice>,
}

/// A hardware/window-scoped partition count gives private and pooled sequences the same
/// F32 reduction tree. Reserved KV capacity only changes storage, never arithmetic.
fn partitions(heads: usize, window: Option<usize>, multiprocessors: usize) -> usize {
    if window.is_some_and(|tokens| tokens < crate::constants::ATTENTION_SPLIT_THRESHOLD) {
        return 1;
    }
    let limit = multiprocessors
        .div_ceil(heads)
        .clamp(1, crate::constants::ATTENTION_SPLIT_MAX_PARTS);
    1 << limit.ilog2()
}

impl Workspace {
    pub(crate) fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        _capacity: usize,
    ) -> Result<Self> {
        let mut buffers = BTreeMap::new();
        let sms = usize::try_from(device.profile()?.multiprocessors)
            .map_err(|_| infer_core::Error::invalid("CUDA multiprocessor count"))?;
        for node in &graph.nodes {
            let TensorOp::Attention {
                query_heads,
                head_dim,
                window,
                ..
            } = node.op
            else {
                continue;
            };
            if query_heads == 0 || head_dim == 0 {
                return Err(infer_core::Error::invalid(
                    "attention dimensions must be positive",
                ));
            }
            let parts = partitions(query_heads, window, sms);
            if parts == 1 || buffers.contains_key(&(query_heads, head_dim)) {
                continue;
            }
            buffers.insert(
                (query_heads, head_dim),
                Buffers {
                    parts,
                    numerator: api::zeros::<f32>(&[query_heads * parts, head_dim])
                        .sync_on(&device.stream)
                        .map_err(device_error)?,
                    maxima: api::zeros::<f32>(&[query_heads * parts, 1])
                        .sync_on(&device.stream)
                        .map_err(device_error)?,
                    sums: api::zeros::<f32>(&[query_heads * parts, 1])
                        .sync_on(&device.stream)
                        .map_err(device_error)?,
                },
            );
        }
        Ok(Self {
            #[cfg(test)]
            taps: taps::Taps::default(),
            oxide: super::super::attention_oxide::Support::prepare(device, graph)?,
            buffers,
            prefill: BTreeMap::new(),
        })
    }

    pub(crate) fn prepare_prefill(
        &mut self,
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &super::super::ProgramWeights,
        lanes: usize,
    ) {
        let enabled = crate::tuning::attention::enabled();
        #[cfg(test)]
        let enabled = enabled || taps::forced_choice().is_some();
        if !enabled {
            self.prefill.retain(|(_, width), _| *width != lanes);
            return;
        }
        let mut choices = BTreeMap::new();
        for node in &graph.nodes {
            let TensorOp::Attention {
                query_heads,
                kv_heads,
                head_dim,
                window,
            } = node.op
            else {
                continue;
            };
            if !node
                .states
                .first()
                .is_some_and(|id| weights.kv_scales.contains_key(id))
            {
                continue;
            }
            let shape = crate::tuning::attention::Shape {
                heads: query_heads,
                kv_heads,
                dim: head_dim,
                lanes,
                window: window.unwrap_or(0),
            };
            let choice = *choices.entry(shape).or_insert_with(|| {
                crate::tuning::attention::select(
                    device,
                    shape,
                    weights.autotune,
                    &weights.resident_tuning,
                )
            });
            #[cfg(test)]
            let choice = taps::forced_choice().unwrap_or(choice);
            self.prefill.insert((node.outputs[0], lanes), choice);
        }
    }

    pub(crate) fn prefill_query_tile(
        &self,
        output: TensorId,
        lanes: usize,
        fp8: bool,
    ) -> Option<usize> {
        if !fp8 {
            return None;
        }
        self.prefill
            .get(&(output, lanes))
            .and_then(|choice| choice.query_tile())
    }

    pub(crate) fn supports(&self, heads: usize, dim: usize) -> bool {
        self.buffers.contains_key(&(heads, dim))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Attention binding includes separate KV, metadata and scale tensors"
    )]
    pub(crate) fn record<E: DType>(
        &mut self,
        scope: &Scope,
        output: &mut Tensor<f32>,
        q: &TensorView<'_, f32>,
        keys: &Tensor<E>,
        values: &Tensor<E>,
        metadata: &Tensor<i32>,
        table: &Tensor<i32>,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        window: i32,
        scales: [f32; 2],
    ) -> std::result::Result<(), DeviceError> {
        let work = self
            .buffers
            .get_mut(&(heads, dim))
            .ok_or_else(|| super::super::capture::error("split KV workspace"))?;
        let parts = work.parts;
        scope.record(
            kernels::partial(
                (&mut work.numerator).partition([1, dim]),
                (&mut work.maxima).partition([1, 1]),
                (&mut work.sums).partition([1, 1]),
                q,
                keys,
                values,
                metadata,
                table,
                window,
                scales[0],
                scales[1],
            )
            .generics(vec![
                E::DTYPE.as_str().into(),
                dim.to_string(),
                (heads / kv_heads).to_string(),
                parts.to_string(),
                crate::constants::KV_BLOCK_TOKENS.to_string(),
            ]),
        )?;
        scope.record(
            kernels::merge(
                output.partition([1, dim]),
                &work.numerator.view(&[heads, parts, dim])?,
                &work.maxima.view(&[heads, parts])?,
                &work.sums.view(&[heads, parts])?,
            )
            .generics(vec![dim.to_string(), parts.to_string()]),
        )?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/resident_attention_decode_workspace.rs"]
mod tests;
