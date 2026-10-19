//! Per-token/head F32 normalization and gating shared across independent state columns.
use super::{ORDERED_DIM, ORDERED_WIDTHS, chunk, kernels, oxide};
use crate::{
    device::{CudaDevice, device_error},
    resident::ProgramWeights,
};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_ir::{DataflowGraph, TensorOp};
use std::collections::{BTreeMap, BTreeSet};

const INPUT_BINDINGS: usize = 5;
type Geometry = (usize, usize, usize);
pub(super) type Buffers = BTreeMap<Geometry, Inputs>;

pub(super) struct Inputs {
    normalized: Tensor<f32>,
    betas: Tensor<f32>,
    decays: Tensor<f32>,
    rust: Option<oxide::Support>,
    chunk: Option<chunk::Scratch>,
}

fn geometries(graph: &DataflowGraph, weights: &ProgramWeights, lanes: usize) -> BTreeSet<Geometry> {
    let mut shapes = BTreeSet::new();
    if !ORDERED_WIDTHS.contains(&lanes) {
        return shapes;
    }
    for node in &graph.nodes {
        if let TensorOp::Delta {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        } = node.op
            && key_dim == ORDERED_DIM
            && value_dim == key_dim
            && key_heads > 0
            && value_heads > 0
            && value_heads.is_multiple_of(key_heads)
            && node.inputs[..crate::constants::GDN_PROJECTED_INPUTS]
                .iter()
                .all(|id| !weights.constants.contains_key(id))
        {
            shapes.insert((key_heads, value_heads, key_dim));
        }
    }
    shapes
}

pub(super) fn allocate(
    device: &CudaDevice,
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    lanes: usize,
) -> Result<Buffers> {
    geometries(graph, weights, lanes)
        .into_iter()
        .map(|shape| {
            let mut inputs = Inputs::new(device, shape, lanes)?;
            inputs.rust = oxide::Support::select(
                device,
                [shape.0, shape.1, shape.2, lanes],
                weights.autotune,
                &weights.resident_tuning,
            )?;
            Ok((shape, inputs))
        })
        .collect()
}

pub(super) fn required_bytes(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    lanes: usize,
) -> Result<u64> {
    geometries(graph, weights, lanes)
        .into_iter()
        .try_fold(0u64, |total, (kh, vh, dim)| {
            let chunked = chunk::enabled(dim, lanes);
            let extra = if chunked {
                chunk::Scratch::bytes((kh, vh, dim), lanes)
            } else {
                0
            };
            let normalized_rows = if chunked { 1 } else { lanes };
            let bytes = kh
                .checked_mul(2)
                .and_then(|x| x.checked_mul(dim))
                .and_then(|x| x.checked_mul(normalized_rows))
                .and_then(|x| vh.checked_mul(2 * lanes).and_then(|gates| x.checked_add(gates)))
                .and_then(|x| x.checked_mul(crate::constants::F32_BYTES))
                .and_then(|x| x.checked_add(extra))
                .and_then(|x| u64::try_from(x).ok())
                .ok_or_else(|| Error::invalid("CUDA recurrent preparation size overflow"))?;
            total
                .checked_add(bytes)
                .ok_or_else(|| Error::invalid("CUDA recurrent preparation size overflow"))
        })
}

impl Inputs {
    pub(super) fn new(device: &CudaDevice, (kh, vh, dim): Geometry, lanes: usize) -> Result<Self> {
        let chunked = chunk::enabled(dim, lanes);
        // The chunked rule normalizes into its own BF16 rows; the F32 rows only feed the
        // token-serial kernels.
        let normalized_rows = if chunked { 1 } else { lanes };
        Ok(Self {
            rust: None,
            chunk: chunked
                .then(|| chunk::Scratch::new(device, (kh, vh, dim), lanes))
                .transpose()?,
            normalized: api::zeros::<f32>(&[normalized_rows, 2 * kh, dim])
                .sync_on(&device.stream)
                .map_err(device_error)?,
            betas: api::zeros::<f32>(&[lanes, vh])
                .sync_on(&device.stream)
                .map_err(device_error)?,
            decays: api::zeros::<f32>(&[lanes, vh])
                .sync_on(&device.stream)
                .map_err(device_error)?,
        })
    }

    pub(super) fn record(
        &mut self,
        scope: &Scope,
        [qkv_storage, beta, alpha, a_log, bias]: [&Tensor<f32>; INPUT_BINDINGS],
        [state, output]: [&mut Tensor<f32>; 2],
        metadata: &Tensor<i32>,
        (kh, vh, dim, lanes, block): (usize, usize, usize, usize, usize),
    ) -> std::result::Result<(), DeviceError> {
        let qkv = qkv_storage.view(&[lanes, 2 * kh + vh, dim])?;
        let beta = beta.view(&[lanes, vh])?;
        let alpha = alpha.view(&[lanes, vh])?;
        let a_log = a_log.view(&[vh])?;
        let bias = bias.view(&[vh])?;
        if self.chunk.is_none() {
            scope.record(
                kernels::normalize_shared((&mut self.normalized).partition([1, 1, dim]), &qkv)
                    .generics(vec![kh.to_string(), dim.to_string()]),
            )?;
        }
        scope.record(kernels::gates_shared(
            (&mut self.betas).partition([1, 1]),
            (&mut self.decays).partition([1, 1]),
            &beta,
            &alpha,
            &a_log,
            &bias,
        ))?;
        if let Some(mut scratch) = self.chunk.take() {
            let result = chunk::record(
                scope,
                &mut scratch,
                &qkv,
                (&self.betas, &self.decays),
                [state, output],
                metadata,
                (kh, vh, dim, lanes),
            );
            self.chunk = Some(scratch);
            return result;
        }
        if let Some(rust) = &self.rust {
            return rust.record(
                scope,
                oxide::Bindings {
                    state,
                    output,
                    qkv: qkv_storage,
                    normalized: &self.normalized,
                    beta: &self.betas,
                    decay: &self.decays,
                    metadata,
                },
                lanes,
            );
        }
        scope.record(
            kernels::delta_precomputed(
                state.partition([1, dim, block]),
                output.partition([1, lanes, block]),
                &qkv,
                &self.normalized,
                &self.betas,
                &self.decays,
                metadata,
            )
            .generics(vec![
                kh.to_string(),
                vh.to_string(),
                dim.to_string(),
                lanes.to_string(),
                block.to_string(),
            ]),
        )?;
        Ok(())
    }
}
