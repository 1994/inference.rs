//! One reusable BF16 staging allocation per program, shared across layers and widths.
use super::{Support, catalog};
use crate::device::{CudaDevice, device_error};
use cutile::{half::bf16, prelude::*};
use infer_core::Result;
use infer_ir::{DataflowGraph, TensorOp};
pub struct Workspace {
    pub support: Support,
}
pub(super) struct Scratch {
    pub query: Tensor<bf16>,
    pub keys: Tensor<bf16>,
    pub values: Tensor<bf16>,
    pub output: Tensor<bf16>,
    pub used: Tensor<i32>,
}
impl Scratch {
    pub(super) fn new(device: &CudaDevice, capacity: usize, lanes: usize) -> Result<Self> {
        let q = lanes * catalog::HEADS * catalog::DIM;
        let kv = capacity * catalog::KV_HEADS * catalog::DIM;
        let alloc = |n| {
            api::zeros::<bf16>(&[n])
                .sync_on(&device.stream)
                .map_err(device_error)
        };
        Ok(Self {
            query: alloc(q)?,
            keys: alloc(kv)?,
            values: alloc(kv)?,
            output: alloc(q)?,
            used: api::zeros::<i32>(&[2])
                .sync_on(&device.stream)
                .map_err(device_error)?,
        })
    }
}
pub fn eligible(
    architecture: &str,
    graph: &DataflowGraph,
    weights: &super::super::ProgramWeights,
    lanes: usize,
) -> bool {
    graph.nodes.iter().any(|node| {
        let TensorOp::Attention {
            query_heads,
            kv_heads,
            head_dim,
            window,
        } = node.op
        else {
            return false;
        };
        node.states
            .first()
            .is_some_and(|id| weights.kv_scales.contains_key(id))
            && catalog::supported(
                architecture,
                query_heads,
                kv_heads,
                head_dim,
                lanes,
                window.unwrap_or(0),
            )
    })
}
impl Workspace {
    pub fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &super::super::ProgramWeights,
        capacity: usize,
        lanes: usize,
    ) -> Result<Option<Self>> {
        if !eligible(&device.profile()?.architecture, graph, weights, lanes) {
            return Ok(None);
        }
        Ok(Some(Self {
            support: Support::load(device, capacity, lanes)?,
        }))
    }
}
