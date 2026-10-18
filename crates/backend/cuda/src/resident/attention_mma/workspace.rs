//! One reusable 16-bit staging allocation per program, shared across layers and widths.
use super::{Arithmetic, Support, catalog};
use crate::{
    AttentionProfile,
    device::{CudaDevice, device_error},
};
use cuda_core::sys::CUdeviceptr;
use cutile::{
    half::{bf16, f16},
    prelude::*,
};
use infer_core::Result;
use infer_ir::{DataflowGraph, TensorOp};

pub struct Workspace {
    pub support: Support,
}
/// 16-bit Q/K/V/O staging plus the two GPU metadata words. BF16 and FP16 use separate typed
/// allocations so the bridge conversion, AOT kernel and readout all share one dtype.
pub(super) struct Buffers<T: DType> {
    pub query: Tensor<T>,
    pub keys: Tensor<T>,
    pub values: Tensor<T>,
    pub output: Tensor<T>,
    pub used: Tensor<i32>,
}
impl<T: DType> Buffers<T> {
    fn new(device: &CudaDevice, capacity: usize, lanes: usize) -> Result<Self> {
        let q = lanes * catalog::HEADS * catalog::DIM;
        let kv = capacity * catalog::KV_HEADS * catalog::DIM;
        let alloc = |n| {
            api::zeros::<T>(&[n])
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
    fn pointers(&self) -> [CUdeviceptr; catalog::OPERANDS] {
        let used = self.used.device_pointer().cu_deviceptr();
        [
            self.query.device_pointer().cu_deviceptr(),
            self.keys.device_pointer().cu_deviceptr(),
            self.values.device_pointer().cu_deviceptr(),
            self.output.device_pointer().cu_deviceptr(),
            used,
            used + size_of::<i32>() as u64,
        ]
    }
    fn retain(&mut self, ctx: &ExecutionContext) -> std::result::Result<(), DeviceError> {
        for tensor in [
            &mut self.query,
            &mut self.keys,
            &mut self.values,
            &mut self.output,
        ] {
            KernelOutputStored::retain(&tensor.partition([1]), ctx)?;
        }
        KernelOutputStored::retain(&(&mut self.used).partition([1]), ctx)?;
        Ok(())
    }
}
pub(super) enum Scratch {
    Bf16(Buffers<bf16>),
    F16(Buffers<f16>),
}
impl Scratch {
    pub(super) fn new(
        device: &CudaDevice,
        capacity: usize,
        lanes: usize,
        arithmetic: Arithmetic,
    ) -> Result<Self> {
        Ok(match arithmetic {
            Arithmetic::Bf16 => Self::Bf16(Buffers::new(device, capacity, lanes)?),
            Arithmetic::F16 => Self::F16(Buffers::new(device, capacity, lanes)?),
        })
    }
    pub(super) fn pointers(&self) -> [CUdeviceptr; catalog::OPERANDS] {
        match self {
            Self::Bf16(buffers) => buffers.pointers(),
            Self::F16(buffers) => buffers.pointers(),
        }
    }
    pub(super) fn retain(
        &mut self,
        ctx: &ExecutionContext,
    ) -> std::result::Result<(), DeviceError> {
        match self {
            Self::Bf16(buffers) => buffers.retain(ctx),
            Self::F16(buffers) => buffers.retain(ctx),
        }
    }
}
/// One pure decision shared by load-time tuning and admission pricing.
///
/// The draft proposal program keeps its existing BF16 behavior. A target program is eligible
/// only when the operator selected an experimental profile and the node's KV cache is
/// FP8-packed. The FP16 profile additionally requires the validated SM120 artifact and geometry.
/// Target F32-KV attention, the canonical profile and unsupported geometry fall back to their
/// existing paths. This helper does not allocate, load a module or inspect a device, so tuning
/// and the memory quote cannot disagree.
pub fn eligible(
    architecture: &str,
    graph: &DataflowGraph,
    weights: &super::super::ProgramWeights,
    lanes: usize,
) -> bool {
    eligible_with_role(
        architecture,
        graph,
        weights,
        weights.fusion.is_some(),
        lanes,
    )
}

/// Node-level form of the same decision. `eligible` only answers whether the graph needs MMA
/// scratch at all; callers that route one attention node must use this function so a supported
/// FP8 node cannot make a sibling F32 target node eligible.
pub fn node_eligible(
    architecture: &str,
    node: &infer_ir::TensorNode,
    weights: &super::super::ProgramWeights,
    draft: bool,
    lanes: usize,
) -> bool {
    let TensorOp::Attention {
        query_heads,
        kv_heads,
        head_dim,
        window,
    } = node.op
    else {
        return false;
    };
    if node.states.is_empty()
        || !catalog::supported(
            architecture,
            query_heads,
            kv_heads,
            head_dim,
            lanes,
            window.unwrap_or(0),
        )
    {
        return false;
    }
    if draft {
        // MTP proposals already use the BF16 profile for both FP8 and F32 KV.
        return true;
    }
    let fp8 = node
        .states
        .first()
        .is_some_and(|id| weights.kv_scales.contains_key(id));
    match weights.attention_profile {
        AttentionProfile::Canonical => false,
        AttentionProfile::Bf16 => fp8,
        AttentionProfile::Fp16 => {
            fp8 && super::f16::supported(
                architecture,
                query_heads,
                kv_heads,
                head_dim,
                lanes,
                window.unwrap_or(0),
            )
        }
    }
}

/// Explicit-role form of [`eligible`], kept crate-visible for the eligibility matrix tests.
pub fn eligible_with_role(
    architecture: &str,
    graph: &DataflowGraph,
    weights: &super::super::ProgramWeights,
    draft: bool,
    lanes: usize,
) -> bool {
    graph
        .nodes
        .iter()
        .any(|node| node_eligible(architecture, node, weights, draft, lanes))
}
impl Workspace {
    pub fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &super::super::ProgramWeights,
        capacity: usize,
        lanes: usize,
        arithmetic: Arithmetic,
    ) -> Result<Option<Self>> {
        if !eligible(&device.profile()?.architecture, graph, weights, lanes) {
            return Ok(None);
        }
        let mut support = Support::load(device, capacity, lanes, arithmetic)?;
        if arithmetic == Arithmetic::Bf16
            && weights.fusion.is_some()
            && graph.nodes.iter().any(|node| {
                matches!(node.op, TensorOp::Attention { .. })
                    && node
                        .states
                        .first()
                        .is_some_and(|id| !weights.kv_scales.contains_key(id))
            })
        {
            support.prepare_float_cache(device)?;
        }
        Ok(Some(Self { support }))
    }
}
