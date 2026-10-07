use crate::device::{CudaDevice, device_error};
use cuda_core::f8e4m3fn;
use cutile::prelude::*;
use infer_core::{Error, Result, TensorId};
use infer_ir::{DataflowGraph, StateKind, TensorStorage};
use std::collections::BTreeMap;

pub(super) type Fp8Caches = BTreeMap<TensorId, (Tensor<f8e4m3fn>, Tensor<f8e4m3fn>)>;

pub(super) fn allocate(
    device: &CudaDevice,
    graph: &DataflowGraph,
    capacity: usize,
    scales: &BTreeMap<TensorId, [f32; 2]>,
) -> Result<Fp8Caches> {
    let mut plan = Vec::new();
    let mut bytes = 0usize;
    let budget = device
        .memory_info()?
        .0
        .saturating_sub(device.profile()?.device_headroom_bytes());
    for spec in &graph.tensors {
        let Some(scale) = scales.get(&spec.id) else {
            continue;
        };
        if !matches!(
            spec.storage,
            TensorStorage::State {
                kind: StateKind::AttentionKv,
                ..
            }
        ) || spec.shape.len() != 2
            || scale.iter().any(|s| !s.is_finite() || *s <= 0.0)
        {
            return Err(Error::invalid("FP8 KV binding/scales"));
        }
        let size = capacity
            .checked_mul(spec.shape[1])
            .ok_or_else(|| Error::invalid("FP8 KV size overflow"))?;
        bytes = size
            .checked_mul(2)
            .and_then(|n| bytes.checked_add(n))
            .ok_or_else(|| Error::invalid("FP8 KV budget overflow"))?;
        if bytes as u64 > budget {
            return Err(Error::new(
                infer_core::ErrorCode::Capacity,
                "FP8 KV exceeds available device memory minus 1 GiB headroom",
            ));
        }
        plan.push((spec.id, size));
    }
    if plan.len() != scales.len() {
        return Err(Error::invalid("unknown FP8 KV state"));
    }
    let mut result = BTreeMap::new();
    for (id, size) in plan {
        let keys = api::zeros::<f8e4m3fn>(&[size])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let values = api::zeros::<f8e4m3fn>(&[size])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        result.insert(id, (keys, values));
    }
    Ok(result)
}
