//! Prefix snapshots preserve head-major KV strides and whole recurrent state.
use crate::{
    device::{CudaDevice, device_error},
    resident::DeviceProgram,
};
use cuda_core::f8e4m3fn;
use cutile::prelude::*;
use infer_core::{Error, Result, TensorId};
use infer_ir::{DataflowGraph, StateKind, TensorStorage};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
pub struct ProgramSnapshot {
    states: BTreeMap<TensorId, Vec<Tensor<f32>>>,
    kv: BTreeSet<TensorId>,
    fp8: BTreeMap<TensorId, (Tensor<f8e4m3fn>, Tensor<f8e4m3fn>)>,
    position: usize,
    pub bytes: u64,
}

impl ProgramSnapshot {
    pub(super) fn capture(
        device: &CudaDevice,
        program: &DeviceProgram,
        graph: &DataflowGraph,
        covered: usize,
    ) -> Result<Self> {
        let kv: BTreeSet<_> = graph
            .tensors
            .iter()
            .filter_map(|t| {
                matches!(
                    t.storage,
                    TensorStorage::State {
                        kind: StateKind::AttentionKv,
                        ..
                    }
                )
                .then_some(t.id)
            })
            .collect();
        let mut states = BTreeMap::new();
        let mut bytes = 0_u64;
        for (id, tensors) in program.states() {
            let mut copies = Vec::with_capacity(tensors.len());
            for tensor in tensors {
                let mut shape = dimensions(tensor)?;
                if kv.contains(id) {
                    if shape.len() != crate::constants::KV_CACHE_RANK || shape[1] < covered {
                        return Err(Error::invariant("prefix KV shape"));
                    }
                    shape[1] = covered;
                }
                let copy = api::zeros::<f32>(&shape)
                    .sync_on(&device.stream)
                    .map_err(device_error)?;
                bytes = bytes
                    .checked_add((copy.size() * size_of::<f32>()) as u64)
                    .ok_or_else(|| Error::invalid("prefix byte overflow"))?;
                copies.push(copy);
            }
            states.insert(*id, copies);
        }
        let mut fp8 = BTreeMap::new();
        for (id, (keys, values)) in program.fp8_states() {
            let allocate = |source: &Tensor<f8e4m3fn>| -> Result<Tensor<f8e4m3fn>> {
                let mut shape = dimensions(source)?;
                if shape.len() != crate::constants::KV_CACHE_RANK || shape[1] < covered {
                    return Err(Error::invariant("prefix FP8 KV shape"));
                }
                shape[1] = covered;
                api::zeros::<f8e4m3fn>(&shape)
                    .sync_on(&device.stream)
                    .map_err(device_error)
            };
            let keys = allocate(keys)?;
            let values = allocate(values)?;
            bytes = bytes
                .checked_add((keys.size() + values.size()) as u64)
                .ok_or_else(|| Error::invalid("prefix byte overflow"))?;
            fp8.insert(*id, (keys, values));
        }
        // Allocate and validate every destination before queuing copies. Keep the whole
        // destination set alive through the drain even when a later copy fails.
        let copied = states
            .iter_mut()
            .try_for_each(|(id, copies)| {
                copies
                    .iter_mut()
                    .zip(&program.states()[id])
                    .try_for_each(|(dst, src)| {
                        copy_state(device, dst, src, kv.contains(id), covered)
                    })
            })
            .and_then(|()| {
                fp8.iter_mut().try_for_each(|(id, (keys, values))| {
                    let (source_keys, source_values) = &program.fp8_states()[id];
                    copy_state(device, keys, source_keys, true, covered)?;
                    copy_state(device, values, source_values, true, covered)
                })
            });
        device.drain()?;
        copied?;
        Ok(Self {
            states,
            kv,
            fp8,
            position: program.position(),
            bytes,
        })
    }

    pub(super) fn restore(
        &self,
        device: &CudaDevice,
        program: &mut DeviceProgram,
        covered: usize,
    ) -> Result<()> {
        let live = program.states_mut();
        if live.len() != self.states.len() {
            return Err(Error::invariant("prefix state set mismatch"));
        }
        for (id, cached) in &self.states {
            let tensors = live
                .get_mut(id)
                .ok_or_else(|| Error::invariant("prefix state missing"))?;
            if tensors.len() != cached.len() {
                return Err(Error::invariant("prefix state arity"));
            }
            for (dst, src) in tensors.iter_mut().zip(cached) {
                copy_state(device, dst, src, self.kv.contains(id), covered)?;
            }
        }
        let live = program.fp8_states_mut();
        if live.len() != self.fp8.len() {
            return Err(Error::invariant("prefix FP8 state set mismatch"));
        }
        for (id, (keys, values)) in &self.fp8 {
            let (dst_keys, dst_values) = live
                .get_mut(id)
                .ok_or_else(|| Error::invariant("prefix FP8 state missing"))?;
            copy_state(device, dst_keys, keys, true, covered)?;
            copy_state(device, dst_values, values, true, covered)?;
        }
        device.drain()?;
        program.set_position(self.position);
        Ok(())
    }
}

fn dimensions<T: DType>(tensor: &Tensor<T>) -> Result<Vec<usize>> {
    tensor
        .shape()
        .iter()
        .map(|&n| usize::try_from(n).map_err(device_error))
        .collect()
}

fn copy_state<T: DType>(
    device: &CudaDevice,
    dst: &mut Tensor<T>,
    src: &Tensor<T>,
    kv: bool,
    covered: usize,
) -> Result<()> {
    let d = dimensions(dst)?;
    let s = dimensions(src)?;
    if !kv {
        if d != s {
            return Err(Error::invariant("prefix recurrent shape"));
        }
        return device.copy_d2d(dst, src, src.size());
    }
    if d.len() != crate::constants::KV_CACHE_RANK
        || s.len() != crate::constants::KV_CACHE_RANK
        || d[0] != s[0]
        || d[2] != s[2]
        || d[1] < covered
        || s[1] < covered
    {
        return Err(Error::invariant("prefix KV stride mismatch"));
    }
    for head in 0..d[0] {
        if let Err(error) = device.copy_d2d_at(
            dst,
            src,
            head * d[1] * d[2],
            head * s[1] * s[2],
            covered * d[2],
        ) {
            device.drain()?;
            return Err(error);
        }
    }
    Ok(())
}

pub struct DraftSnapshot {
    pub program: ProgramSnapshot,
    pub hidden: Vec<f32>,
}

#[cfg(test)]
#[path = "../../tests/unit/executor_prefix_state.rs"]
mod tests;
