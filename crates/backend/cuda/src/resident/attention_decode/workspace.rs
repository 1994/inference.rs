//! Capture-time split-KV planning and reusable scratch, independent of model identity.
use super::kernels;
use crate::device::{CudaDevice, device_error};
use cutile::prelude::*;
use infer_core::Result;
use infer_ir::{DataflowGraph, TensorOp};
use std::collections::BTreeMap;

struct Buffers {
    parts: usize,
    numerator: Tensor<f32>,
    maxima: Tensor<f32>,
    sums: Tensor<f32>,
}

pub struct Workspace {
    buffers: BTreeMap<(usize, usize), Buffers>,
}

/// Bound the extra reduction work by available parallelism and the KV workload.
/// Like `FlashInfer`'s work estimator, this uses device occupancy inputs rather than a
/// GPU/model name. This conservative policy is not a substitute for kernel autotuning.
fn partitions(heads: usize, capacity: usize, multiprocessors: usize) -> usize {
    if capacity < crate::constants::ATTENTION_SPLIT_THRESHOLD {
        return 1;
    }
    let limit = multiprocessors
        .div_ceil(heads)
        .min(capacity / crate::constants::ATTENTION_SPLIT_MIN_TOKENS)
        .clamp(1, crate::constants::ATTENTION_SPLIT_MAX_PARTS);
    1 << limit.ilog2()
}

impl Workspace {
    pub(crate) fn new(device: &CudaDevice, graph: &DataflowGraph, capacity: usize) -> Result<Self> {
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
            let parts = partitions(query_heads, capacity.min(window.unwrap_or(capacity)), sms);
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
        Ok(Self { buffers })
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
                window,
                scales[0],
                scales[1],
            )
            .generics(vec![
                E::DTYPE.as_str().into(),
                dim.to_string(),
                (heads / kv_heads).to_string(),
                parts.to_string(),
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
mod tests {
    use super::partitions;
    #[test]
    fn split_plan_obeys_parallelism_and_short_window_limits() {
        assert_eq!(partitions(16, 512, 170), 1);
        assert_eq!(partitions(16, 4096, 8), 1);
        assert_eq!(partitions(16, 4096, 64), 4);
        assert_eq!(partitions(16, 4096, 170), 8);
        assert_eq!(partitions(8, 16384, 170), 16);
    }
}
