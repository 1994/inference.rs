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
    oxide_available: bool,
    mma: Option<super::super::attention_mma::Workspace>,
    pub(in crate::resident) oxide: Option<super::super::attention_oxide::Support>,
    pub(in crate::resident) shared: Option<super::super::attention_oxide::Support>,
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
            oxide_available: super::super::attention_oxide::Support::eligible(device, graph)?,
            mma: None,
            oxide: None,
            shared: None,
            buffers,
            prefill: BTreeMap::new(),
        })
    }

    pub(crate) fn prepare_mma(
        &mut self,
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &super::super::ProgramWeights,
        capacity: usize,
        lanes: usize,
    ) -> Result<()> {
        #[cfg(test)]
        if super::super::attention_oxide::disabled() {
            return Ok(());
        }
        self.mma =
            super::super::attention_mma::Workspace::new(device, graph, weights, capacity, lanes)?;
        Ok(())
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
        if !enabled && !self.oxide_available && self.mma.is_none() {
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
                let baseline = if self.oxide_available {
                    let baseline = crate::tuning::attention::exact::select(
                        device,
                        shape,
                        weights.autotune,
                        &weights.resident_tuning,
                    );
                    crate::tuning::attention::shared::select(
                        device,
                        shape,
                        weights.autotune,
                        &weights.resident_tuning,
                        baseline,
                    )
                } else if crate::tuning::attention::enabled() {
                    crate::tuning::attention::select(
                        device,
                        shape,
                        weights.autotune,
                        &weights.resident_tuning,
                    )
                } else {
                    crate::tuning::attention::Choice::Simt
                };
                #[cfg(test)]
                if taps::forced_choice().is_some() {
                    return baseline;
                }
                if self.mma.is_some() {
                    crate::tuning::attention::mma::select(
                        device,
                        shape,
                        false,
                        &weights.resident_tuning,
                        baseline,
                    )
                } else {
                    baseline
                }
            });
            #[cfg(test)]
            let choice = taps::forced_choice().unwrap_or(choice);
            self.prefill.insert((node.outputs[0], lanes), choice);
        }
        self.prepare_mma_modules(device);
        self.prepare_modules(device);
    }

    fn prepare_mma_modules(&mut self, device: &CudaDevice) {
        if let Some(mma) = self.mma.as_mut() {
            let variants = self
                .prefill
                .values()
                .filter_map(|choice| {
                    if let crate::tuning::attention::Choice::Mma { variant } = choice {
                        Some(*variant)
                    } else {
                        None
                    }
                })
                .collect::<std::collections::BTreeSet<_>>();
            for variant in variants {
                if let Err(error) = mma.support.prepare_variant(device, variant) {
                    tracing::warn!(target:"infer::tuning",%error,?variant,"MMA launch module unavailable: canonical fallback");
                    for choice in self.prefill.values_mut() {
                        if *choice == (crate::tuning::attention::Choice::Mma { variant }) {
                            *choice = if self.oxide_available {
                                crate::tuning::attention::Choice::RustSimt
                            } else {
                                crate::tuning::attention::Choice::Simt
                            };
                        }
                    }
                }
            }
        }
    }

    fn prepare_modules(&mut self, device: &CudaDevice) {
        if self.oxide.is_none() && self.prefill.values().any(|choice| choice.uses_oxide()) {
            match super::super::attention_oxide::Support::load(device) {
                Ok(Some(module)) => self.oxide = Some(module),
                result => {
                    tracing::warn!(target:"infer::tuning",error=?result.err(),"Rust attention module unavailable: cuTile fallback");
                    self.oxide_available = false;
                    for choice in self.prefill.values_mut() {
                        if choice.uses_oxide() {
                            *choice = crate::tuning::attention::Choice::Simt;
                        }
                    }
                }
            }
        }
        if self.shared.is_none()
            && self
                .prefill
                .values()
                .any(|choice| *choice == crate::tuning::attention::Choice::SharedKv)
        {
            match super::super::attention_oxide::Support::load_shared(device) {
                Ok(Some(module)) => self.shared = Some(module),
                result => {
                    tracing::warn!(target:"infer::tuning",error=?result.err(),"shared KV module unavailable: exact fallback");
                    for choice in self.prefill.values_mut() {
                        if *choice == crate::tuning::attention::Choice::SharedKv {
                            *choice = crate::tuning::attention::Choice::RustSimt;
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn prefill_mma(
        &mut self,
        output: TensorId,
        lanes: usize,
    ) -> Option<(
        &mut super::super::attention_mma::Support,
        super::super::attention_mma::catalog::Variant,
    )> {
        if let crate::tuning::attention::Choice::Mma { variant } =
            *self.prefill.get(&(output, lanes))?
        {
            Some((&mut self.mma.as_mut()?.support, variant))
        } else {
            None
        }
    }

    pub(crate) fn prefill_oxide(
        &self,
        output: TensorId,
        lanes: usize,
    ) -> Option<&super::super::attention_oxide::Support> {
        match self.prefill.get(&(output, lanes)) {
            Some(crate::tuning::attention::Choice::RustSimt) => self.oxide.as_ref(),
            Some(crate::tuning::attention::Choice::SharedKv) => self.shared.as_ref(),
            _ => None,
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
