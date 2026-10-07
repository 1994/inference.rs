//! Setup responsibilities.
use super::{HostBackend, HostConfig};
use infer_core::{Error, ErrorCode, Result};
use infer_ir::{ModelIr, TensorStorage};
use infer_models::{HostTensor, ModelPackage};
use infer_state::cache::PrefixCache;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, collections::VecDeque, time::Instant};

impl HostBackend {
    ///
    /// # Errors
    /// Returns a model-loading, unsupported-device, or capacity error if weights or execution buffers cannot be prepared.
    pub fn from_package(package: &mut ModelPackage, config: HostConfig) -> Result<Self> {
        let weights = package.load_host_weights(config.memory_bytes)?;
        Self::with_graph(
            package.imported.model.clone(),
            package.graph.clone(),
            weights,
            config,
        )
    }
    ///
    /// # Errors
    /// Returns an invalid-input error for invalid configuration, or a capacity error if the requested resources cannot be reserved.
    pub fn new(
        model: ModelIr,
        weights: BTreeMap<String, HostTensor>,
        config: HostConfig,
    ) -> Result<Self> {
        let graph = infer_model_recipes::decoder::lower(&model)?;
        Self::with_graph(model, graph, weights, config)
    }
    fn with_graph(
        model: ModelIr,
        graph: infer_ir::DataflowGraph,
        mut weights: BTreeMap<String, HostTensor>,
        config: HostConfig,
    ) -> Result<Self> {
        graph.validate()?;
        if config.memory_bytes == 0
            || config.block_size == 0
            || config.trace_capacity == 0
            || config.prefix_cache_bytes >= config.memory_bytes
        {
            return Err(Error::invalid("invalid host configuration"));
        }

        let mut bound = BTreeMap::new();
        let mut digest = Sha256::new();
        digest.update(serde_json::to_vec(&model).map_err(|e| Error::invalid(e.to_string()))?);
        let mut bytes = 0u64;
        for spec in &graph.tensors {
            let TensorStorage::Weight { slot } = &spec.storage else {
                continue;
            };
            let tensor = weights
                .remove(slot)
                .ok_or_else(|| Error::invalid(format!("unbound host weight {slot}")))?;
            tensor.validate()?;
            if tensor.shape != spec.shape {
                return Err(Error::invalid(format!("host shape mismatch for {slot}")));
            }
            bytes = bytes
                .checked_add(tensor.data.len() as u64 * crate::constants::F32_BYTES_U64)
                .ok_or_else(|| Error::invalid("host weight overflow"))?;
            digest.update(slot.as_bytes());
            for value in &tensor.data {
                digest.update(value.to_le_bytes());
            }
            bound.insert(spec.id, tensor);
        }
        if !weights.is_empty() {
            return Err(Error::invalid("unconsumed canonical host weights"));
        }
        let scratch = (graph.scratch_elements as u64)
            .checked_mul(crate::constants::F32_BYTES_U64)
            .ok_or_else(|| Error::invalid("host scratch overflow"))?;
        if bytes
            .checked_add(scratch)
            .and_then(|n| n.checked_add(config.probe_bytes))
            .and_then(|n| n.checked_add(config.prefix_cache_bytes))
            .is_none_or(|n| n > config.memory_bytes)
        {
            return Err(Error::new(
                ErrorCode::Capacity,
                "host weight/scratch budget exhausted",
            ));
        }
        let specs = graph.tensors.iter().map(|s| (s.id, s.clone())).collect();
        let slots = graph.lifetimes.iter().map(|s| (s.tensor, s.slot)).collect();
        let mut last_nodes = BTreeMap::new();
        for node in &graph.nodes {
            if let Some(layer) = node.layer {
                last_nodes.insert(layer, node.id);
            }
        }
        let layer_outputs = last_nodes
            .into_iter()
            .map(|(layer, op)| (op, layer))
            .collect();
        let identity = format!(
            "host-paged-dataflow-f32-v2:{:x}:page{}",
            digest.finalize(),
            config.block_size
        );
        let prefixes = PrefixCache::new(
            identity.as_bytes(),
            config.block_size,
            config.prefix_cache_bytes,
            crate::constants::PREFIX_CACHE_MAX_ENTRIES,
        )?;
        Ok(Self {
            model,
            graph,
            weights: bound,
            slots,
            specs,
            identity,
            config,
            sequences: BTreeMap::new(),
            tokens_executed: 0,
            traces: VecDeque::new(),
            trace_dropped: 0,
            prefixes,
            prefix_hits: 0,
            trace_origin: Instant::now(),
            layer_outputs,
            probes: VecDeque::new(),
            probe_dropped: 0,
        })
    }
    ///
    /// # Errors
    /// Returns a capacity or backend error if fresh execution state cannot be allocated.
    pub fn fresh(&self) -> Result<Self> {
        let weights = self
            .specs
            .values()
            .filter_map(|spec| match &spec.storage {
                TensorStorage::Weight { slot } => {
                    Some((slot.clone(), self.weights[&spec.id].clone()))
                }
                _ => None,
            })
            .collect();
        Self::new(self.model.clone(), weights, self.config.clone())
    }
}
