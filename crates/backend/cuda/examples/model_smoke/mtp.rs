//! Single-layer Qwen MTP with a distinct KV cache and shared target vocabulary head.
use super::{
    Model,
    weights::{Projection, floats},
};
use infer_core::{Error, Result, TensorId};
use infer_ir::{Mixer, TensorOp};
use infer_models::QuantizedPackage;
use infer_state::physical::PhysicalTensor;
use std::{collections::BTreeMap, path::Path};

pub struct Mtp {
    pub model: Model,
    fc: Projection,
    embedding_norm: Vec<f32>,
    hidden_norm: Vec<f32>,
}

impl Mtp {
    pub fn load(root: &Path, target: &Model, capacity: usize) -> Result<Self> {
        let mut package = QuantizedPackage::open(root, infer_core::ModelId::ONE)?;
        if package.imported.mtp_layers != 1 {
            return Err(Error::unsupported("diagnostic supports one MTP layer"));
        }
        let get = |name: &str| {
            package
                .mtp
                .get(name)
                .cloned()
                .ok_or_else(|| Error::invalid(format!("missing {name}")))
        };
        let fc = get("mtp.fc.weight")?;
        let embedding_norm = get("mtp.pre_fc_norm_embedding.weight")?;
        let hidden_norm = get("mtp.pre_fc_norm_hidden.weight")?;
        let hidden = package.imported.model.hidden_size;
        if fc.shape != [hidden, hidden * 2]
            || embedding_norm.shape != [hidden]
            || hidden_norm.shape != [hidden]
        {
            return Err(Error::invalid("MTP fusion weight shapes"));
        }
        let fc = Projection::load(&target.device, &mut package, &fc)?;
        let embedding_norm = floats(&mut package, &embedding_norm.data)?;
        let hidden_norm = floats(&mut package, &hidden_norm.data)?;
        let mut block = package.imported.model.clone();
        block.mixers = vec![
            block
                .mixers
                .iter()
                .find(|m| matches!(m, Mixer::Attention { .. }))
                .cloned()
                .ok_or_else(|| Error::invalid("MTP attention configuration"))?,
        ];
        block.state.clear();
        package.graph = infer_compiler::dataflow::lower(&block)?;
        let mut bindings = BTreeMap::new();
        for spec in &package.graph.tensors {
            if let infer_ir::TensorStorage::Weight { slot } = &spec.storage {
                let weight = if slot == "embed_tokens.weight" || slot == "lm_head.weight" {
                    package.weights.get(slot)
                } else {
                    package.mtp.get(&format!("mtp.{slot}"))
                }
                .ok_or_else(|| Error::invalid(format!("missing MTP binding {slot}")))?;
                if weight.shape != spec.shape {
                    return Err(Error::invalid(format!("MTP shape mismatch {slot}")));
                }
                bindings.insert(slot.clone(), weight.clone());
            }
        }
        package.weights = bindings;
        let mut model = Model::from_package(package, capacity, Some(target))?;
        model.kv_offset = 1;
        Ok(Self {
            model,
            fc,
            embedding_norm,
            hidden_norm,
        })
    }

    pub fn step(
        &mut self,
        token: u32,
        previous_hidden: &[f32],
        position: usize,
        logits: bool,
    ) -> Result<Vec<f32>> {
        let source = self
            .model
            .embeddings
            .values()
            .next()
            .ok_or_else(|| Error::invalid("MTP embedding"))?;
        let embedded = self
            .model
            .package
            .read_float_row(source, token as usize, 1024 * 1024)?;
        let config = &self.model.package.imported.model;
        let op = TensorOp::Norm {
            epsilon: config.norm_epsilon,
            offset: config.norm_weight_offset,
            head_dim: config.hidden_size,
        };
        let norm = |input: &[f32], weight: &[f32]| -> Result<Vec<f32>> {
            infer_backend_host::reference_operation(&op, &[input, weight], None, 0, 0)?
                .pop()
                .ok_or_else(|| Error::invariant("MTP norm output"))
        };
        let mut fused = norm(&embedded, &self.embedding_norm)?;
        fused.extend(norm(previous_hidden, &self.hidden_norm)?);
        let fused = self.fc.apply(&self.model.device, &fused)?;
        self.model.forward(token, position, logits, Some(fused))
    }

    pub fn checkpoint(&self) -> BTreeMap<TensorId, PhysicalTensor> {
        self.model.state.clone()
    }

    pub fn restore(&mut self, state: BTreeMap<TensorId, PhysicalTensor>) {
        self.model.state = state;
    }
}
