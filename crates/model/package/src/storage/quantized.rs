//! Validated native storage bindings for mixed FP8/NVFP4 packages.
use crate::{
    ImportedModel, ModelRegistry, SafetensorsFile, TensorDtype, default_registry, package_path,
};
use infer_core::{Error, ErrorCode, ModelId, Result};
use infer_ir::{DataflowGraph, TensorStorage};
use infer_spi::ModelProvider;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path, sync::Arc};

mod activations;

/// Values sharing one E4M3 block scale in an NVFP4 weight.
const NVFP4_GROUP_SIZE: usize = 16;
/// Bits per weight in channel-wise FP8 quantization.
const FP8_WEIGHT_BITS: usize = 8;
/// Bits per weight in NVFP4 quantization.
const NVFP4_WEIGHT_BITS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightEncoding {
    Float,
    Fp8Channel,
    Nvfp4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightSource {
    pub name: String,
    pub shard: String,
    pub dtype: TensorDtype,
    pub shape: Vec<usize>,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceWeight {
    pub shape: Vec<usize>,
    pub encoding: WeightEncoding,
    pub data: WeightSource,
    pub scale: Option<WeightSource>,
    pub global_scale: Option<WeightSource>,
    pub input_global_scale: Option<WeightSource>,
    #[serde(default)]
    pub fp8_token_input: bool,
}

pub struct QuantizedPackage {
    /// Provider that owns this package's architecture and weight-name mapping.
    pub provider: Arc<dyn ModelProvider>,
    pub imported: ImportedModel,
    pub graph: DataflowGraph,
    pub weights: BTreeMap<String, DeviceWeight>,
    pub mtp: BTreeMap<String, DeviceWeight>,
    /// KV storage declared by the checkpoint, separate from weight encoding.
    pub kv_cache_dtype: Option<TensorDtype>,
    shards: BTreeMap<String, SafetensorsFile>,
    tensor_map: BTreeMap<String, String>,
    activation_rules: Vec<activations::Rule>,
}

impl QuantizedPackage {
    /// Open and validate every text-backbone weight without reading its payload.
    /// # Errors
    /// Rejects unsupported compression, missing scales, mismatched shapes and invalid shards.
    pub fn open(root: impl AsRef<Path>, id: ModelId) -> Result<Self> {
        Self::open_with(default_registry(), root, id)
    }

    /// Open with an explicit provider registry, so a caller can add its own model families.
    /// # Errors
    /// Rejects unsupported compression, mismatched shapes, invalid shards and unregistered
    /// architectures.
    pub fn open_with(
        registry: &ModelRegistry,
        root: impl AsRef<Path>,
        id: ModelId,
    ) -> Result<Self> {
        let root = root.as_ref();
        let config = crate::storage::package::read_bounded(
            &package_path(root, "config.json")?,
            crate::constants::CONFIG_MAX_BYTES,
        )?;
        let value: serde_json::Value =
            serde_json::from_slice(&config).map_err(|e| Error::invalid(e.to_string()))?;
        if let Some(quant) = value.get("quantization_config") {
            if quant["quant_method"] != "compressed-tensors" {
                return Err(Error::unsupported(
                    "only compressed-tensors quantized packages are supported",
                ));
            }
            validate_groups(quant)?;
        }
        let provider = registry.resolve(&config)?;
        let imported = provider.import(id, &config)?;
        let graph = provider.graph(&imported.model)?;
        let (shards, tensor_map) = crate::storage::package::load_shards(root)?;
        let mut package = Self {
            provider,
            imported,
            graph,
            weights: BTreeMap::new(),
            mtp: BTreeMap::new(),
            kv_cache_dtype: declared_kv_dtype(&value["quantization_config"]["kv_cache_scheme"])?,
            shards,
            tensor_map,
            activation_rules: activations::Rule::parse(&value["quantization_config"])?,
        };
        package.bind_backbone()?;
        package.bind_mtp()?;
        Ok(package)
    }

    /// Read one stored tensor under an explicit staging budget.
    /// # Errors
    /// Returns errors for a foreign source descriptor, I/O failure, or exceeded budget.
    pub fn read(&mut self, source: &WeightSource, budget: u64) -> Result<Vec<u8>> {
        Ok(self.read_view(source, budget)?.to_vec())
    }

    /// Borrow one stored tensor directly from its mapped shard.
    /// # Errors
    /// Rejects foreign source descriptors, invalid bounds or an exceeded budget.
    pub fn read_view(&self, source: &WeightSource, budget: u64) -> Result<&[u8]> {
        let file = self
            .shards
            .get(&source.shard)
            .ok_or_else(|| Error::invalid("unknown weight shard"))?;
        let header = file
            .tensors
            .get(&source.name)
            .ok_or_else(|| Error::invalid("unknown weight tensor"))?;
        if header.shape != source.shape
            || header.dtype != source.dtype
            || header.byte_len() != source.bytes
        {
            return Err(Error::invalid("weight source descriptor changed"));
        }
        file.bytes(&source.name, budget)
    }

    /// Read one embedding row without staging the entire embedding table.
    /// # Errors
    /// Rejects foreign descriptors, invalid rows and excessive staging.
    pub fn read_float_row(
        &mut self,
        source: &WeightSource,
        row: usize,
        budget: u64,
    ) -> Result<Vec<f32>> {
        let file = self
            .shards
            .get_mut(&source.shard)
            .ok_or_else(|| Error::invalid("unknown embedding shard"))?;
        let header = file
            .tensors
            .get(&source.name)
            .ok_or_else(|| Error::invalid("unknown embedding tensor"))?;
        if header.shape != source.shape
            || header.dtype != source.dtype
            || header.byte_len() != source.bytes
        {
            return Err(Error::invalid("embedding descriptor changed"));
        }
        file.read_row_f32(&source.name, row, budget)
    }

    /// Inspect a tensor descriptor without reading its payload.
    /// # Errors
    /// Rejects names not present in the validated package index.
    pub fn source(&self, name: &str) -> Result<WeightSource> {
        let shard = self
            .tensor_map
            .get(name)
            .ok_or_else(|| Error::invalid(format!("missing weight {name}")))?;
        let header = &self.shards[shard].tensors[name];
        Ok(WeightSource {
            name: name.into(),
            shard: shard.clone(),
            dtype: header.dtype,
            shape: header.shape.clone(),
            bytes: header.byte_len(),
        })
    }

    fn bind_backbone(&mut self) -> Result<()> {
        let anchor = self.provider.anchor_slot();
        let prefix = self
            .provider
            .weight_prefixes()
            .iter()
            .copied()
            .find(|prefix| self.tensor_map.contains_key(&format!("{prefix}{anchor}")))
            .ok_or_else(|| Error::invalid(format!("missing {anchor} tensor")))?;
        let mut weights = BTreeMap::new();
        for spec in &self.graph.tensors {
            if let TensorStorage::Weight { slot } = &spec.storage {
                let name = self.provider.weight_source(slot, prefix);
                weights.insert(slot.clone(), self.bind(&name, &spec.shape)?);
            }
        }
        self.weights = weights;
        Ok(())
    }

    fn bind(&self, name: &str, shape: &[usize]) -> Result<DeviceWeight> {
        let base = name.strip_suffix(".weight").unwrap_or(name);
        let packed = format!("{base}.weight_packed");
        let (data, encoding) = if self.tensor_map.contains_key(&packed) {
            (self.source(&packed)?, WeightEncoding::Nvfp4)
        } else {
            let data = self.source(name)?;
            let encoding = if data.dtype == TensorDtype::F8E4m3 {
                WeightEncoding::Fp8Channel
            } else {
                WeightEncoding::Float
            };
            (data, encoding)
        };
        let mut weight = DeviceWeight {
            shape: shape.into(),
            encoding,
            data,
            scale: None,
            global_scale: None,
            input_global_scale: None,
            fp8_token_input: encoding == WeightEncoding::Fp8Channel
                && activations::Rule::for_module(&self.activation_rules, base)?,
        };
        match encoding {
            WeightEncoding::Float => {
                if !weight.data.dtype.is_host_float() || weight.data.shape != shape {
                    return Err(Error::invalid(format!(
                        "{name}: floating weight dtype/shape mismatch"
                    )));
                }
            }
            WeightEncoding::Fp8Channel | WeightEncoding::Nvfp4 => {
                self.bind_scales(base, &mut weight)?;
            }
        }
        Ok(weight)
    }

    fn bind_scales(&self, base: &str, weight: &mut DeviceWeight) -> Result<()> {
        let [rows, columns] = weight.shape.as_slice() else {
            return Err(Error::invalid("quantized weights must be matrices"));
        };
        let scale = self.source(&format!("{base}.weight_scale"))?;
        if weight.encoding == WeightEncoding::Fp8Channel {
            if weight.data.shape != weight.shape
                || scale.shape != [*rows, 1]
                || !scale.dtype.is_host_float()
            {
                return Err(Error::invalid(format!(
                    "{base}: invalid per-channel FP8 binding"
                )));
            }
        } else {
            if !columns.is_multiple_of(NVFP4_GROUP_SIZE)
                || weight.data.dtype != TensorDtype::U8
                || weight.data.shape != [*rows, columns / 2]
                || scale.dtype != TensorDtype::F8E4m3
                || scale.shape != [*rows, columns / NVFP4_GROUP_SIZE]
            {
                return Err(Error::invalid(format!(
                    "{base}: invalid NVFP4 block-16 binding"
                )));
            }
            for (suffix, target) in [
                ("weight_global_scale", &mut weight.global_scale),
                ("input_global_scale", &mut weight.input_global_scale),
            ] {
                let source = self.source(&format!("{base}.{suffix}"))?;
                if source.dtype != TensorDtype::F32 || source.shape != [1] {
                    return Err(Error::invalid(format!("{base}: invalid global scale")));
                }
                *target = Some(source);
            }
        }
        weight.scale = Some(scale);
        Ok(())
    }

    fn bind_mtp(&mut self) -> Result<()> {
        let Some(prefix) = self
            .imported
            .speculation
            .as_ref()
            .map(|plan| plan.prefix.clone())
            .filter(|prefix| !prefix.is_empty())
        else {
            return Ok(());
        };
        let names: Vec<_> = self
            .tensor_map
            .keys()
            .filter(|name| name.starts_with(&prefix))
            .cloned()
            .collect();
        // MTP is optional at load time; enabling it requires the complete head in the executor.
        for name in names {
            if name.ends_with(".weight") {
                let shape = self.source(&name)?.shape;
                self.mtp.insert(name.clone(), self.bind(&name, &shape)?);
            }
        }
        Ok(())
    }
}

fn validate_groups(quant: &serde_json::Value) -> Result<()> {
    let groups = quant["config_groups"]
        .as_object()
        .ok_or_else(|| Error::invalid("missing quantization groups"))?;
    if groups.is_empty() {
        return Err(Error::invalid("empty quantization groups"));
    }
    for group in groups.values() {
        let weights = &group["weights"];
        let valid = weights["type"] == "float"
            && weights["symmetric"] == true
            && weights["dynamic"] == false
            && ((group["format"] == "float-quantized"
                && weights["num_bits"] == FP8_WEIGHT_BITS
                && weights["strategy"] == "channel")
                || (group["format"] == "nvfp4-pack-quantized"
                    && weights["num_bits"] == NVFP4_WEIGHT_BITS
                    && weights["strategy"] == "tensor_group"
                    && weights["group_size"] == NVFP4_GROUP_SIZE));
        if !valid {
            return Err(Error::new(
                ErrorCode::Unsupported,
                "unsupported mixed-precision quantization group",
            ));
        }
    }
    Ok(())
}

fn declared_kv_dtype(scheme: &serde_json::Value) -> Result<Option<TensorDtype>> {
    if scheme.is_null() {
        return Ok(None);
    }
    if scheme["num_bits"] == FP8_WEIGHT_BITS
        && scheme["type"] == "float"
        && scheme["strategy"] == "tensor"
        && scheme["dynamic"] == false
        && scheme["symmetric"] == true
    {
        Ok(Some(TensorDtype::F8E4m3))
    } else {
        Err(Error::unsupported(
            "unsupported checkpoint KV quantization scheme",
        ))
    }
}

#[cfg(test)]
mod kv_tests {
    use super::*;
    #[test]
    fn checkpoint_kv_storage_is_declared_independently_of_weights() -> Result<()> {
        assert_eq!(declared_kv_dtype(&serde_json::Value::Null)?, None);
        let mut scheme = serde_json::json!({"num_bits":8,"type":"float","strategy":"tensor","dynamic":false,"symmetric":true});
        assert_eq!(declared_kv_dtype(&scheme)?, Some(TensorDtype::F8E4m3));
        scheme["dynamic"] = true.into();
        assert!(declared_kv_dtype(&scheme).is_err());
        scheme["dynamic"] = false.into();
        scheme["type"] = "int".into();
        assert!(declared_kv_dtype(&scheme).is_err());
        Ok(())
    }
}
