use crate::{
    HostTensor, ImportedQwen, QwenProvider, SafetensorsFile, SafetensorsIndex, TensorDtype,
};
use infer_core::{Error, ErrorCode, ModelId, Result};
use infer_ir::{DataflowGraph, TensorStorage};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, collections::BTreeSet, io::Read, path::Path, path::PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightBinding {
    pub slot: String,
    pub source: String,
    pub shard: String,
    pub shape: Vec<usize>,
    pub dtype: TensorDtype,
    pub bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageManifest {
    pub schema_version: u32,
    pub model: infer_ir::ModelIr,
    pub bindings: Vec<WeightBinding>,
    pub ignored_tensors: Vec<String>,
    pub weights_bytes: u64,
    pub host_f32_bytes: u64,
    pub metadata_fingerprint: String,
}
pub struct QwenPackage {
    pub root: PathBuf,
    pub imported: ImportedQwen,
    pub graph: DataflowGraph,
    pub manifest: PackageManifest,
    pub(crate) shards: BTreeMap<String, SafetensorsFile>,
}
fn read_bounded(path: &Path, max: u64) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path)
        .map_err(|e| Error::invalid(format!("{}: {e}", path.display())))?;
    if file
        .metadata()
        .map_err(|e| Error::invalid(e.to_string()))?
        .len()
        > max
    {
        return Err(Error::new(
            ErrorCode::Capacity,
            "package metadata exceeds budget",
        ));
    }
    let mut bytes = Vec::new();
    file.take(max + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| Error::invalid(e.to_string()))?;
    if bytes.len() as u64 > max {
        return Err(Error::new(
            ErrorCode::Capacity,
            "package metadata grew beyond budget",
        ));
    }
    Ok(bytes)
}
///
/// # Errors
/// Returns an invalid-input or I/O error for a path that is invalid, inaccessible, or escapes the package root.
pub fn package_path(root: &Path, name: &str) -> Result<PathBuf> {
    let root = root
        .canonicalize()
        .map_err(|e| Error::invalid(e.to_string()))?;
    if name.is_empty()
        || name.contains('\\')
        || Path::new(name)
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err(Error::invalid("escaping package path"));
    }
    let path = root
        .join(name)
        .canonicalize()
        .map_err(|e| Error::invalid(format!("{name}: {e}")))?;
    if !path.starts_with(&root) {
        return Err(Error::invalid("package symlink escapes root"));
    }
    Ok(path)
}
impl QwenPackage {
    /// Recheck public package metadata against the immutable opened shard headers.
    /// # Errors
    /// Returns an invalid-input error if the model, graph, bindings, or byte totals changed inconsistently.
    pub fn validate_weight_bindings(&self) -> Result<()> {
        if self.manifest.model != self.imported.model
            || self.graph != infer_compiler::dataflow::lower(&self.imported.model)?
        {
            return Err(Error::invalid("package model/dataflow metadata mismatch"));
        }
        let tensor_map = self
            .shards
            .iter()
            .flat_map(|(shard, file)| {
                file.tensors
                    .keys()
                    .map(move |name| (name.clone(), shard.clone()))
            })
            .collect();
        let expected = bind_weights(
            &self.imported.model,
            &self.graph,
            &[],
            &self.shards,
            &tensor_map,
        )?;
        if expected.bindings != self.manifest.bindings
            || expected.weights_bytes != self.manifest.weights_bytes
            || expected.host_f32_bytes != self.manifest.host_f32_bytes
        {
            return Err(Error::invalid("package weight binding metadata mismatch"));
        }
        Ok(())
    }
    ///
    /// # Errors
    /// Returns an I/O or invalid-input error for missing, malformed, unsupported, or oversized assets.
    pub fn open(root: impl AsRef<Path>, id: ModelId) -> Result<Self> {
        let root = root
            .as_ref()
            .canonicalize()
            .map_err(|e| Error::invalid(e.to_string()))?;
        let config = read_bounded(&package_path(&root, "config.json")?, 1024 * 1024)?;
        let imported = QwenProvider.import_manifest(id, &config)?;
        let graph = infer_compiler::dataflow::lower(&imported.model)?;
        let (shards, tensor_map) = load_shards(&root)?;
        let manifest = bind_weights(&imported.model, &graph, &config, &shards, &tensor_map)?;
        Ok(Self {
            root,
            imported,
            graph,
            manifest,
            shards,
        })
    }
    /// Budget applies to retained F32 weights plus the conversion scratch.
    ///
    /// # Errors
    /// Returns an I/O, invalid-input, unsupported, or capacity error if shards, tensor metadata, dtypes, or the memory budget prevent loading.
    pub fn load_host_weights(&mut self, budget: u64) -> Result<BTreeMap<String, HostTensor>> {
        let scratch = self
            .manifest
            .bindings
            .iter()
            .map(|b| b.bytes)
            .max()
            .unwrap_or(0);
        if self
            .manifest
            .host_f32_bytes
            .checked_add(scratch)
            .is_none_or(|n| n > budget)
        {
            return Err(Error::new(
                ErrorCode::Capacity,
                format!(
                    "host weights need {} bytes plus {scratch} conversion scratch, budget {budget}",
                    self.manifest.host_f32_bytes
                ),
            ));
        }
        let mut output = BTreeMap::new();
        for binding in &self.manifest.bindings {
            let tensor = self
                .shards
                .get_mut(&binding.shard)
                .ok_or_else(|| Error::invariant("validated shard"))?
                .read_f32(&binding.source, budget)?;
            output.insert(binding.slot.clone(), tensor);
        }
        Ok(output)
    }
}

type Shards = BTreeMap<String, SafetensorsFile>;
type TensorShards = BTreeMap<String, String>;
fn load_shards(root: &Path) -> Result<(Shards, TensorShards)> {
    let index_path = root.join("model.safetensors.index.json");
    let index = if index_path.exists() {
        Some(SafetensorsIndex::parse(&read_bounded(
            &package_path(root, "model.safetensors.index.json")?,
            16 * 1024 * 1024,
        )?)?)
    } else {
        None
    };
    let names = index.as_ref().map_or_else(
        || vec!["model.safetensors".into()],
        SafetensorsIndex::shards,
    );
    let mut shards = BTreeMap::new();
    let mut tensor_map = BTreeMap::new();
    for name in names {
        let file = SafetensorsFile::open(package_path(root, &name)?)?;
        for tensor in file.tensors.keys() {
            if tensor_map.insert(tensor.clone(), name.clone()).is_some() {
                return Err(Error::invalid(format!(
                    "tensor {tensor} duplicated across shards"
                )));
            }
        }
        shards.insert(name, file);
    }
    if let Some(index) = &index {
        if index.weight_map != tensor_map {
            return Err(Error::invalid(
                "shard contents disagree with HF weight index",
            ));
        }
        let bytes = shards
            .values()
            .flat_map(|s| s.tensors.values())
            .try_fold(0u64, |sum, h| sum.checked_add(h.byte_len()))
            .ok_or_else(|| Error::invalid("package size overflow"))?;
        if bytes != index.weight_bytes()? {
            return Err(Error::invalid(
                "HF index total_size disagrees with shard payload",
            ));
        }
    }
    Ok((shards, tensor_map))
}
fn bind_weights(
    model: &infer_ir::ModelIr,
    graph: &DataflowGraph,
    config: &[u8],
    shards: &Shards,
    tensor_map: &TensorShards,
) -> Result<PackageManifest> {
    let prefixes = ["model.language_model.", "model.", ""];
    let prefix = prefixes
        .into_iter()
        .find(|p| tensor_map.contains_key(&format!("{p}embed_tokens.weight")))
        .ok_or_else(|| Error::invalid("missing text embedding tensor"))?;
    let mut bindings = Vec::new();
    let mut used = BTreeSet::new();
    let mut host_bytes = 0u64;
    let mut weights_bytes = 0u64;
    for spec in &graph.tensors {
        let TensorStorage::Weight { slot } = &spec.storage else {
            continue;
        };
        let source = if slot == "lm_head.weight" {
            slot.clone()
        } else {
            format!("{prefix}{slot}")
        };
        let shard = tensor_map
            .get(&source)
            .ok_or_else(|| Error::invalid(format!("unbound required weight {source}")))?;
        let header = &shards[shard].tensors[&source];
        if header.shape != spec.shape {
            return Err(Error::invalid(format!(
                "{source}: expected {:?}, found {:?}",
                spec.shape, header.shape
            )));
        }
        if !header.dtype.is_host_float() {
            return Err(Error::unsupported(format!(
                "{source}: precision package provider required"
            )));
        }
        weights_bytes = weights_bytes
            .checked_add(header.byte_len())
            .ok_or_else(|| Error::invalid("package size overflow"))?;
        host_bytes = host_bytes
            .checked_add(spec.elements()? as u64 * 4)
            .ok_or_else(|| Error::invalid("host size overflow"))?;
        used.insert(source.clone());
        bindings.push(WeightBinding {
            slot: slot.clone(),
            source,
            shard: shard.clone(),
            shape: spec.shape.clone(),
            dtype: header.dtype,
            bytes: header.byte_len(),
        });
    }
    let ignored_tensors = tensor_map
        .keys()
        .filter(|k| !used.contains(*k))
        .cloned()
        .collect();
    let mut digest = Sha256::new();
    digest.update(config);
    digest.update(serde_json::to_vec(&bindings).map_err(|e| Error::invalid(e.to_string()))?);
    let manifest = PackageManifest {
        schema_version: 1,
        model: model.clone(),
        bindings,
        ignored_tensors,
        weights_bytes,
        host_f32_bytes: host_bytes,
        metadata_fingerprint: format!("{:x}", digest.finalize()),
    };
    Ok(manifest)
}
