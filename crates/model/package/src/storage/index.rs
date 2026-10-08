//! Indexed weight shards and declared byte sizes.
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsIndex {
    pub metadata: BTreeMap<String, serde_json::Value>,
    pub weight_map: BTreeMap<String, String>,
}
impl SafetensorsIndex {
    ///
    /// # Errors
    /// Returns an invalid-input error for malformed or inconsistent package manifest data.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let index: Self =
            serde_json::from_slice(bytes).map_err(|e| Error::invalid(e.to_string()))?;
        if index.weight_map.is_empty() {
            return Err(Error::invalid("empty safetensors tensor index"));
        }
        for (tensor, shard) in &index.weight_map {
            if tensor.is_empty()
                || shard.is_empty()
                || !shard.ends_with(".safetensors")
                || shard.contains('\\')
                || shard
                    .split('/')
                    .any(|p| p.is_empty() || p == ".." || p == ".")
                || std::path::Path::new(shard)
                    .components()
                    .any(|c| !matches!(c, std::path::Component::Normal(_)))
            {
                return Err(Error::invalid("invalid/escaping weight shard path"));
            }
        }
        index.declared_weight_bytes()?;
        Ok(index)
    }
    ///
    /// # Errors
    /// Returns an invalid-input error if the declared weight size overflows.
    pub fn weight_bytes(&self) -> Result<u64> {
        self.declared_weight_bytes()?
            .ok_or_else(|| Error::invalid("weight index missing total_size"))
    }

    /// `metadata.total_size` when the exporter declared one.
    ///
    /// The index is not the authority on package size: the shard headers are, and the load path
    /// cross-checks against them when a declaration exists. Some exporters (`ModelScope`) omit it,
    /// so absence is not an error.
    /// # Errors
    /// Returns an invalid-input error if the declared size overflows or is not a whole number.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Legacy scientific-notation sizes are accepted only after checking finite, positive, integral values within the u64 range"
    )]
    #[expect(
        clippy::cast_precision_loss,
        reason = "Floating-point statistics, normalization, and deterministic sampling intentionally convert bounded counts to floating-point values"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "Legacy scientific-notation sizes are accepted only after checking finite, positive, integral values within the u64 range"
    )]
    pub fn declared_weight_bytes(&self) -> Result<Option<u64>> {
        let Some(number) = self.metadata.get("total_size") else {
            return Ok(None);
        };
        let size = number
            .as_u64()
            .or_else(|| {
                number
                    .as_f64()
                    .filter(|n| {
                        n.is_finite() && *n > 0.0 && n.fract() == 0.0 && *n < (u64::MAX as f64)
                    })
                    .map(|n| n as u64)
            })
            .ok_or_else(|| Error::invalid("invalid total_size"))?;
        if size == 0 {
            return Err(Error::invalid("empty weights"));
        }
        Ok(Some(size))
    }
    #[must_use]
    pub fn shards(&self) -> Vec<String> {
        self.weight_map
            .values()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}
