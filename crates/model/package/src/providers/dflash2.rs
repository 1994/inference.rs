//! `DFlash2` draft geometry and the compatibility contract a target has to satisfy.
//!
//! D1's first step: the geometry the official draft declares, the checks that decide whether it can
//! be bound to a target at all, and the part of the resource quote that geometry alone determines.
//! Nothing here touches a device - the draft graph, its CUDA module and the proposal SPI come
//! later, and the SPI that carries proposals belongs to I1.
use infer_core::{Error, Result};
use infer_ir::{BackboneKind, Mixer, ModelIr, Operation};
use std::collections::BTreeSet;

/// The architecture string the official draft declares in its configuration.
pub const ARCHITECTURE: &str = "DFlash2DraftModel";

/// Operations the draft's fused modules require from a backend.
///
/// The mapping from the plan's module list - feature fusion, double-tap dynamic convolution,
/// block-internal non-causal attention, candidate selector - onto IR operations is provisional
/// until the draft graph lands. What D1 accepts is the check below: whatever the loader lowers to
/// must be a subset of what the backend reports.
pub const REQUIRED_OPERATIONS: [Operation; 5] = [
    Operation::Convolution,
    Operation::Attention,
    Operation::MatMul,
    Operation::RmsNorm,
    Operation::Residual,
];

/// Checks that a configuration declares this draft architecture.
///
/// An explicit `DFlash2` request must never be served silently by another algorithm, so the loader
/// calls this before it maps any field.
///
/// # Errors
/// Rejects a configuration whose declared architectures do not include [`ARCHITECTURE`].
pub fn check_declared_architecture(declared: &[String]) -> Result<()> {
    if declared.iter().any(|name| name == ARCHITECTURE) {
        return Ok(());
    }
    Err(Error::invalid(format!(
        "configuration does not declare {ARCHITECTURE}"
    )))
}

/// Geometry declared by the draft's `config.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftGeometry {
    pub hidden_size: usize,
    pub layers: usize,
    pub attention_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    /// Target layers whose hidden states the draft consumes, in ascending order.
    pub target_taps: Vec<usize>,
    pub sliding_window: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
}

/// What the draft costs before weights, arena, graph and rollback are known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftQuote {
    /// K and V for one token across every draft layer.
    pub kv_bytes_per_token: usize,
    /// The window the draft attention keeps per sequence.
    pub window_kv_bytes: usize,
    /// One retained target feature vector per tap, per token.
    pub feature_bytes_per_token: usize,
}

impl DraftGeometry {
    /// `z-lab/Qwen3.8-27B-DFlash2` as the plan records its configuration: five draft layers, hidden
    /// 5120, 32 attention heads, 8 KV heads, head dim 128, native block size 8, target taps
    /// `[5, 19, 33, 47, 61]`, sliding window 2048, selector rank 256 and top-k 16.
    #[must_use]
    pub fn official() -> Self {
        Self {
            hidden_size: 5120,
            layers: 5,
            attention_heads: 32,
            key_value_heads: 8,
            head_dim: 128,
            block_size: 8,
            target_taps: vec![5, 19, 33, 47, 61],
            sliding_window: 2048,
            selector_rank: 256,
            selector_top_k: 16,
        }
    }

    /// # Errors
    /// Rejects a geometry that cannot describe a draft: an empty dimension, KV heads that do not
    /// divide the attention heads, an empty or unordered tap list, a selector that takes more
    /// candidates than it scores, or a block wider than the window it slides in.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.layers == 0
            || self.attention_heads == 0
            || self.key_value_heads == 0
            || self.head_dim == 0
            || self.block_size == 0
            || self.sliding_window == 0
            || self.selector_rank == 0
        {
            return Err(Error::invalid("draft geometry has an empty dimension"));
        }
        if !self.attention_heads.is_multiple_of(self.key_value_heads) {
            return Err(Error::invalid(
                "draft key/value heads do not divide the attention heads",
            ));
        }
        if self.selector_top_k == 0 || self.selector_top_k > self.selector_rank {
            return Err(Error::invalid(
                "draft selector takes more candidates than it scores",
            ));
        }
        if self.block_size > self.sliding_window {
            return Err(Error::invalid(
                "draft block is wider than the sliding window",
            ));
        }
        if self.target_taps.is_empty() || self.target_taps.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(Error::invalid(
                "draft target taps must be a non-empty ascending list",
            ));
        }
        Ok(())
    }

    /// Checks the target-side acceptance conditions: a causal decoder whose hidden geometry the
    /// taps can feed, with every tapped layer present.
    ///
    /// Target identity, tokenizer correspondence and embedding sharing need the package's declared
    /// metadata and a token map, so they stay with the loader that resolves them.
    ///
    /// # Errors
    /// Reports the first condition the target fails, naming the tap or the geometry that differs.
    pub fn check_target(&self, target: &ModelIr) -> Result<()> {
        if target.backbone != BackboneKind::Decoder {
            return Err(Error::invalid("draft requires a causal decoder target"));
        }
        if target.hidden_size != self.hidden_size {
            return Err(Error::invalid(format!(
                "draft hidden {} does not match target hidden {}",
                self.hidden_size, target.hidden_size
            )));
        }
        let layers = target.mixers.len();
        for tap in &self.target_taps {
            let Some(mixer) = target.mixers.get(*tap) else {
                return Err(Error::invalid(format!(
                    "draft tap {tap} has no target layer (target has {layers})"
                )));
            };
            if !matches!(mixer, Mixer::Attention { .. }) {
                return Err(Error::invalid(format!(
                    "draft tap {tap} does not name a target attention layer"
                )));
            }
        }
        Ok(())
    }

    /// # Errors
    /// Names the first required operation the backend does not report.
    pub fn check_operations(&self, supported: &BTreeSet<Operation>) -> Result<()> {
        for operation in REQUIRED_OPERATIONS {
            if !supported.contains(&operation) {
                return Err(Error::invalid(format!(
                    "draft requires an unsupported operation: {operation:?}"
                )));
            }
        }
        Ok(())
    }

    /// The part of the draft's footprint that its geometry fixes.
    ///
    /// Weights, arena, graph and rollback cost need tensor shapes and device facts, so they are not
    /// quoted here; the plan's acceptance is that the complete quote is explainable, and this is
    /// the half that is explainable from the configuration alone.
    #[must_use]
    pub const fn quote(&self, dtype_bytes: usize) -> DraftQuote {
        let kv_bytes_per_token =
            self.layers * self.key_value_heads * self.head_dim * dtype_bytes * 2;
        DraftQuote {
            kv_bytes_per_token,
            window_kv_bytes: kv_bytes_per_token * self.sliding_window,
            feature_bytes_per_token: self.target_taps.len() * self.hidden_size * dtype_bytes,
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/providers_dflash2.rs"]
mod tests;
