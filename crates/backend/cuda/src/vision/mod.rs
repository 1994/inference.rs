//! Vision-modality execution: provider-declared geometry, bound weights, captured programs.
#[cfg(test)]
#[path = "attention_bench_check.rs"]
pub(crate) mod attention_bench;
#[cfg(test)]
#[path = "attention_check.rs"]
mod attention_check;
pub(crate) mod kernels;
pub(crate) mod program;
mod session;
mod weights;
pub use program::{
    AttentionMode, add, attention, attention_with, block, gelu, layernorm, merger, mlp_width,
    patch_embed, patch_width, project, project_with, tower,
};
pub use session::{
    argmax, generate, media_positions, placeholder_tokens, prefill, require_placements,
};
pub use weights::{VisionWeights, matrix};
