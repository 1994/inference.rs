//! Small, untrained causal transformer for correctness/control-path testing.
//! It is an executable reference fixture, not a Qwen implementation.
mod fixture;
mod forward;
mod validation;
use infer_ir::ModelIr;
use math::{add_in_place, matvec, rms_norm, rope};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWeights {
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub attention_out: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceModel {
    pub ir: ModelIr,
    pub embeddings: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub lm_head: Vec<f32>,
}
mod math;
mod registry;
#[cfg(test)]
mod tests;
pub use registry::*;
mod backend;
pub use backend::*;
