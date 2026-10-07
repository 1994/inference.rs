use super::ProgramWeights;
use infer_core::{Error, Result};
use infer_ir::{DataflowGraph, TensorOp, TensorStorage};
use std::collections::BTreeMap;

/// `GatedNorm` binds activation, gate and norm weight inputs.
const GATED_NORM_INPUTS: usize = 3;
/// Delta-rule mixer binds qkv, beta, alpha, log-decay and bias inputs.
const DELTA_INPUTS: usize = 5;
/// Attention binds query, key and value inputs.
const ATTENTION_INPUTS: usize = 3;

pub(super) fn validate(graph: &DataflowGraph, weights: &ProgramWeights) -> Result<()> {
    graph.validate()?;
    let specs: BTreeMap<_, _> = graph.tensors.iter().map(|spec| (spec.id, spec)).collect();
    for node in &graph.nodes {
        let (inputs, outputs, states) = match &node.op {
            TensorOp::Embedding | TensorOp::Silu | TensorOp::Sigmoid | TensorOp::Rope { .. } => {
                (1, 1, 0)
            }
            TensorOp::Linear | TensorOp::Norm { .. } | TensorOp::Add | TensorOp::Multiply => {
                (2, 1, 0)
            }
            TensorOp::Split { widths, .. } => (1, widths.len(), 0),
            TensorOp::GatedNorm { .. } => (GATED_NORM_INPUTS, 1, 0),
            TensorOp::Conv { .. } => (2, 1, 1),
            TensorOp::Delta { .. } => (DELTA_INPUTS, 1, 1),
            TensorOp::Attention { .. } => (ATTENTION_INPUTS, 1, 1),
        };
        if node.inputs.len() != inputs
            || node.outputs.len() != outputs
            || node.states.len() != states
        {
            return Err(Error::invalid("resident operation binding count"));
        }
        let output = specs[&node.outputs[0]].elements()?;
        let input = specs[&node.inputs[0]].elements()?;
        if node.op == TensorOp::Linear {
            weights
                .projections
                .get(&node.inputs[1])
                .ok_or_else(|| Error::invalid("missing projection"))?
                .validate(output, input)?;
        }
        if let TensorOp::Rope {
            heads,
            head_dim,
            rotary_dim,
            theta,
        } = node.op
            && (heads.checked_mul(head_dim) != Some(output)
                || input != output
                || rotary_dim == 0
                || rotary_dim > head_dim
                || !theta.is_finite()
                || theta <= 0.0)
        {
            return Err(Error::invalid("resident rotary geometry"));
        }
        if let TensorOp::Norm {
            head_dim,
            epsilon,
            offset,
        } = node.op
        {
            validate_norm(head_dim, input, output, epsilon, offset)?;
        }
        if let TensorOp::GatedNorm { head_dim, epsilon } = node.op {
            validate_norm(head_dim, input, output, epsilon, 0.0)?;
        }
        if let TensorOp::Attention {
            kv_heads, head_dim, ..
        } = node.op
        {
            let state = specs[&node.states[0]];
            if state.shape.len() != 2
                || kv_heads.checked_mul(head_dim) != state.shape.get(1).copied()
            {
                return Err(Error::invalid("resident KV geometry"));
            }
        }
    }
    for spec in &graph.tensors {
        if matches!(spec.storage, TensorStorage::Weight { .. })
            && !weights.constants.contains_key(&spec.id)
            && !weights.projections.contains_key(&spec.id)
            && !weights.embeddings.contains_key(&spec.id)
        {
            return Err(Error::invalid("missing resident weight binding"));
        }
    }
    Ok(())
}

fn validate_norm(
    dimension: usize,
    input: usize,
    output: usize,
    epsilon: f32,
    offset: f32,
) -> Result<()> {
    if dimension == 0
        || dimension > crate::constants::MAX_HIDDEN_SIZE
        || input != output
        || !input.is_multiple_of(dimension)
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || !offset.is_finite()
    {
        return Err(Error::invalid("resident normalization geometry"));
    }
    Ok(())
}
