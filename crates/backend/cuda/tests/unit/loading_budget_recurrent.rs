use super::fixture_weights;
use crate::resident::ProgramGeometry;
use crate::resident::recurrent_prefill::required_bytes;
use infer_core::{OpId, Result, TensorId};
use infer_ir::{DataflowGraph, TensorNode, TensorOp};

fn graph() -> DataflowGraph {
    DataflowGraph {
        nodes: vec![TensorNode {
            id: OpId::ONE,
            layer: None,
            op: TensorOp::Delta {
                key_heads: 16,
                value_heads: 48,
                key_dim: 128,
                value_dim: 128,
            },
            inputs: vec![TensorId::ONE; 3],
            outputs: vec![TensorId::ONE],
            states: vec![],
        }],
        ..DataflowGraph::default()
    }
}

#[test]
fn quantized_recurrent_admission_counts_retained_prompt_graphs_and_shares_layers() -> Result<()> {
    let mut weights = fixture_weights();
    weights.input_scales.insert(TensorId::ONE, 1.0);
    weights.chunked_recurrent = true;
    weights.prefill_width = 256;
    weights.narrow_prefill_width = 64;
    let mut graph = graph();
    // Even scalar fallback nodes share output storage by value geometry.
    graph.nodes[0].op = TensorOp::Delta {
        key_heads: 0,
        value_heads: 48,
        key_dim: 128,
        value_dim: 128,
    };
    let mut later = graph.nodes[0].clone();
    later.id = OpId::new(2)?;
    later.op = TensorOp::Delta {
        key_heads: 16,
        value_heads: 48,
        key_dim: 128,
        value_dim: 128,
    };
    graph.nodes.push(later.clone());
    later.id = OpId::new(3)?;
    graph.nodes.push(later);
    let quote = |private| {
        ProgramGeometry::new(&weights, private)
            .prompts
            .into_iter()
            .filter(|width| *width > 0)
            .try_fold(0, |total, width| {
                Ok::<_, infer_core::Error>(total + required_bytes(&graph, &weights, width)?)
            })
    };
    // 6 MiB output scratch at 256, one quarter at 64, shared across layers.
    let wide = 6_291_456;
    assert_eq!(quote(false)?, wide / 4);
    assert_eq!(quote(true)?, wide + wide / 4);
    weights.fp8_inputs.insert(TensorId::ONE);
    assert_eq!(required_bytes(&graph, &weights, 256)?, wide);
    Ok(())
}

#[test]
fn recurrent_admission_counts_all_prompt_scratch_widths() -> Result<()> {
    let mut weights = fixture_weights();
    weights.chunked_recurrent = true;
    assert_eq!(required_bytes(&graph(), &weights, 16)?, 48 * 128 * 16 * 4);
    assert_eq!(required_bytes(&graph(), &weights, 512)?, 48 * 128 * 512 * 4);
    Ok(())
}

#[test]
fn recurrent_admission_rejects_scratch_overflow_before_allocation() {
    let mut weights = fixture_weights();
    weights.chunked_recurrent = true;
    let mut graph = graph();
    graph.nodes[0].op = TensorOp::Delta {
        key_heads: 1,
        value_heads: usize::MAX,
        key_dim: 128,
        value_dim: 128,
    };
    assert!(required_bytes(&graph, &weights, 256).is_err());
}
