use super::*;
use infer_core::{OpId, Result};
use infer_ir::{TensorSpec, TensorStorage};

fn graph() -> Result<DataflowGraph> {
    let mut graph = DataflowGraph::default();
    for id in 1..=4 {
        graph.tensors.push(TensorSpec {
            id: TensorId::new(id)?,
            shape: vec![5120],
            dtype: DType::F32,
            storage: TensorStorage::Activation,
        });
    }
    for (id, op, inputs, output) in [
        (1, TensorOp::Linear, vec![1, 10], 2),
        (2, TensorOp::Add, vec![2, 3], 4),
    ] {
        graph.nodes.push(TensorNode {
            id: OpId::new(id)?,
            layer: None,
            op,
            inputs: inputs
                .into_iter()
                .map(TensorId::new)
                .collect::<Result<_>>()?,
            outputs: vec![TensorId::new(output)?],
            states: vec![],
        });
    }
    Ok(graph)
}

#[test]
fn residual_plan_preserves_operand_order_and_rejects_exposed_values() -> Result<()> {
    for side in [0, 1] {
        let mut graph = graph()?;
        if side == 1 {
            graph.nodes[1].inputs.swap(0, 1);
        }
        let pairs = from_contracts(&graph, |_| true, |_| false);
        assert_eq!(
            pairs.get(&0),
            Some(&Pair {
                residual: TensorId::new(3)?,
                projection_first: side == 0
            })
        );
        assert!(from_contracts(&graph, |_| false, |_| false).is_empty());
        assert!(from_contracts(&graph, |_| true, |_| true).is_empty());
        graph.hidden = Some(TensorId::new(2)?);
        assert!(from_contracts(&graph, |_| true, |_| false).is_empty());
        graph.hidden = None;
        graph.logits = Some(TensorId::new(2)?);
        assert!(from_contracts(&graph, |_| true, |_| false).is_empty());
    }
    Ok(())
}

#[test]
fn residual_plan_rejects_changed_contracts() -> Result<()> {
    for case in 0..14 {
        let mut graph = graph()?;
        match case {
            0 => graph.nodes[0].op = TensorOp::Add,
            1 => graph.nodes[1].op = TensorOp::Multiply,
            2 => graph.nodes[0].states.push(TensorId::new(5)?),
            3 => graph.nodes[1].states.push(TensorId::new(5)?),
            4 => graph.nodes[1].inputs[1] = TensorId::new(2)?,
            5 => graph.nodes[1].outputs[0] = TensorId::new(1)?,
            6 => graph.nodes[1].outputs[0] = TensorId::new(3)?,
            7 => graph.tensors[1].shape = vec![0],
            8 => graph.tensors[2].shape = vec![2560, 2],
            9 => {
                graph.nodes[0].inputs.pop();
            }
            10 => graph.nodes[1].outputs.push(TensorId::new(5)?),
            11 => {
                let mut consumer = graph.nodes[1].clone();
                consumer.id = OpId::new(3)?;
                graph.nodes.push(consumer);
            }
            12 => graph.nodes[1].inputs[0] = TensorId::new(1)?,
            _ => graph.tensors[2].dtype = DType::Bf16,
        }
        assert!(
            from_contracts(&graph, |_| true, |_| false).is_empty(),
            "case={case}"
        );
    }
    Ok(())
}

#[test]
fn residual_output_spans_reject_partial_aliases_and_allow_read_aliases() {
    assert!(disjoint_outputs(&[(0, 4), (0, 4), (4, 8), (8, 12)]));
    for spans in [
        [(0, 4), (4, 8), (0, 4), (8, 12)],
        [(0, 4), (4, 8), (8, 12), (2, 6)],
        [(0, 4), (4, 8), (6, 10), (10, 14)],
        [(0, 4), (4, 8), (8, 12), (10, 14)],
    ] {
        assert!(!disjoint_outputs(&spans));
    }
}
