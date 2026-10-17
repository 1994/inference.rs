use super::gdn_ba_nodes;
use crate::resident::prefill_projection::GDN_BA_SHAPE;
use infer_core::{OpId, TensorId};
use infer_ir::{TensorNode, TensorOp};

fn linear(layer: Option<usize>, input: u64, weight: u64, output: u64) -> TensorNode {
    TensorNode {
        id: OpId::new(output).unwrap(),
        layer,
        op: TensorOp::Linear,
        inputs: vec![
            TensorId::new(input).unwrap(),
            TensorId::new(weight).unwrap(),
        ],
        outputs: vec![TensorId::new(output).unwrap()],
        states: Vec::new(),
    }
}

#[test]
fn gdn_ba_pair_is_same_layer_same_input_and_two_plain_linears() {
    assert_eq!(GDN_BA_SHAPE, [48, 5120]);
    let first = linear(Some(3), 1, 10, 2);
    let second = linear(Some(3), 1, 11, 3);
    assert!(gdn_ba_nodes(&first, &second));

    let other_layer = linear(Some(4), 1, 11, 3);
    assert!(!gdn_ba_nodes(&first, &other_layer));
    let other_input = linear(Some(3), 9, 11, 3);
    assert!(!gdn_ba_nodes(&first, &other_input));
    let mut not_linear = second.clone();
    not_linear.op = TensorOp::Silu;
    assert!(!gdn_ba_nodes(&first, &not_linear));
    let mut second_output = second;
    second_output.outputs.push(TensorId::new(12).unwrap());
    assert!(!gdn_ba_nodes(&first, &second_output));
}
