use infer_core::ModelId;

use super::*;
#[test]
fn causal_hidden_prefix_is_independent_of_future_tokens() {
    let model = ReferenceModel::fixture(ModelId::new(1).unwrap(), 7);
    let short = model.forward(&[1, 2]).unwrap();
    let long = model.forward(&[1, 2, 3, 4]).unwrap();
    assert_eq!(short.hidden, &long.hidden[..2]);
    assert_eq!(short.logits.len(), 32);
}
#[test]
fn zero_weights_produce_zero_hidden_and_logits() {
    let mut model = ReferenceModel::fixture(ModelId::new(1).unwrap(), 7);
    model.embeddings.fill(0.0);
    let output = model.forward(&[1, 2, 3]).unwrap();
    assert!(
        output
            .logits
            .iter()
            .chain(output.hidden.iter().flatten())
            .all(|v| *v == 0.0)
    );
}
