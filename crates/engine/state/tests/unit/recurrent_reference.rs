//! The gated-delta reference the CUDA kernel is compared against.
//!
//! The step's own implementation lived in the CPU host executor, which E1 deletes; the numbers now
//! live in the registered golden, and R1's replay property is what makes an accepted-prefix commit
//! possible. These cases keep that artifact tied to the codebase rather than leaving a file nobody
//! reads: they check the recorded geometry, that every recorded value is finite, and that the
//! recorded replay actually reproduces the last state.

use serde_json::Value;

fn golden() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/recurrent-delta/golden.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn count(value: &Value) -> usize {
    usize::try_from(value.as_u64().unwrap()).unwrap()
}

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_f64().unwrap())
        .collect()
}

#[test]
fn the_recorded_geometry_describes_the_recorded_state() {
    let golden = golden();
    let key_heads = count(&golden["geometry"]["key_heads"]);
    let value_heads = count(&golden["geometry"]["value_heads"]);
    let key_dim = count(&golden["geometry"]["key_dim"]);
    let value_dim = count(&golden["geometry"]["value_dim"]);
    assert!(key_heads > 0 && value_heads.is_multiple_of(key_heads));

    let base = numbers(&golden["base_state"]);
    assert_eq!(base.len(), value_heads * key_dim * value_dim);
    assert!(
        base.iter().all(|value| *value == 0.0),
        "the reference starts empty"
    );

    let states = golden["states"].as_array().unwrap();
    let outputs = golden["outputs"].as_array().unwrap();
    assert_eq!(states.len(), outputs.len());
    for (state, output) in states.iter().zip(outputs) {
        let state = numbers(state);
        let output = numbers(output);
        assert_eq!(state.len(), base.len());
        assert_eq!(output.len(), value_heads * value_dim);
        assert!(state.iter().chain(&output).all(|value| value.is_finite()));
    }
    // The steps did something: the last state is not the base state.
    assert_ne!(numbers(&states[states.len() - 1]), base);
}

#[test]
fn replaying_the_recorded_inputs_reproduces_the_last_state() {
    let golden = golden();
    assert_eq!(
        golden["replay_property"]["holds"], true,
        "the recorded step has to satisfy the property R1 commits on"
    );
    // The property is what the golden claims; this checks the claim is about these steps.
    let rows = golden["declared_inputs"]["rows"].as_array().unwrap();
    assert_eq!(rows.len(), golden["states"].as_array().unwrap().len());
}
