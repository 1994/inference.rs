//! The accepted-prefix reference against the goldens recorded before the executor was deleted.

use super::*;
use serde_json::Value;

fn golden(name: &str) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples")
        .join(name)
        .join("golden.json");
    serde_json::from_slice(&std::fs::read(path).expect("the reference golden is registered"))
        .expect("the reference golden parses")
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "The goldens record decimal numbers narrowed to the f32 the reference stores"
)]
fn numbers(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .expect("an array")
        .iter()
        .map(|item| item.as_f64().expect("a number") as f32)
        .collect()
}

fn count(value: &Value) -> usize {
    usize::try_from(value.as_u64().expect("a count")).expect("a usize")
}

fn close(left: &[f32], right: &[f32], tolerance: f32, what: &str) {
    assert_eq!(left.len(), right.len(), "{what}: length");
    for (index, (a, b)) in left.iter().zip(right).enumerate() {
        assert!(
            (a - b).abs() <= tolerance,
            "{what}[{index}]: reference {a} vs golden {b}"
        );
    }
}

fn delta() -> (DeltaGeometry, Value) {
    let golden = golden("recurrent-delta");
    let geometry = DeltaGeometry {
        key_heads: count(&golden["geometry"]["key_heads"]),
        value_heads: count(&golden["geometry"]["value_heads"]),
        key_dim: count(&golden["geometry"]["key_dim"]),
        value_dim: count(&golden["geometry"]["value_dim"]),
        norm_epsilon: golden["geometry"]["norm_epsilon"]
            .as_f64()
            .expect("epsilon"),
    };
    (geometry, golden)
}

fn recorded_steps(golden: &Value) -> Vec<Vec<Vec<f32>>> {
    golden["declared_inputs"]["rows"]
        .as_array()
        .expect("recorded steps")
        .iter()
        .map(|rows| rows.as_array().expect("rows").iter().map(numbers).collect())
        .collect()
}

#[test]
fn the_delta_reference_reproduces_the_recorded_steps() {
    let (geometry, golden) = delta();
    let steps = recorded_steps(&golden);
    let mut state = numbers(&golden["base_state"]);
    assert_eq!(state.len(), geometry.state_len());
    for (index, rows) in steps.iter().enumerate() {
        let borrowed: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        let out = geometry.step(&mut state, &borrowed).expect("the step fits");
        close(
            &state,
            &numbers(&golden["states"][index]),
            1e-6,
            "delta state",
        );
        close(
            &out,
            &numbers(&golden["outputs"][index]),
            1e-6,
            "delta output",
        );
    }
}

#[test]
fn folding_the_delta_prefix_is_the_recorded_commit() {
    let (geometry, golden) = delta();
    let steps = recorded_steps(&golden);
    let base = numbers(&golden["base_state"]);
    let folded = geometry
        .fold(&base, &steps, steps.len())
        .expect("the fold runs");
    // R1's property: replaying the accepted prefix lands on the recorded state, so a commit needs
    // no per-candidate snapshot.
    close(
        &folded,
        &numbers(&golden["states"][steps.len() - 1]),
        1e-6,
        "folded delta state",
    );
    assert_eq!(golden["replay_property"]["holds"], true);
    // A prefix shorter than the window lands somewhere else, which is what makes it a prefix.
    let partial = geometry.fold(&base, &steps, 1).expect("the fold runs");
    assert_ne!(partial, folded);
}

#[test]
fn folding_rejects_an_accepted_prefix_beyond_the_record() {
    let (geometry, golden) = delta();
    let steps = recorded_steps(&golden);
    let error = geometry
        .fold(&numbers(&golden["base_state"]), &steps, steps.len() + 1)
        .expect_err("an over-long prefix is refused");
    assert!(
        error.to_string().contains("beyond the recorded window"),
        "{error}"
    );
}

fn conv() -> (ConvGeometry, Value) {
    let golden = golden("recurrent-conv");
    let geometry = ConvGeometry {
        channels: count(&golden["geometry"]["channels"]),
        kernel: count(&golden["geometry"]["kernel"]),
    };
    (geometry, golden)
}

#[test]
fn the_conv_reference_reproduces_the_recorded_steps() {
    let (geometry, golden) = conv();
    let rows = golden["declared_inputs"]["rows"]
        .as_array()
        .expect("recorded steps");
    let mut history = numbers(&golden["base_history"]);
    assert_eq!(history.len(), geometry.history_len());
    for (index, row) in rows.iter().enumerate() {
        let activation = numbers(&row["activation"]);
        let weights = numbers(&row["weights"]);
        let out = geometry
            .step(&mut history, &activation, &weights)
            .expect("the step fits");
        close(
            &history,
            &numbers(&golden["histories"][index]),
            1e-6,
            "conv window",
        );
        close(
            &out,
            &numbers(&golden["outputs"][index]),
            1e-6,
            "conv output",
        );
    }
}

#[test]
fn gathering_the_conv_window_is_the_recorded_commit() {
    let (geometry, golden) = conv();
    let accepted: Vec<Vec<f32>> = golden["declared_inputs"]["rows"]
        .as_array()
        .expect("recorded steps")
        .iter()
        .map(|row| numbers(&row["activation"]))
        .collect();
    let folded = geometry
        .fold(&numbers(&golden["base_history"]), &accepted)
        .expect("the gather runs");
    let histories = golden["histories"].as_array().expect("histories");
    close(
        &folded,
        &numbers(&histories[histories.len() - 1]),
        1e-6,
        "gathered window",
    );
    assert_eq!(golden["replay_property"]["holds"], true);
}
