use super::*;
use crate::{
    mlp::ProjectionWeight,
    resident::{DeviceProgram, ProgramWeights, slot_batch::SlotPool},
};
use cutile::half::bf16;
use infer_core::{OpId, TensorId};
use infer_ir::{DType, DataflowGraph, TensorNode, TensorSpec};
use std::collections::BTreeMap;

fn fixture(device: &CudaDevice) -> Result<(DataflowGraph, ProgramWeights)> {
    let mut graph = DataflowGraph::default();
    for (index, shape, storage) in [
        (
            1,
            vec![32, 32],
            TensorStorage::Weight {
                slot: "embedding".into(),
            },
        ),
        (2, vec![32], TensorStorage::Activation),
        (
            3,
            vec![32, 4],
            TensorStorage::Weight {
                slot: "conv".into(),
            },
        ),
        (4, vec![32], TensorStorage::Activation),
        (
            5,
            vec![32, 4],
            TensorStorage::State {
                layer: 0,
                kind: StateKind::Conv,
            },
        ),
        (
            6,
            vec![32, 32],
            TensorStorage::Weight {
                slot: "projection".into(),
            },
        ),
        (7, vec![32], TensorStorage::Activation),
    ] {
        graph.tensors.push(TensorSpec {
            id: TensorId::new(index)?,
            shape,
            dtype: DType::F32,
            storage,
        });
    }
    for (index, op, inputs, outputs, states) in [
        (1, TensorOp::Embedding, vec![1], vec![2], vec![]),
        (
            2,
            TensorOp::Conv {
                channels: 32,
                kernel: 4,
            },
            vec![2, 3],
            vec![4],
            vec![5],
        ),
        (3, TensorOp::Linear, vec![4, 6], vec![7], vec![]),
    ] {
        let ids = |values: Vec<u64>| {
            values
                .into_iter()
                .map(TensorId::new)
                .collect::<Result<Vec<_>>>()
        };
        graph.nodes.push(TensorNode {
            id: OpId::new(index)?,
            layer: None,
            op,
            inputs: ids(inputs)?,
            outputs: ids(outputs)?,
            states: ids(states)?,
        });
    }
    graph.hidden = Some(TensorId::new(4)?);
    graph.logits = Some(TensorId::new(7)?);
    let dense = (0..1024_u16)
        .map(|i| bf16::from_f32(f32::from(i % 29) / 29.0 - 0.5))
        .collect();
    let embedding = device.upload(dense, &[32, 32])?;
    let weights = ProgramWeights {
        batch_width: 3,
        prefill_width: 3,
        fusion: None,
        kv_scales: BTreeMap::new(),
        tiling: BTreeMap::new(),
        embeddings: BTreeMap::from([(TensorId::ONE, Arc::clone(&embedding))]),
        projections: BTreeMap::from([(TensorId::new(6)?, ProjectionWeight::Dense(embedding))]),
        constants: BTreeMap::from([(TensorId::new(3)?, device.upload(vec![0.25; 128], &[32, 4])?)]),
        rope_axes: BTreeMap::new(),
        rope_frequencies: BTreeMap::new(),
    };
    Ok((graph, weights))
}

#[test]
#[ignore = "requires CUDA hardware; run explicitly inside tools/bench/safe-run.sh"]
fn pooled_verify_preserves_independent_prefixes_and_reuse() -> Result<()> {
    cutile::jit_cache::enable_default().map_err(device_error)?;
    let device = CudaDevice::new(0)?;
    let (graph, weights) = fixture(&device)?;
    let mut pool = SlotPool::new(&device, &graph, &weights, 3, 32, 32, (32, 3))?;
    let mut serial = (0..3)
        .map(|_| DeviceProgram::new(&device, &graph, &weights, 32, 32, 32))
        .collect::<Result<Vec<_>>>()?;
    // Different accepted prefixes, an inactive slot, reversed task order, and a short
    // chain exercise row groups beyond the first four rows and recurrent rollback.
    for plans in [
        vec![
            (0, vec![1, 2, 3], 0),
            (1, vec![7, 8, 9], 1),
            (2, vec![12, 13, 14], 2),
        ],
        vec![(2, vec![15, 16, 17], 0), (0, vec![4, 5], 1)],
        vec![(1, vec![10], 0), (2, vec![18, 19, 20], 2)],
    ] {
        let mut lanes = Vec::new();
        let mut expected = Vec::new();
        for (slot, tokens, _) in &plans {
            let position = serial[*slot].position();
            for (offset, &token) in tokens.iter().enumerate() {
                lanes.push((*slot, offset, token, position + offset, position + offset));
            }
            expected.push(serial[*slot].step_batch(tokens, position)?);
        }
        let actual = pool.run_verify(&device, &lanes)?;
        for ((slot, tokens, accepted), reference) in plans.into_iter().zip(expected) {
            compare(&actual[slot * 3..][..tokens.len()], &reference);
            pool.commit_verify(&device, slot, accepted)?;
            serial[slot].commit_batch(accepted + 1)?;
            assert_eq!(pool.cursors()[slot], serial[slot].position());
        }
    }
    serial[1].reset()?;
    serial[1].step(21, 0, 0, None, false)?;
    pool.bind(1, serial[1].states(), serial[1].fp8_states(), 1, &device)?;
    let expected = serial[1].step_batch(&[22, 23, 24], 1)?;
    let actual = pool.run_verify(
        &device,
        &[(1, 0, 22, 1, 1), (1, 1, 23, 2, 2), (1, 2, 24, 3, 3)],
    )?;
    compare(&actual[3..6], &expected);
    pool.commit_verify(&device, 1, 2)?;
    Ok(())
}

fn compare(actual: &[(Vec<f32>, Vec<f32>)], expected: &[(Vec<f32>, Vec<f32>)]) {
    for (a, b) in actual.iter().zip(expected) {
        for (x, y) in a.0.iter().chain(&a.1).zip(b.0.iter().chain(&b.1)) {
            assert!((x - y).abs() < 0.00002, "{x} != {y}");
        }
    }
}

#[test]
fn verify_lanes_rejects_holes_duplicates_and_invalid_positions() {
    let cursors = [2, 5, 7];
    assert!(verify_lanes(&[(1, 1, 3, 6, 6)], &cursors, 3, 16, 32).is_err());
    assert!(verify_lanes(&[(0, 0, 1, 2, 2); 2], &cursors, 3, 16, 32).is_err());
    assert!(verify_lanes(&[(usize::MAX, 0, 1, 2, 2)], &cursors, 3, 16, 32).is_err());
    assert!(verify_lanes(&[(0, 0, 1, 3, 3)], &cursors, 3, 16, 32).is_err());
    assert!(verify_lanes(&[(0, 0, 32, 2, 2)], &cursors, 3, 16, 32).is_err());
    assert!(verify_lanes(&[(0, 0, 1, 2, 1)], &cursors, 3, 16, 32).is_err());
    assert!(verify_lanes(&[(2, 0, 1, 7, 7)], &cursors, 3, 7, 32).is_err());
    assert!(verify_lanes(&[(2, 0, 1, 7, 7), (0, 0, 2, 2, 2)], &cursors, 3, 16, 32).is_ok());
}
