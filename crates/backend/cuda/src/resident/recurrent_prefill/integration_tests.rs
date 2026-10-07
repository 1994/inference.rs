use crate::{
    device::CudaDevice,
    mlp::ProjectionWeight,
    resident::{DeviceProgram, ProgramWeights},
};
use cutile::half::bf16;
use infer_core::{OpId, Result, TensorId};
use infer_ir::{DType, DataflowGraph, StateKind, TensorNode, TensorOp, TensorSpec, TensorStorage};
use std::collections::{BTreeMap, BTreeSet};

fn fixture(device: &CudaDevice) -> Result<(DataflowGraph, ProgramWeights)> {
    let mut graph = DataflowGraph::default();
    for (id, shape, kind) in [
        (1, vec![16, 128], 0),
        (2, vec![128], 1),
        (3, vec![16, 2], 0),
        (4, vec![2], 1),
        (5, vec![16, 2], 0),
        (6, vec![2], 1),
        (7, vec![2], 0),
        (8, vec![2], 0),
        (9, vec![2, 32, 32], 2),
        (10, vec![64], 1),
        (11, vec![16, 64], 0),
        (12, vec![16], 1),
    ] {
        let storage = match kind {
            0 => TensorStorage::Weight {
                slot: format!("w{id}"),
            },
            1 => TensorStorage::Activation,
            _ => TensorStorage::State {
                layer: 0,
                kind: StateKind::LinearAttention,
            },
        };
        graph.tensors.push(TensorSpec {
            id: TensorId::new(id)?,
            shape,
            dtype: DType::F32,
            storage,
        });
    }
    for (id, op, inputs, output, state) in [
        (1, TensorOp::Embedding, vec![1], 2, None),
        (2, TensorOp::Embedding, vec![3], 4, None),
        (3, TensorOp::Embedding, vec![5], 6, None),
        (
            4,
            TensorOp::Delta {
                key_heads: 1,
                value_heads: 2,
                key_dim: 32,
                value_dim: 32,
            },
            vec![2, 4, 6, 7, 8],
            10,
            Some(9),
        ),
        (5, TensorOp::Linear, vec![10, 11], 12, None),
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
            states: state
                .into_iter()
                .map(TensorId::new)
                .collect::<Result<_>>()?,
        });
    }
    graph.hidden = Some(TensorId::new(10)?);
    graph.logits = Some(TensorId::new(12)?);
    let mut embeddings = BTreeMap::new();
    for (id, columns) in [(1, 128), (3, 2), (5, 2)] {
        let data = super::tests::values(16 * columns, usize::try_from(id).unwrap() + 2)
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        embeddings.insert(TensorId::new(id)?, device.upload(data, &[16, columns])?);
    }
    let dense = device.upload(
        super::tests::values(16 * 64, 11)
            .into_iter()
            .map(bf16::from_f32)
            .collect(),
        &[16, 64],
    )?;
    let weights = ProgramWeights {
        batch_width: 3,
        prefill_width: 1,
        fusion: None,
        embeddings,
        projections: BTreeMap::from([(TensorId::new(11)?, ProjectionWeight::Dense(dense))]),
        constants: BTreeMap::from([
            (TensorId::new(7)?, device.upload(vec![-0.2, -0.4], &[2])?),
            (TensorId::new(8)?, device.upload(vec![0.1, -0.1], &[2])?),
        ]),
        kv_scales: BTreeMap::new(),
        tiling: BTreeMap::new(),
        fp8_inputs: BTreeSet::new(),
        input_scales: BTreeMap::new(),
        rope_axes: BTreeMap::new(),
        rope_frequencies: BTreeMap::new(),
    };
    Ok((graph, weights))
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn recurrent_prefill_preserves_state_tail_and_verification_rollback() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for width in [32, 128] {
        let (graph, mut weights) = fixture(&device)?;
        let mut reference = DeviceProgram::new(&device, &graph, &weights, 256, 64, 16)?;
        weights.prefill_width = width;
        let mut candidate = DeviceProgram::new(&device, &graph, &weights, 256, 64, 16)?;
        let mut position = 0;
        for count in [width - 3, 1, 2] {
            let tokens: Vec<u32> = (position..position + count)
                .map(|i| u32::try_from(i % 16).unwrap())
                .collect();
            let actual = candidate.prefill_batch(&tokens, position, true)?;
            for (i, &token) in tokens.iter().enumerate() {
                let expected = reference.step(token, position + i, position + i, None, true)?;
                close(&actual[i].0, &expected.0);
            }
            position += count;
        }
        assert!(candidate.discard_verification());
        assert!(!candidate.discard_verification());
        candidate.ensure_verification(&graph, &weights)?;
        let a = candidate.step_batch(&[3, 4, 5], position)?;
        let b = reference.step_batch(&[3, 4, 5], position)?;
        for (a, b) in a.iter().zip(&b) {
            close(&a.0, &b.0);
            close(&a.1, &b.1);
        }
        candidate.commit_batch(1)?;
        reference.commit_batch(1)?;
        assert!(candidate.discard_verification());
        candidate.ensure_verification(&graph, &weights)?;
        let a = candidate.step(7, position + 1, position + 1, None, true)?;
        let b = reference.step(7, position + 1, position + 1, None, true)?;
        close(&a.0, &b.0);
        close(&a.1, &b.1);
    }
    Ok(())
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < 0.00002, "{a} != {b}");
    }
}
