use crate::{
    device::CudaDevice,
    mlp::ProjectionWeight,
    resident::{DeviceProgram, ProgramWeights},
};
use cutile::half::bf16;
use infer_core::{OpId, Result, TensorId};
use infer_ir::{DType, DataflowGraph, StateKind, TensorNode, TensorOp, TensorSpec, TensorStorage};
use std::collections::{BTreeMap, BTreeSet};

/// Delta geometry for [`fixture`]: key heads, value heads, head dimension.
type DeltaGeometry = (usize, usize, usize);

/// Geometry small enough to reason about by hand.
const SMALL: DeltaGeometry = (1, 2, 32);

fn fixture(device: &CudaDevice) -> Result<(DataflowGraph, ProgramWeights)> {
    fixture_at(device, SMALL)
}

fn fixture_at(
    device: &CudaDevice,
    (key_heads, value_heads, dim): DeltaGeometry,
) -> Result<(DataflowGraph, ProgramWeights)> {
    // Activation tensors are declared with one lane's element count; the arena multiplies by the
    // captured width. The Delta binds `[lanes, 2*kh+vh, dim]` for qkv, `[lanes, vh]` for
    // beta/alpha and `[vh, dim, dim]` for the state, so the declared sizes follow from these.
    let qkv_columns = (2 * key_heads + value_heads) * dim;
    let gate_columns = value_heads;
    let out_columns = value_heads * dim;
    let mut graph = DataflowGraph::default();
    for (id, shape, kind) in [
        (1, vec![16, qkv_columns], 0),
        (2, vec![qkv_columns], 1),
        (3, vec![16, gate_columns], 0),
        (4, vec![gate_columns], 1),
        (5, vec![16, gate_columns], 0),
        (6, vec![gate_columns], 1),
        (7, vec![value_heads], 0),
        (8, vec![value_heads], 0),
        (9, vec![value_heads, dim, dim], 2),
        (10, vec![out_columns], 1),
        (11, vec![16, out_columns], 0),
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
                key_heads,
                value_heads,
                key_dim: dim,
                value_dim: dim,
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
    for (id, columns) in [(1, qkv_columns), (3, gate_columns), (5, gate_columns)] {
        let data = super::tests::values(16 * columns, usize::try_from(id).unwrap() + 2)
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        embeddings.insert(TensorId::new(id)?, device.upload(data, &[16, columns])?);
    }
    let dense = device.upload(
        super::tests::values(16 * out_columns, 11)
            .into_iter()
            .map(bf16::from_f32)
            .collect(),
        &[16, out_columns],
    )?;
    let weights = ProgramWeights {
        chunked_recurrent: false,
        batch_width: 3,
        prefill_width: 1,
        narrow_prefill_width: 0,
        fusion: None,
        embeddings,
        projections: BTreeMap::from([(TensorId::new(11)?, ProjectionWeight::Dense(dense))]),
        constants: BTreeMap::from([
            (
                TensorId::new(7)?,
                device.upload(super::tests::values(value_heads, 3), &[value_heads])?,
            ),
            (
                TensorId::new(8)?,
                device.upload(super::tests::values(value_heads, 5), &[value_heads])?,
            ),
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

/// The chunked recurrence disagrees with the per-lane one on the 27B (5% of hidden L2) while
/// both kernels reproduce an independent reference on identical inputs, so the difference has to
/// be in what the capture binds. This runs the same capture comparison at the geometry the 27B
/// actually uses, which needs no checkpoint and finishes in milliseconds — a fixture this cheap
/// is the place to instrument until the binding is found.
#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn chunked_recurrence_matches_per_lane_at_model_geometry() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for width in [32, 128] {
        let (graph, mut weights) = fixture_at(&device, (16, 48, 128))?;
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
                let error = actual[i]
                    .0
                    .iter()
                    .zip(&expected.0)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                eprintln!(
                    "width={width} count={count} lane={i} chunked-vs-per-lane max_abs={error:e}"
                );
                close(&actual[i].0, &expected.0);
            }
            position += count;
        }
    }
    Ok(())
}
