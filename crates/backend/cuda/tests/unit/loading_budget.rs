use super::*;
use infer_core::{OpId, TensorId};
use infer_ir::{DType, TensorNode, TensorSpec};
use std::collections::BTreeMap;

fn fixture_weights() -> ProgramWeights {
    ProgramWeights {
        autotune: false,
        resident_tuning: std::sync::Arc::new(
            crate::tuning::resident_gemm::ResidentTuningSession::default(),
        ),
        chunked_recurrent: false,
        batch_width: 3,
        prefill_width: 32,
        narrow_prefill_width: 0,
        fusion: None,
        projections: BTreeMap::new(),
        fp8_inputs: std::collections::BTreeSet::new(),
        input_scales: BTreeMap::new(),
        kv_scales: BTreeMap::new(),
        tiling: BTreeMap::new(),
        expert_banks: BTreeMap::new(),
        constants: BTreeMap::new(),
        embeddings: BTreeMap::new(),
        rope_axes: BTreeMap::new(),
        rope_frequencies: BTreeMap::new(),
    }
}

#[test]
fn admission_charges_shared_projection_geometry_once() -> Result<()> {
    let mut weights = fixture_weights();
    let mut graph = DataflowGraph::default();
    for (id, size) in [(1, 128), (2, 256), (3, 128), (4, 256)] {
        graph.tensors.push(TensorSpec {
            id: TensorId::new(id)?,
            shape: vec![size],
            dtype: DType::F32,
            storage: TensorStorage::Activation,
        });
    }
    for (id, input, output, weight) in [(1, 1, 2, 10), (2, 3, 4, 11)] {
        graph.nodes.push(TensorNode {
            id: OpId::new(id)?,
            layer: None,
            op: TensorOp::Linear,
            inputs: vec![TensorId::new(input)?, TensorId::new(weight)?],
            outputs: vec![TensorId::new(output)?],
            states: vec![],
        });
    }
    // Repeated layers share one [3, K] / [3, N] pair. Prefill has zero-copy views.
    assert_eq!(
        projection_workspace(&graph, &weights, ProgramGeometry::new(&weights, true))?,
        (128 + 256) * 3 * F32
    );
    weights.input_scales.insert(TensorId::new(10)?, 1.0);
    weights.input_scales.insert(TensorId::new(11)?, 1.0);
    assert_eq!(
        projection_workspace(&graph, &weights, ProgramGeometry::new(&weights, true))?,
        (128 + 256) * 3 * F32 + (128 / 2 + 128 / 16) * (1 + 3 + 32)
    );
    // Wider verification batches also share one pack/unpack pair per geometry.
    weights.batch_width = 5;
    assert_eq!(
        projection_workspace(&graph, &weights, ProgramGeometry::new(&weights, true))?,
        (128 + 256) * 5 * F32 + (128 / 2 + 128 / 16) * (1 + 5 + 32)
    );
    // Without a private verification graph its pack/unpack pairs and staging rows vanish.
    assert_eq!(
        projection_workspace(&graph, &weights, ProgramGeometry::new(&weights, false))?,
        (128 / 2 + 128 / 16) * (1 + 32)
    );
    Ok(())
}

#[test]
fn kv_block_bytes_counts_attention_states_in_storage_bytes() -> Result<()> {
    let mut weights = fixture_weights();
    let mut graph = DataflowGraph::default();
    let kv = |id| {
        Ok(TensorSpec {
            id: TensorId::new(id)?,
            // [capacity, kv_heads * head_dim], one spec covering keys and values.
            shape: vec![64, 512],
            dtype: DType::F32,
            storage: TensorStorage::State {
                layer: usize::try_from(id).expect("fixture layer index fits usize"),
                kind: StateKind::AttentionKv,
            },
        })
    };
    graph.tensors.push(kv(1)?);
    graph.tensors.push(kv(2)?);
    // A conv state and an activation are not KV and must not be priced per block.
    graph.tensors.push(TensorSpec {
        id: TensorId::new(3)?,
        shape: vec![128, 4],
        dtype: DType::F32,
        storage: TensorStorage::State {
            layer: 3,
            kind: StateKind::Conv,
        },
    });
    graph.tensors.push(TensorSpec {
        id: TensorId::new(4)?,
        shape: vec![512],
        dtype: DType::F32,
        storage: TensorStorage::Activation,
    });
    let block = crate::constants::KV_BLOCK_TOKENS as u64;
    // Unscaled: 2 layers * 32 tokens * 512 columns * (K+V) * f32.
    assert_eq!(kv_block_bytes(&graph, &weights)?, 2 * block * 512 * 2 * F32);
    // An fp8-packed cache stores one byte per element.
    weights.kv_scales.insert(TensorId::new(1)?, [1.0, 1.0]);
    assert_eq!(
        kv_block_bytes(&graph, &weights)?,
        block * 512 * 2 * (F32 + 1)
    );
    Ok(())
}

fn activation_graph(width: usize) -> Result<DataflowGraph> {
    let embedding = TensorId::ONE;
    let hidden = TensorId::new(2)?;
    let logits = TensorId::new(3)?;
    Ok(DataflowGraph {
        tensors: vec![
            TensorSpec {
                id: embedding,
                shape: vec![width, width],
                dtype: DType::Bf16,
                storage: TensorStorage::Weight {
                    slot: "embedding".into(),
                },
            },
            TensorSpec {
                id: hidden,
                shape: vec![width],
                dtype: DType::F32,
                storage: TensorStorage::Activation,
            },
            TensorSpec {
                id: logits,
                shape: vec![width],
                dtype: DType::F32,
                storage: TensorStorage::Activation,
            },
        ],
        nodes: vec![
            TensorNode {
                id: OpId::ONE,
                layer: None,
                op: TensorOp::Embedding,
                inputs: vec![embedding],
                outputs: vec![hidden],
                states: vec![],
            },
            TensorNode {
                id: OpId::new(2)?,
                layer: None,
                op: TensorOp::Add,
                inputs: vec![hidden, hidden],
                outputs: vec![logits],
                states: vec![],
            },
        ],
        hidden: Some(hidden),
        logits: Some(logits),
        ..Default::default()
    })
}

#[test]
fn pooled_admission_excludes_uncaptured_wide_prompt_and_private_readback() -> Result<()> {
    let mut weights = fixture_weights();
    weights.prefill_width = 256;
    weights.narrow_prefill_width = 64;
    let graph = activation_graph(48)?;
    let quote = |private| {
        program(
            &graph,
            &weights,
            256,
            48,
            48,
            OutputReadout::Logits,
            ProgramGeometry::new(&weights, private),
        )
    };
    // Two live F32 activation rows per graph lane; external rows retain their configured
    // allocation, and prompt readback stages only its final row.
    let pooled = (1 + 64) * (2 * 48) * F32 + (1 + 256) * 48 * F32 + 2 * (48 + 48) * F32;
    assert_eq!(quote(false)?, pooled);
    // Private capture adds the wide prompt arena, three verification arena rows and
    // three readback rows. Its narrow prompt is shared with the pooled geometry.
    assert_eq!(
        quote(true)? - quote(false)?,
        (256 + 3) * (2 * 48) * F32 + 3 * (48 + 48) * F32
    );
    Ok(())
}

#[test]
fn pooled_admission_keeps_verification_required_by_prompt_fallback() -> Result<()> {
    let mut weights = fixture_weights();
    weights.prefill_width = 3;
    let graph = activation_graph(48)?;
    let quote = |private| {
        program(
            &graph,
            &weights,
            256,
            48,
            48,
            OutputReadout::Logits,
            ProgramGeometry::new(&weights, private),
        )
    };
    // Without a prompt graph, prefill uses the private verification graph even when the
    // caller requests pooled decode; its activations and readback must remain charged.
    let expected = (1 + 3) * (2 * 48) * F32 + (1 + 3) * 48 * F32 + (1 + 3) * (48 + 48) * F32;
    assert_eq!(quote(false)?, expected);
    assert_eq!(quote(true)?, expected);
    Ok(())
}
