use super::*;
use infer_core::{OpId, TensorId};
use infer_ir::{DType, TensorNode, TensorSpec};
use std::collections::BTreeMap;

#[test]
fn admission_charges_shared_projection_geometry_once() -> Result<()> {
    let mut weights = ProgramWeights {
        batch_width: 3,
        prefill_width: 32,
        narrow_prefill_width: 0,
        fusion: None,
        projections: BTreeMap::new(),
        fp8_inputs: std::collections::BTreeSet::new(),
        input_scales: BTreeMap::new(),
        kv_scales: BTreeMap::new(),
        tiling: BTreeMap::new(),
        constants: BTreeMap::new(),
        embeddings: BTreeMap::new(),
        rope_axes: BTreeMap::new(),
        rope_frequencies: BTreeMap::new(),
    };
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
        projection_workspace(&graph, &weights, true)?,
        (128 + 256) * 3 * F32
    );
    weights.input_scales.insert(TensorId::new(10)?, 1.0);
    weights.input_scales.insert(TensorId::new(11)?, 1.0);
    assert_eq!(
        projection_workspace(&graph, &weights, true)?,
        (128 + 256) * 3 * F32 + (128 / 2 + 128 / 16) * (1 + 3 + 32)
    );
    // Wider verification batches also share one pack/unpack pair per geometry.
    weights.batch_width = 5;
    assert_eq!(
        projection_workspace(&graph, &weights, true)?,
        (128 + 256) * 5 * F32 + (128 / 2 + 128 / 16) * (1 + 5 + 32)
    );
    // Without a private verification graph its pack/unpack pairs and staging rows vanish.
    assert_eq!(
        projection_workspace(&graph, &weights, false)?,
        (128 / 2 + 128 / 16) * (1 + 32)
    );
    Ok(())
}
