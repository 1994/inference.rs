//! Validate BF16 activation rounding after a non-linear F32 operation.
use infer_backend_cuda::{
    device::CudaDevice,
    resident::{DeviceProgram, ProgramWeights},
};
use infer_core::{Error, OpId, Result, TensorId};
use infer_ir::{DType, DataflowGraph, TensorNode, TensorOp, TensorSpec, TensorStorage};

pub fn run(
    device: &CudaDevice,
    original: &DataflowGraph,
    weights: &ProgramWeights,
    inputs: &[Vec<f32>],
) -> Result<()> {
    let mut graph = original.clone();
    let index = graph
        .nodes
        .iter()
        .position(|n| n.op == TensorOp::Linear)
        .ok_or_else(|| Error::invariant("projection test node"))?;
    let input = graph.nodes[index].inputs[0];
    let nonlinear = TensorId::new(104)?;
    graph.nodes[index].inputs[0] = nonlinear;
    graph.nodes.insert(
        index,
        TensorNode {
            id: OpId::new(77)?,
            layer: None,
            op: TensorOp::Silu,
            inputs: vec![input],
            outputs: vec![nonlinear],
            states: vec![],
        },
    );
    graph.tensors.push(TensorSpec {
        id: nonlinear,
        shape: vec![inputs[0].len()],
        dtype: DType::F32,
        storage: TensorStorage::Activation,
    });
    let mut program = DeviceProgram::new(device, &graph, weights, 128, inputs[0].len(), 256)?;
    let tokens: Vec<u32> = (1..=17).collect();
    let outputs = program.prefill_batch(&tokens, 0, true)?;
    for (token, (hidden, _)) in (1u16..=17).zip(outputs) {
        let nonlinear: Vec<_> = super::values(inputs[0].len(), f32::from(token))
            .into_iter()
            .map(|x| cutile::half::bf16::from_f32(x).to_f32())
            .map(|x| x / (1.0 + (-x).exp()))
            .collect();
        let rounded: Vec<_> = nonlinear
            .iter()
            .copied()
            .map(|x| cutile::half::bf16::from_f32(x).to_f32())
            .collect();
        if nonlinear == rounded {
            return Err(Error::invariant("rounding fixture is degenerate"));
        }
        let expected = infer_backend_host::reference_operation(
            &TensorOp::Linear,
            &[&rounded, &inputs[1]],
            None,
            0,
            0,
        )?;
        let expected = &expected[0];
        if hidden.len() != expected.len()
            || hidden
                .iter()
                .zip(expected)
                .any(|(a, b)| !a.is_finite() || (a - b).abs() > 3e-4 * b.abs().max(1.0))
        {
            return Err(Error::invariant("BF16 rounded projection mismatch"));
        }
    }
    println!("PASS prefill32 BF16 rounding after F32 SiLU, nonuniform weights and partial tile");
    Ok(())
}
