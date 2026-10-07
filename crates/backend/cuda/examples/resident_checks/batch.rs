use infer_backend_cuda::resident::DeviceProgram;
use infer_core::{Error, Result};
use infer_ir::TensorOp;
use infer_state::physical::PhysicalTensor;

pub fn run(
    program: &mut DeviceProgram,
    op: &TensorOp,
    inputs: &[Vec<f32>],
    initial: Option<&PhysicalTensor>,
) -> Result<()> {
    let mut state = initial.cloned();
    let mut first_checkpoint = None;
    let first = program.step_batch(&[1, 2, 3], 0)?;
    for (position, (hidden, _)) in first.iter().enumerate() {
        let token = u16::try_from(position + 1).map_err(|e| Error::invalid(e.to_string()))?;
        let expected = reference(op, inputs, token, position, &mut state)?;
        compare(hidden, &expected)?;
        if position == 0 {
            first_checkpoint.clone_from(&state);
        }
    }
    program.commit_batch(1)?;
    state = first_checkpoint;
    let second = program.step_batch(&[4, 5], 1)?;
    for (offset, (hidden, _)) in second.iter().enumerate() {
        let token = u16::try_from(offset + 4).map_err(|e| Error::invalid(e.to_string()))?;
        compare(
            hidden,
            &reference(op, inputs, token, offset + 1, &mut state)?,
        )?;
    }
    let (hidden, _) = program.step(6, 3, 3, None, true)?;
    compare(&hidden, &reference(op, inputs, 6, 3, &mut state)?)?;
    if program.position() != 4 || program.commit_batch(1).is_ok() {
        return Err(Error::invariant("batch checkpoint lifetime"));
    }
    prefill(program, op, inputs, initial)?;
    println!("PASS batch/prefix rollback/padded tail/single continuation/prefill: {op:?}");
    Ok(())
}

fn prefill(
    program: &mut DeviceProgram,
    op: &TensorOp,
    inputs: &[Vec<f32>],
    initial: Option<&PhysicalTensor>,
) -> Result<()> {
    let mut state = initial.cloned();
    let width = program.prefill_width();
    for (batch, read_logits) in [false, true].into_iter().enumerate() {
        let start = batch * width;
        let count = if width == 32 && batch == 1 { 17 } else { width };
        let tokens = (start + 1..=start + count)
            .map(|v| u32::try_from(v).map_err(|e| Error::invalid(e.to_string())))
            .collect::<Result<Vec<_>>>()?;
        let outputs = program.prefill_batch(&tokens, start, read_logits)?;
        for (lane, (hidden, logits)) in outputs.iter().enumerate() {
            let position = start + lane;
            let token = u16::try_from(position + 1).map_err(|e| Error::invalid(e.to_string()))?;
            let expected = reference(op, inputs, token, position, &mut state)?;
            compare(hidden, &expected)?;
            if read_logits && lane + 1 == count {
                compare(
                    logits,
                    &expected.iter().map(|v| 2.0 * v).collect::<Vec<_>>(),
                )?;
            } else if !logits.is_empty() {
                return Err(Error::invariant("unexpected prefill logits"));
            }
        }
        if program.commit_batch(1).is_ok() {
            return Err(Error::invariant(
                "prefill must invalidate verification checkpoint",
            ));
        }
    }
    let position = if width == 32 { width + 17 } else { 2 * width };
    let token = u16::try_from(position + 1).map_err(|e| Error::invalid(e.to_string()))?;
    let (hidden, _) = program.step(u32::from(token), position, position, None, true)?;
    compare(
        &hidden,
        &reference(op, inputs, token, position, &mut state)?,
    )
}

fn reference(
    op: &TensorOp,
    inputs: &[Vec<f32>],
    token: u16,
    position: usize,
    state: &mut Option<PhysicalTensor>,
) -> Result<Vec<f32>> {
    let external: Vec<_> = super::values(inputs[0].len(), f32::from(token))
        .into_iter()
        .map(|v| cutile::half::bf16::from_f32(v).to_f32())
        .collect();
    let mut refs: Vec<_> = inputs.iter().map(Vec::as_slice).collect();
    refs[0] = &external;
    if matches!(op, TensorOp::Attention { .. }) {
        refs[1] = &external[..32];
        refs[2] = &external[32..];
    }
    let (keys, values);
    if matches!(
        op,
        TensorOp::Attention {
            window: Some(35),
            ..
        }
    ) {
        keys = super::quantization::round(refs[1], 0.001);
        values = super::quantization::round(refs[2], 0.002);
        refs[1] = &keys;
        refs[2] = &values;
    }
    infer_backend_host::reference_operation(op, &refs, state.as_mut(), u32::from(token), position)?
        .pop()
        .ok_or_else(|| Error::invariant("reference output"))
}

fn compare(actual: &[f32], expected: &[f32]) -> Result<()> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(a, b)| !a.is_finite() || (a - b).abs() > 3e-4 * b.abs().max(1.0))
    {
        return Err(Error::invariant("batched verification mismatch"));
    }
    Ok(())
}

pub fn prompt32(
    device: &infer_backend_cuda::device::CudaDevice,
    graph: &infer_ir::DataflowGraph,
    weights: &mut infer_backend_cuda::resident::ProgramWeights,
    op: &TensorOp,
    inputs: &[Vec<f32>],
    initial: Option<&PhysicalTensor>,
) -> Result<()> {
    weights.prefill_width = 32;
    let mut program = DeviceProgram::new(device, graph, weights, 128, inputs[0].len(), 256)?;
    prefill(&mut program, op, inputs, initial)?;
    if *op == TensorOp::Linear {
        super::rounded_projection::run(device, graph, weights, inputs)?;
    }
    println!("PASS prefill32/tail/decode: {op:?}");
    Ok(())
}
