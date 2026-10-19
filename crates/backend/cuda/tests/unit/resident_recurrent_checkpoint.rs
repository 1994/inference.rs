use super::checkpoint;
use crate::{device::CudaDevice, resident::recurrent::recurrent};
use cutile::prelude::*;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn upload<T: DType>(
    device: &CudaDevice,
    values: Vec<T>,
    shape: &[usize],
) -> infer_core::Result<Tensor<T>> {
    Arc::try_unwrap(device.upload(values, shape)?)
        .map_err(|_| infer_core::Error::invariant("checkpoint test upload ownership"))
}

fn values(count: usize, salt: usize) -> Vec<f32> {
    (0..count)
        .map(|index| f32::from(u16::try_from((index * salt + 3) % 97).unwrap() + 1) / 193.0 - 0.251)
        .collect()
}

fn assert_bits(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "{label} index={index}"
        );
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label} index={index}"
        );
    }
}

struct ConvBuffers {
    original: Vec<Tensor<f32>>,
    candidate: Vec<Tensor<f32>>,
    snapshots: Vec<Tensor<f32>>,
    original_output: Tensor<f32>,
    candidate_output: Tensor<f32>,
}

fn run_conv(
    device: &CudaDevice,
    buffers: &mut ConvBuffers,
    input: &Tensor<f32>,
    weights: &Tensor<f32>,
    metadata: &Tensor<i32>,
) -> TestResult {
    let [h0, h1, h2] = buffers.original.as_mut_slice() else {
        return Err("original convolution fixture layout".into());
    };
    recurrent::conv4(
        (&mut buffers.original_output).partition([crate::constants::CONV_KERNEL_TILE]),
        h0.partition([crate::constants::CONV_KERNEL_TILE]),
        h1.partition([crate::constants::CONV_KERNEL_TILE]),
        h2.partition([crate::constants::CONV_KERNEL_TILE]),
        input,
        weights,
        metadata,
    )
    .sync_on(&device.stream)?;
    let [h0, h1, h2] = buffers.candidate.as_mut_slice() else {
        return Err("candidate convolution fixture layout".into());
    };
    let [c0, c1, c2] = buffers.snapshots.as_mut_slice() else {
        return Err("convolution checkpoint fixture layout".into());
    };
    checkpoint::conv4_checkpoint(
        (&mut buffers.candidate_output).partition([crate::constants::CONV_KERNEL_TILE]),
        h0.partition([crate::constants::CONV_KERNEL_TILE]),
        h1.partition([crate::constants::CONV_KERNEL_TILE]),
        h2.partition([crate::constants::CONV_KERNEL_TILE]),
        c0.partition([crate::constants::CONV_KERNEL_TILE]),
        c1.partition([crate::constants::CONV_KERNEL_TILE]),
        c2.partition([crate::constants::CONV_KERNEL_TILE]),
        input,
        weights,
        metadata,
    )
    .sync_on(&device.stream)?;
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run explicitly inside tools/bench/safe-run.sh"]
fn fused_conv_checkpoints_preserve_state_output_and_inactive_tail() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for channels in [32, 128, 257] {
        let initial: Vec<Vec<f32>> = (0..3).map(|index| values(channels, index + 3)).collect();
        let upload_states = |host: &[Vec<f32>]| {
            host.iter()
                .map(|row| upload(&device, row.clone(), &[channels]))
                .collect::<infer_core::Result<Vec<_>>>()
        };
        let mut previous = vec![vec![17.0; channels]; 3];
        let mut buffers = ConvBuffers {
            original: upload_states(&initial)?,
            candidate: upload_states(&initial)?,
            snapshots: upload_states(&previous)?,
            original_output: upload(&device, vec![0.0; channels], &[channels])?,
            candidate_output: upload(&device, vec![0.0; channels], &[channels])?,
        };
        let weights = upload(&device, values(channels * 4, 11), &[channels, 4])?;
        // Three active tokens with inactive replays between them retain nonzero state.
        for (step, position) in [0, -1, 1, -1, 2].into_iter().enumerate() {
            let input = upload(&device, values(channels, step + 17), &[channels])?;
            let metadata = upload(&device, vec![0, 0, position, 0], &[4])?;
            run_conv(&device, &mut buffers, &input, &weights, &metadata)?;
            assert_bits(
                "conv output",
                &device.read_borrowed(&buffers.candidate_output)?,
                &device.read_borrowed(&buffers.original_output)?,
            );
            for (index, ((candidate, original), snapshot)) in buffers
                .candidate
                .iter()
                .zip(&buffers.original)
                .zip(&buffers.snapshots)
                .enumerate()
            {
                let candidate = device.read_borrowed(candidate)?;
                assert_bits("conv state", &candidate, &device.read_borrowed(original)?);
                if position >= 0 {
                    previous[index] = candidate;
                }
                assert_bits(
                    "conv checkpoint",
                    &device.read_borrowed(snapshot)?,
                    &previous[index],
                );
            }
        }
    }
    Ok(())
}

struct DeltaBuffers {
    original_state: Tensor<f32>,
    candidate_state: Tensor<f32>,
    checkpoint: Tensor<f32>,
    original_output: Tensor<f32>,
    candidate_output: Tensor<f32>,
}

struct DeltaInputs {
    qkv: Tensor<f32>,
    beta: Tensor<f32>,
    alpha: Tensor<f32>,
    a_log: Tensor<f32>,
    bias: Tensor<f32>,
    metadata: Tensor<i32>,
}

fn run_delta(
    device: &CudaDevice,
    buffers: &mut DeltaBuffers,
    inputs: &DeltaInputs,
    (key_heads, value_heads, dim, block): (usize, usize, usize, usize),
) -> TestResult {
    let generics = vec![
        key_heads.to_string(),
        value_heads.to_string(),
        dim.to_string(),
        block.to_string(),
    ];
    recurrent::delta(
        (&mut buffers.original_output).partition([1, 1, block]),
        (&mut buffers.original_state).partition([1, dim, block]),
        &inputs.qkv,
        &inputs.beta,
        &inputs.alpha,
        &inputs.a_log,
        &inputs.bias,
        &inputs.metadata,
    )
    .generics(generics.clone())
    .sync_on(&device.stream)?;
    checkpoint::delta_checkpoint(
        (&mut buffers.candidate_output).partition([1, 1, block]),
        (&mut buffers.candidate_state).partition([1, dim, block]),
        (&mut buffers.checkpoint).partition([1, dim, block]),
        &inputs.qkv,
        &inputs.beta,
        &inputs.alpha,
        &inputs.a_log,
        &inputs.bias,
        &inputs.metadata,
    )
    .generics(generics)
    .sync_on(&device.stream)?;
    Ok(())
}

fn check_delta(
    device: &CudaDevice,
    (key_heads, value_heads, dim): (usize, usize, usize),
    split: usize,
) -> TestResult {
    let state_size = value_heads * dim * dim;
    let initial = values(state_size, 23);
    let mut previous = vec![17.0; state_size];
    let mut buffers = DeltaBuffers {
        original_state: upload(device, initial.clone(), &[value_heads, dim, dim])?,
        candidate_state: upload(device, initial, &[value_heads, dim, dim])?,
        checkpoint: upload(device, previous.clone(), &[value_heads, dim, dim])?,
        original_output: upload(device, vec![0.0; value_heads * dim], &[value_heads, 1, dim])?,
        candidate_output: upload(device, vec![0.0; value_heads * dim], &[value_heads, 1, dim])?,
    };
    // The original and checkpoint kernels see identical operands, including inactive metadata.
    for (step, position) in [0, -1, 1, -1, 2].into_iter().enumerate() {
        let qkv_columns = (2 * key_heads + value_heads) * dim;
        let inputs = DeltaInputs {
            qkv: upload(device, values(qkv_columns, step + 17), &[qkv_columns])?,
            beta: upload(device, values(value_heads, step + 13), &[value_heads])?,
            alpha: upload(device, values(value_heads, step + 7), &[value_heads])?,
            a_log: upload(device, values(value_heads, 3), &[value_heads])?,
            bias: upload(device, values(value_heads, 5), &[value_heads])?,
            metadata: upload(device, vec![0, 0, position, 0], &[4])?,
        };
        run_delta(
            device,
            &mut buffers,
            &inputs,
            (key_heads, value_heads, dim, dim / split),
        )?;
        assert_bits(
            "delta output",
            &device.read_borrowed(&buffers.candidate_output)?,
            &device.read_borrowed(&buffers.original_output)?,
        );
        let candidate = device.read_borrowed(&buffers.candidate_state)?;
        assert_bits(
            "delta state",
            &candidate,
            &device.read_borrowed(&buffers.original_state)?,
        );
        if position >= 0 {
            previous = candidate;
        }
        assert_bits(
            "delta checkpoint",
            &device.read_borrowed(&buffers.checkpoint)?,
            &previous,
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run explicitly inside tools/bench/safe-run.sh"]
fn fused_delta_checkpoints_preserve_nonzero_state_and_split_arithmetic() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for geometry in [(1, 2, 32), (16, 48, 128)] {
        for split in [1, 4] {
            check_delta(&device, geometry, split)?;
        }
    }
    Ok(())
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 1e-5 * b.abs().max(1.0),
            "{label}[{index}]: {a} != {b}"
        );
    }
}

/// Three lanes' inputs of one verification step: `qkv`, `beta`, `alpha` and metadata each.
fn lane_inputs(
    device: &CudaDevice,
    (key_heads, value_heads, dim): (usize, usize, usize),
    lane: usize,
    position: i32,
) -> infer_core::Result<DeltaInputs> {
    let qkv_columns = (2 * key_heads + value_heads) * dim;
    Ok(DeltaInputs {
        qkv: upload(device, values(qkv_columns, lane + 17), &[qkv_columns])?,
        beta: upload(device, values(value_heads, lane + 13), &[value_heads])?,
        alpha: upload(device, values(value_heads, lane + 7), &[value_heads])?,
        a_log: upload(device, values(value_heads, 3), &[value_heads])?,
        bias: upload(device, values(value_heads, 5), &[value_heads])?,
        metadata: upload(device, vec![0, 0, position, 0], &[4])?,
    })
}

#[test]
#[ignore = "requires CUDA hardware; run explicitly inside tools/bench/safe-run.sh"]
#[expect(
    clippy::too_many_lines,
    reason = "Reference and fused launches bind every lane explicitly"
)]
fn three_lane_delta_matches_sequential_checkpoints() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (key_heads, value_heads, dim) in [(1, 2, 32), (16, 48, 128)] {
        for split in [1, 4] {
            let block = dim / split;
            let generics = vec![
                key_heads.to_string(),
                value_heads.to_string(),
                dim.to_string(),
                block.to_string(),
            ];
            let state_size = value_heads * dim * dim;
            for positions in [[0, 1, 2], [7, 8, -1], [-1, -1, -1]] {
                let lanes = positions
                    .iter()
                    .enumerate()
                    .map(|(lane, &p)| lane_inputs(&device, (key_heads, value_heads, dim), lane, p))
                    .collect::<infer_core::Result<Vec<_>>>()?;
                // Reference: the per-lane checkpoint kernel, one lane after another.
                let mut state = upload(&device, values(state_size, 23), &[value_heads, dim, dim])?;
                let mut expected = Vec::new();
                for inputs in &lanes {
                    let mut output = upload(
                        &device,
                        vec![0.0; value_heads * dim],
                        &[value_heads, 1, dim],
                    )?;
                    let mut snapshot =
                        upload(&device, vec![0.0; state_size], &[value_heads, dim, dim])?;
                    checkpoint::delta_checkpoint(
                        (&mut output).partition([1, 1, block]),
                        (&mut state).partition([1, dim, block]),
                        (&mut snapshot).partition([1, dim, block]),
                        &inputs.qkv,
                        &inputs.beta,
                        &inputs.alpha,
                        &inputs.a_log,
                        &inputs.bias,
                        &inputs.metadata,
                    )
                    .generics(generics.clone())
                    .sync_on(&device.stream)?;
                    expected.push((
                        device.read_borrowed(&output)?,
                        device.read_borrowed(&state)?,
                    ));
                }
                let mut fused = upload(&device, values(state_size, 23), &[value_heads, dim, dim])?;
                let mut outputs = (0..3)
                    .map(|_| {
                        upload(
                            &device,
                            vec![0.0; value_heads * dim],
                            &[value_heads, 1, dim],
                        )
                    })
                    .collect::<infer_core::Result<Vec<_>>>()?;
                let mut snapshots = (0..3)
                    .map(|_| upload(&device, vec![0.0; state_size], &[value_heads, dim, dim]))
                    .collect::<infer_core::Result<Vec<_>>>()?;
                let [o0, o1, o2] = outputs.as_mut_slice() else {
                    return Err("three outputs".into());
                };
                let [c0, c1, c2] = snapshots.as_mut_slice() else {
                    return Err("three snapshots".into());
                };
                checkpoint::delta_checkpoint3(
                    o0.partition([1, 1, block]),
                    o1.partition([1, 1, block]),
                    o2.partition([1, 1, block]),
                    (&mut fused).partition([1, dim, block]),
                    c0.partition([1, dim, block]),
                    c1.partition([1, dim, block]),
                    c2.partition([1, dim, block]),
                    &lanes[0].qkv,
                    &lanes[1].qkv,
                    &lanes[2].qkv,
                    &lanes[0].beta,
                    &lanes[1].beta,
                    &lanes[2].beta,
                    &lanes[0].alpha,
                    &lanes[1].alpha,
                    &lanes[2].alpha,
                    &lanes[0].a_log,
                    &lanes[0].bias,
                    &lanes[0].metadata,
                    &lanes[1].metadata,
                    &lanes[2].metadata,
                )
                .generics(generics.clone())
                .sync_on(&device.stream)?;
                for (lane, (output, state)) in expected.iter().enumerate() {
                    let label = format!("{positions:?} split={split} lane={lane}");
                    assert_close(&label, &device.read_borrowed(&outputs[lane])?, output);
                    // A padding lane's snapshot is never restored; the per-lane kernel skips it.
                    if positions[lane] >= 0 {
                        assert_close(&label, &device.read_borrowed(&snapshots[lane])?, state);
                    }
                }
                assert_close(
                    "final state",
                    &device.read_borrowed(&fused)?,
                    &expected[2].1,
                );
            }
        }
    }
    Ok(())
}

// Tile IR is only specialized at launch; compile it on header-only CI as well.
#[test]
fn three_lane_delta_compiles_without_a_device() -> TestResult {
    use cutile::cutile_compiler::compile_api::KernelCompiler;
    let mut strides: Vec<(&str, &[i32])> =
        vec![("state", &[16384, 128, 1]), ("a_log", &[1]), ("bias", &[1])];
    for name in ["out0", "out1", "out2"] {
        strides.push((name, &[128, 128, 1]));
    }
    for name in ["checkpoint0", "checkpoint1", "checkpoint2"] {
        strides.push((name, &[16384, 128, 1]));
    }
    for name in [
        "qkv0",
        "qkv1",
        "qkv2",
        "beta0",
        "beta1",
        "beta2",
        "alpha0",
        "alpha1",
        "alpha2",
        "metadata0",
        "metadata1",
        "metadata2",
    ] {
        strides.push((name, &[1]));
    }
    KernelCompiler::new(
        checkpoint::__module_ast_self,
        "checkpoint",
        "delta_checkpoint3",
    )
    .target("sm_120")
    .generics(vec!["16".into(), "48".into(), "128".into(), "32".into()])
    .strides(&strides)
    .compile()?;
    Ok(())
}
