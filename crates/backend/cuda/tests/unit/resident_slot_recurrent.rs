use super::fused;

// Tile IR is only specialized at launch; compile the snapshot-free kernels on header-only CI.
#[test]
fn snapshot_free_recurrent_kernels_compile_without_a_device()
-> Result<(), Box<dyn std::error::Error>> {
    use cutile::cutile_compiler::compile_api::KernelCompiler;
    let lanes = |names: &[&'static str]| -> Vec<(&'static str, &'static [i32])> {
        names.iter().map(|&name| (name, &[1][..])).collect()
    };
    let rows = [
        "qkv0",
        "qkv1",
        "qkv2",
        "beta0",
        "beta1",
        "beta2",
        "alpha0",
        "alpha1",
        "alpha2",
        "a_log",
        "bias",
        "metadata0",
        "metadata1",
        "metadata2",
    ];
    let mut verify = lanes(&rows);
    verify.extend([
        ("out0", &[128, 128, 1][..]),
        ("out1", &[128, 128, 1][..]),
        ("out2", &[128, 128, 1][..]),
        ("state", &[16384, 128, 1][..]),
    ]);
    let mut commit = lanes(&rows);
    commit.push(("state", &[16384, 128, 1][..]));
    let generics = || vec!["16".into(), "48".into(), "128".into(), "32".into()];
    for (entry, strides) in [("delta_verify3", &verify), ("delta_commit3", &commit)] {
        KernelCompiler::new(fused::__module_ast_self, "fused", entry)
            .target("sm_120")
            .generics(generics())
            .strides(strides)
            .compile()?;
    }
    let mut conv_verify = lanes(&[
        "out0",
        "out1",
        "out2",
        "h0",
        "h1",
        "h2",
        "x0",
        "x1",
        "x2",
        "metadata0",
        "metadata1",
        "metadata2",
    ]);
    conv_verify.push(("w", &[4, 1][..]));
    let conv_commit = lanes(&[
        "h0",
        "h1",
        "h2",
        "x0",
        "x1",
        "x2",
        "metadata0",
        "metadata1",
        "metadata2",
    ]);
    for (entry, strides) in [
        ("conv_verify3", &conv_verify),
        ("conv_commit3", &conv_commit),
    ] {
        KernelCompiler::new(fused::__module_ast_self, "fused", entry)
            .target("sm_120")
            .strides(strides)
            .compile()?;
    }
    Ok(())
}

use crate::{device::CudaDevice, resident::recurrent::recurrent};
use cutile::prelude::*;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn upload(
    device: &CudaDevice,
    values: Vec<f32>,
    shape: &[usize],
) -> infer_core::Result<Tensor<f32>> {
    Arc::try_unwrap(device.upload(values, shape)?)
        .map_err(|_| infer_core::Error::invariant("slot recurrent test upload"))
}

fn values(count: usize, salt: usize) -> Vec<f32> {
    (0..count)
        .map(|i| f32::from(u16::try_from((i * salt + 7) % 89).unwrap()) / 97.0 - 0.43)
        .collect()
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

const PATTERNS: [[i32; 3]; 3] = [[4, 5, 6], [9, 10, -1], [-1, -1, -1]];

#[test]
#[ignore = "requires CUDA hardware; run explicitly inside tools/bench/safe-run.sh"]
#[expect(
    clippy::too_many_lines,
    reason = "Reference and fused launches bind every lane explicitly"
)]
fn conv_verify_and_commit_match_sequential_lanes() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let channels = 256;
    let tile = crate::constants::CONV_KERNEL_TILE;
    let w = upload(&device, values(channels * 4, 5), &[channels, 4])?;
    let inputs = (0..3)
        .map(|lane| upload(&device, values(channels, lane + 11), &[channels]))
        .collect::<infer_core::Result<Vec<_>>>()?;
    for positions in PATTERNS {
        let metadata = positions
            .iter()
            .map(|&p| device.upload(vec![p, 0, p, 0], &[4]))
            .collect::<infer_core::Result<Vec<_>>>()?;
        for count in 1..=3usize {
            // Reference: conv4 lane after lane, keeping the state after `count` active lanes.
            let mut state = (0..3)
                .map(|h| upload(&device, values(channels, h + 3), &[channels]))
                .collect::<infer_core::Result<Vec<_>>>()?;
            let mut expected_out = Vec::new();
            let mut expected_state = None;
            for lane in 0..3 {
                let mut out = upload(&device, vec![0.0; channels], &[channels])?;
                let [h0, h1, h2] = state.as_mut_slice() else {
                    unreachable!()
                };
                recurrent::conv4(
                    (&mut out).partition([tile]),
                    h0.partition([tile]),
                    h1.partition([tile]),
                    h2.partition([tile]),
                    &inputs[lane],
                    &w,
                    &metadata[lane],
                )
                .sync_on(&device.stream)?;
                expected_out.push(device.read_borrowed(&out)?);
                if lane + 1 == count {
                    expected_state = Some(
                        state
                            .iter()
                            .map(|h| device.read_borrowed(h))
                            .collect::<infer_core::Result<Vec<_>>>()?,
                    );
                }
            }
            let mut fused_state = (0..3)
                .map(|h| upload(&device, values(channels, h + 3), &[channels]))
                .collect::<infer_core::Result<Vec<_>>>()?;
            let mut outs = (0..3)
                .map(|_| upload(&device, vec![0.0; channels], &[channels]))
                .collect::<infer_core::Result<Vec<_>>>()?;
            {
                let [o0, o1, o2] = outs.as_mut_slice() else {
                    unreachable!()
                };
                let [h0, h1, h2] = fused_state.as_slice() else {
                    unreachable!()
                };
                fused::conv_verify3(
                    o0.partition([tile]),
                    o1.partition([tile]),
                    o2.partition([tile]),
                    h0,
                    h1,
                    h2,
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    &w,
                    &metadata[0],
                    &metadata[1],
                    &metadata[2],
                )
                .sync_on(&device.stream)?;
            }
            for (lane, out) in outs.iter().enumerate() {
                assert_close(
                    &format!("{positions:?} conv out {lane}"),
                    &device.read_borrowed(out)?,
                    &expected_out[lane],
                );
            }
            {
                let [h0, h1, h2] = fused_state.as_mut_slice() else {
                    unreachable!()
                };
                fused::conv_commit3(
                    h0.partition([tile]),
                    h1.partition([tile]),
                    h2.partition([tile]),
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    &metadata[0],
                    &metadata[1],
                    &metadata[2],
                    i32::try_from(count)?,
                )
                .sync_on(&device.stream)?;
            }
            let expected_state = expected_state.ok_or("conv reference state")?;
            for (h, expected) in fused_state.iter().zip(&expected_state) {
                assert_close(
                    &format!("{positions:?} count {count} conv state"),
                    &device.read_borrowed(h)?,
                    expected,
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run explicitly inside tools/bench/safe-run.sh"]
#[expect(
    clippy::too_many_lines,
    reason = "Reference and fused launches bind every lane explicitly"
)]
fn delta_verify_and_commit_match_sequential_lanes() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (key_heads, value_heads, dim) = (2usize, 4usize, 32usize);
    let generics = vec![
        key_heads.to_string(),
        value_heads.to_string(),
        dim.to_string(),
        dim.to_string(),
    ];
    let qkv_len = (2 * key_heads + value_heads) * dim;
    let state_len = value_heads * dim * dim;
    let lanes = (0..3)
        .map(|lane| {
            Ok((
                upload(&device, values(qkv_len, lane + 17), &[qkv_len])?,
                upload(&device, values(value_heads, lane + 13), &[value_heads])?,
                upload(&device, values(value_heads, lane + 7), &[value_heads])?,
            ))
        })
        .collect::<infer_core::Result<Vec<_>>>()?;
    let a_log = upload(&device, values(value_heads, 3), &[value_heads])?;
    let bias = upload(&device, values(value_heads, 5), &[value_heads])?;
    for positions in PATTERNS {
        let metadata = positions
            .iter()
            .map(|&p| device.upload(vec![p, 0, p, 0], &[4]))
            .collect::<infer_core::Result<Vec<_>>>()?;
        for count in 1..=3usize {
            let mut state = upload(&device, values(state_len, 23), &[value_heads, dim, dim])?;
            let mut expected_out = Vec::new();
            let mut expected_state = None;
            for (lane, (qkv, beta, alpha)) in lanes.iter().enumerate() {
                let mut out = upload(
                    &device,
                    vec![0.0; value_heads * dim],
                    &[value_heads, 1, dim],
                )?;
                recurrent::delta(
                    (&mut out).partition([1, 1, dim]),
                    (&mut state).partition([1, dim, dim]),
                    qkv,
                    beta,
                    alpha,
                    &a_log,
                    &bias,
                    &metadata[lane],
                )
                .generics(generics.clone())
                .sync_on(&device.stream)?;
                expected_out.push(device.read_borrowed(&out)?);
                if lane + 1 == count {
                    expected_state = Some(device.read_borrowed(&state)?);
                }
            }
            let mut fused_state = upload(&device, values(state_len, 23), &[value_heads, dim, dim])?;
            let mut outs = (0..3)
                .map(|_| {
                    upload(
                        &device,
                        vec![0.0; value_heads * dim],
                        &[value_heads, 1, dim],
                    )
                })
                .collect::<infer_core::Result<Vec<_>>>()?;
            {
                let [o0, o1, o2] = outs.as_mut_slice() else {
                    unreachable!()
                };
                fused::delta_verify3(
                    o0.partition([1, 1, dim]),
                    o1.partition([1, 1, dim]),
                    o2.partition([1, 1, dim]),
                    &fused_state,
                    &lanes[0].0,
                    &lanes[1].0,
                    &lanes[2].0,
                    &lanes[0].1,
                    &lanes[1].1,
                    &lanes[2].1,
                    &lanes[0].2,
                    &lanes[1].2,
                    &lanes[2].2,
                    &a_log,
                    &bias,
                    &metadata[0],
                    &metadata[1],
                    &metadata[2],
                )
                .generics(generics.clone())
                .sync_on(&device.stream)?;
            }
            for (lane, out) in outs.iter().enumerate() {
                assert_close(
                    &format!("{positions:?} delta out {lane}"),
                    &device.read_borrowed(out)?,
                    &expected_out[lane],
                );
            }
            fused::delta_commit3(
                (&mut fused_state).partition([1, dim, dim]),
                &lanes[0].0,
                &lanes[1].0,
                &lanes[2].0,
                &lanes[0].1,
                &lanes[1].1,
                &lanes[2].1,
                &lanes[0].2,
                &lanes[1].2,
                &lanes[2].2,
                &a_log,
                &bias,
                &metadata[0],
                &metadata[1],
                &metadata[2],
                i32::try_from(count)?,
            )
            .generics(generics.clone())
            .sync_on(&device.stream)?;
            let expected_state = expected_state.ok_or("delta reference state")?;
            assert_close(
                &format!("{positions:?} count {count} delta state"),
                &device.read_borrowed(&fused_state)?,
                &expected_state,
            );
        }
    }
    Ok(())
}
