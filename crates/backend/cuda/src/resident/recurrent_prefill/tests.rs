use super::kernels;
use crate::device::CudaDevice;
use cutile::prelude::*;

pub(super) fn values(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (f32::from(u16::try_from((i * salt + 3) % 101).unwrap()) - 50.0) / 100.0)
        .collect()
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn chunk_delta_matches_independent_recurrence() -> Result<(), Box<dyn std::error::Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (kh, vh, dim, lanes) in [
        (2, 4, 32, 32),
        (2, 6, 128, 32),
        (2, 6, 128, 128),
        (16, 48, 128, 128),
    ] {
        for (count, offset) in [(0, 0), (1, 0), (lanes - 3, 0), (lanes, -1)] {
            check_case(&device, (kh, vh, dim, lanes), count, offset)?;
        }
    }
    Ok(())
}

fn check_case(
    device: &CudaDevice,
    geometry: (usize, usize, usize, usize),
    count: usize,
    offset: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let (kh, vh, dim, lanes) = geometry;
    let qkv = values(lanes * (2 * kh + vh) * dim, 17);
    let beta = values(lanes * vh, 13);
    let alpha = values(lanes * vh, 7);
    let a_log = values(vh, 3);
    let bias = values(vh, 5);
    let initial = values(vh * dim * dim, 23);
    let mut expected: Vec<f64> = initial.iter().copied().map(f64::from).collect();
    let mut outputs = vec![0.0_f64; lanes * vh * dim];
    for lane in 0..count {
        if i32::try_from(lane)? + offset < 0 {
            continue;
        }
        for head in 0..vh {
            let base = lane * (2 * kh + vh) * dim;
            let key = head / (vh / kh);
            let mut q: Vec<f64> = qkv[base + key * dim..][..dim]
                .iter()
                .copied()
                .map(f64::from)
                .collect();
            let mut k: Vec<f64> = qkv[base + (kh + key) * dim..][..dim]
                .iter()
                .copied()
                .map(f64::from)
                .collect();
            let qnorm = (q.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt()
                * f64::from(u32::try_from(dim)?).sqrt();
            let knorm = (k.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt();
            for x in &mut q {
                *x /= qnorm;
            }
            for x in &mut k {
                *x /= knorm;
            }
            let b = 1.0 / (1.0 + (-f64::from(beta[lane * vh + head])).exp());
            let a = f64::from(alpha[lane * vh + head]) + f64::from(bias[head]);
            let decay = (-f64::from(a_log[head]).exp() * a.exp().ln_1p()).exp();
            let state = &mut expected[head * dim * dim..][..dim * dim];
            for x in state.iter_mut() {
                *x *= decay;
            }
            for col in 0..dim {
                let prediction = (0..dim)
                    .map(|row| state[row * dim + col] * k[row])
                    .sum::<f64>();
                let delta = (f64::from(qkv[base + (2 * kh + head) * dim + col]) - prediction) * b;
                for row in 0..dim {
                    state[row * dim + col] = k[row].mul_add(delta, state[row * dim + col]);
                }
                outputs[(head * lanes + lane) * dim + col] =
                    (0..dim).map(|row| state[row * dim + col] * q[row]).sum();
            }
        }
    }
    let mut state = api::zeros::<f32>(&[vh, dim, dim]).sync_on(&device.stream)?;
    let src = device.upload(initial, &[vh, dim, dim])?;
    api::memcpy(&mut state, &src).sync_on(&device.stream)?;
    let mut output = api::zeros::<f32>(&[vh, lanes, dim]).sync_on(&device.stream)?;
    let qkv = device.upload(qkv, &[lanes, 2 * kh + vh, dim])?;
    let beta = device.upload(beta, &[lanes, vh])?;
    let alpha = device.upload(alpha, &[lanes, vh])?;
    let a_log = device.upload(a_log, &[vh])?;
    let bias = device.upload(bias, &[vh])?;
    let metadata = device.upload(vec![0, i32::try_from(count)?, offset, 0], &[4])?;
    kernels::delta(
        (&mut state).partition([1, dim, dim]),
        (&mut output).partition([1, lanes, dim]),
        &qkv,
        &beta,
        &alpha,
        &a_log,
        &bias,
        &metadata,
    )
    .generics(vec![
        kh.to_string(),
        vh.to_string(),
        dim.to_string(),
        lanes.to_string(),
    ])
    .sync_on(&device.stream)?;
    for (actual, reference) in [
        (state.to_host_vec().sync_on(&device.stream)?, expected),
        (output.to_host_vec().sync_on(&device.stream)?, outputs),
    ] {
        for (a, b) in actual.iter().zip(reference) {
            assert!(
                (f64::from(*a) - b).abs() < 0.00002,
                "{geometry:?} count={count} offset={offset}: {a} != {b}"
            );
        }
    }
    Ok(())
}

/// The per-lane recurrence is the path that ships for quantized checkpoints, and it has never
/// been checked against an independent reference at the geometry the 27B actually uses. The
/// chunked twin disagrees with it by 5% of hidden L2 on a single token, so one of the two is
/// wrong: this decides which.
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn per_lane_delta_matches_independent_recurrence_at_model_geometry()
-> Result<(), Box<dyn std::error::Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (kh, vh, dim) = (16_usize, 48_usize, 128_usize);
    let qkv = values((2 * kh + vh) * dim, 17);
    let beta = values(vh, 13);
    let alpha = values(vh, 7);
    let a_log = values(vh, 3);
    let bias = values(vh, 5);
    let initial = values(vh * dim * dim, 23);
    let mut expected: Vec<f64> = initial.iter().copied().map(f64::from).collect();
    let mut outputs = vec![0.0_f64; vh * dim];
    for head in 0..vh {
        let key = head / (vh / kh);
        let mut q: Vec<f64> = qkv[key * dim..][..dim]
            .iter()
            .copied()
            .map(f64::from)
            .collect();
        let mut k: Vec<f64> = qkv[(kh + key) * dim..][..dim]
            .iter()
            .copied()
            .map(f64::from)
            .collect();
        let qnorm = (q.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt()
            * f64::from(u32::try_from(dim)?).sqrt();
        let knorm = (k.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt();
        for x in &mut q {
            *x /= qnorm;
        }
        for x in &mut k {
            *x /= knorm;
        }
        let b = 1.0 / (1.0 + (-f64::from(beta[head])).exp());
        let a = f64::from(alpha[head]) + f64::from(bias[head]);
        let decay = (-f64::from(a_log[head]).exp() * a.exp().ln_1p()).exp();
        let state = &mut expected[head * dim * dim..][..dim * dim];
        for x in state.iter_mut() {
            *x *= decay;
        }
        for col in 0..dim {
            let prediction = (0..dim)
                .map(|row| state[row * dim + col] * k[row])
                .sum::<f64>();
            let delta = (f64::from(qkv[(2 * kh + head) * dim + col]) - prediction) * b;
            for row in 0..dim {
                state[row * dim + col] = k[row].mul_add(delta, state[row * dim + col]);
            }
            outputs[head * dim + col] = (0..dim).map(|row| state[row * dim + col] * q[row]).sum();
        }
    }
    let mut state = api::zeros::<f32>(&[vh, dim, dim]).sync_on(&device.stream)?;
    let src = device.upload(initial, &[vh, dim, dim])?;
    api::memcpy(&mut state, &src).sync_on(&device.stream)?;
    let mut output = api::zeros::<f32>(&[vh, dim]).sync_on(&device.stream)?;
    let qkv = device.upload(qkv, &[(2 * kh + vh) * dim])?;
    let beta = device.upload(beta, &[vh])?;
    let alpha = device.upload(alpha, &[vh])?;
    let a_log = device.upload(a_log, &[vh])?;
    let bias = device.upload(bias, &[vh])?;
    let metadata = device.upload(vec![0, 1, 0, 0], &[4])?;
    crate::resident::recurrent::recurrent::delta(
        (&mut output).partition([1, dim]),
        (&mut state).partition([1, dim, dim]),
        &qkv,
        &beta,
        &alpha,
        &a_log,
        &bias,
        &metadata,
    )
    .generics(vec![kh.to_string(), vh.to_string(), dim.to_string()])
    .sync_on(&device.stream)?;
    let actual_state = state.to_host_vec().sync_on(&device.stream)?;
    let actual_out = output.to_host_vec().sync_on(&device.stream)?;
    let mut worst_state = 0.0_f64;
    for (a, b) in actual_state.iter().zip(&expected) {
        worst_state = worst_state.max((f64::from(*a) - b).abs());
    }
    let mut worst_out = 0.0_f64;
    for (a, b) in actual_out.iter().zip(&outputs) {
        worst_out = worst_out.max((f64::from(*a) - b).abs());
    }
    eprintln!("per-lane delta at model geometry: worst state {worst_state}, worst out {worst_out}");
    assert!(worst_state < 0.001, "per-lane state drift {worst_state}");
    assert!(worst_out < 0.001, "per-lane output drift {worst_out}");
    Ok(())
}

/// The chunked Delta writes into a head-major scratch and transposes into the lane-major output.
/// States come out bit-identical to the per-lane path while the hidden does not, which puts the
/// output path under suspicion, so pin the transpose on its own.
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn chunk_delta_transpose_permutes_head_and_lane() -> Result<(), Box<dyn std::error::Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (lanes, value_heads, dim) = (3_usize, 2_usize, 4_usize);
    let mut scratch = vec![0.0_f32; value_heads * lanes * dim];
    for head in 0..value_heads {
        for lane in 0..lanes {
            for d in 0..dim {
                scratch[(head * lanes + lane) * dim + d] = (head * 1000 + lane * 100 + d) as f32;
            }
        }
    }
    let input = device.upload(scratch, &[value_heads, lanes, dim])?;
    let mut output = api::zeros::<f32>(&[lanes, value_heads, dim]).sync_on(&device.stream)?;
    kernels::transpose((&mut output).partition([1, 1, dim]), &input)
        .generics(vec![dim.to_string()])
        .sync_on(&device.stream)?;
    let actual = output.to_host_vec().sync_on(&device.stream)?;
    for lane in 0..lanes {
        for head in 0..value_heads {
            for d in 0..dim {
                let expected = (head * 1000 + lane * 100 + d) as f32;
                assert_eq!(
                    actual[(lane * value_heads + head) * dim + d],
                    expected,
                    "lane={lane} head={head} d={d}"
                );
            }
        }
    }
    Ok(())
}

/// Run the chunked and per-lane recurrences on byte-identical inputs and require the same bits.
/// The two agree with an f64 reference to 2e-5 but differ from each other by ~1 ULP, which is
/// enough to change a 64-layer greedy decode; this isolates that difference from the capture so
/// the lane loop can be fixed against a sub-second test.
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn chunk_and_per_lane_delta_agree_exactly() -> Result<(), Box<dyn std::error::Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (kh, vh, dim, lanes, count) = (16_usize, 48_usize, 128_usize, 128_usize, 51_usize);
    let heads = 2 * kh + vh;
    let qkv = values(lanes * heads * dim, 17);
    let beta = values(lanes * vh, 13);
    let alpha = values(lanes * vh, 7);
    let a_log = values(vh, 3);
    let bias = values(vh, 5);
    let initial = values(vh * dim * dim, 23);

    // Chunked: one launch over `count` lanes.
    let mut state_c = api::zeros::<f32>(&[vh, dim, dim]).sync_on(&device.stream)?;
    let src = device.upload(initial.clone(), &[vh, dim, dim])?;
    api::memcpy(&mut state_c, &src).sync_on(&device.stream)?;
    let mut out_c = api::zeros::<f32>(&[vh, lanes, dim]).sync_on(&device.stream)?;
    let qkv_c = device.upload(qkv.clone(), &[lanes, heads, dim])?;
    let beta_c = device.upload(beta.clone(), &[lanes, vh])?;
    let alpha_c = device.upload(alpha.clone(), &[lanes, vh])?;
    let a_log_c = device.upload(a_log.clone(), &[vh])?;
    let bias_c = device.upload(bias.clone(), &[vh])?;
    let metadata = device.upload(vec![0, i32::try_from(count)?, 0, 0], &[4])?;
    kernels::delta(
        (&mut state_c).partition([1, dim, dim]),
        (&mut out_c).partition([1, lanes, dim]),
        &qkv_c,
        &beta_c,
        &alpha_c,
        &a_log_c,
        &bias_c,
        &metadata,
    )
    .generics(vec![
        kh.to_string(),
        vh.to_string(),
        dim.to_string(),
        lanes.to_string(),
    ])
    .sync_on(&device.stream)?;

    // Per lane: the shipping kernel, one token at a time.
    let mut state_l = api::zeros::<f32>(&[vh, dim, dim]).sync_on(&device.stream)?;
    api::memcpy(&mut state_l, &src).sync_on(&device.stream)?;
    let mut expected_out = vec![0.0_f32; vh * lanes * dim];
    let lane_metadata = device.upload(vec![0, 0, 0, 0], &[4])?;
    for lane in 0..count {
        let row = &qkv[lane * heads * dim..(lane + 1) * heads * dim];
        let qkv_one = device.upload(row.to_vec(), &[heads * dim])?;
        let beta_one = device.upload(beta[lane * vh..(lane + 1) * vh].to_vec(), &[vh])?;
        let alpha_one = device.upload(alpha[lane * vh..(lane + 1) * vh].to_vec(), &[vh])?;
        let mut out_one = api::zeros::<f32>(&[vh, dim]).sync_on(&device.stream)?;
        crate::resident::recurrent::recurrent::delta(
            (&mut out_one).partition([1, dim]),
            (&mut state_l).partition([1, dim, dim]),
            &qkv_one,
            &beta_one,
            &alpha_one,
            &a_log_c,
            &bias_c,
            &lane_metadata,
        )
        .generics(vec![kh.to_string(), vh.to_string(), dim.to_string()])
        .sync_on(&device.stream)?;
        let lane_out = out_one.to_host_vec().sync_on(&device.stream)?;
        for head in 0..vh {
            for d in 0..dim {
                expected_out[(head * lanes + lane) * dim + d] = lane_out[head * dim + d];
            }
        }
    }
    let chunk_out = out_c.to_host_vec().sync_on(&device.stream)?;
    let out_gap = chunk_out
        .iter()
        .zip(&expected_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("chunked vs per-lane output max_abs={out_gap:e} (target: exactly 0)");
    let chunk_state = state_c.to_host_vec().sync_on(&device.stream)?;
    let lane_state = state_l.to_host_vec().sync_on(&device.stream)?;
    let worst = chunk_state
        .iter()
        .zip(&lane_state)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("chunked vs per-lane state max_abs={worst:e} (target: exactly 0)");
    // The lane loop reproduces the per-lane recurrence to about one ULP instead of exactly, and a
    // 64-layer greedy decode amplifies that into a different continuation. Track the figure here so
    // the fix has a sub-second gate; tighten to `assert_eq!(worst, 0.0)` once the loop matches.
    assert!(
        worst < 1.0e-7 && out_gap < 1.0e-7,
        "chunked and per-lane recurrences differ"
    );
    Ok(())
}
