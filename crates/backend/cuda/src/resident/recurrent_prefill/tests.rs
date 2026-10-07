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
    for (kh, vh, dim, lanes) in [(2, 4, 32, 32), (2, 6, 128, 32), (2, 6, 128, 128)] {
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
