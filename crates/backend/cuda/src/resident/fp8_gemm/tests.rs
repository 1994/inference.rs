use super::kernels;
use crate::device::CudaDevice;
use cuda_core::f8e4m3fn;
use cutile::prelude::*;

fn decode(code: u8) -> f32 {
    let magnitude = code & 127;
    let exponent = magnitude >> 3;
    let mantissa = f32::from(magnitude & 7);
    let value = if exponent == 0 {
        mantissa / 512.0
    } else {
        (1.0 + mantissa / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    };
    if code & 128 == 0 { value } else { -value }
}
fn quantize(value: f32) -> u8 {
    let value_abs = value.abs().min(448.0);
    let magnitude = (0..=126u8)
        .min_by(|&a, &b| {
            (decode(a) - value_abs)
                .abs()
                .total_cmp(&(decode(b) - value_abs).abs())
                .then((a % 2).cmp(&(b % 2)))
        })
        .unwrap();
    magnitude | if value.is_sign_negative() { 128 } else { 0 }
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn fp8_token_mma_matches_independent_quantization() -> Result<(), Box<dyn std::error::Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (m, n, k) in [
        (1, 65, 144),
        (3, 64, 256),
        (12, 129, 512),
        (32, 64, 1024),
        (1, 65, 17408),
        (3, 64, 5120),
    ] {
        let input: Vec<f32> = (0..m * k)
            .map(|i| {
                if m > 1 && i < k {
                    0.0
                } else {
                    (f32::from(u16::try_from(i * 17 % 131).unwrap()) - 65.0) / 37.0
                }
            })
            .collect();
        let codes: Vec<u8> = (0..n * k)
            .map(|i| 48 + u8::try_from(i % 16).unwrap() + if i % 3 == 0 { 128 } else { 0 })
            .collect();
        let scales: Vec<f32> = (0..n)
            .map(|i| f32::from(u16::try_from(i % 13 + 1).unwrap()) / 11.0)
            .collect();
        let x = device.upload(input.clone(), &[m, k])?;
        let w = device.upload(codes.iter().copied().map(f8e4m3fn).collect(), &[n, k])?;
        let ws = device.upload(scales.clone(), &[n])?;
        let mut q = api::zeros::<f8e4m3fn>(&[m, k]).sync_on(&device.stream)?;
        let mut qs = api::zeros::<f32>(&[m, 1]).sync_on(&device.stream)?;
        kernels::quantize(
            (&mut q).partition([1, k.next_power_of_two()]),
            (&mut qs).partition([1, 1]),
            &x,
        )
        .generics(vec![k.to_string(), k.next_power_of_two().to_string()])
        .sync_on(&device.stream)?;
        let output = kernels::matmul(
            api::zeros::<f32>(&[m, n]).partition([16, 64]),
            &q,
            &w,
            &qs,
            &ws,
        )
        .generics(vec![k.to_string()])
        .first()
        .unpartition()
        .sync_on(&device.stream)?;
        let actual = output.to_host_vec().sync_on(&device.stream)?;
        let encoded = q.to_host_vec().sync_on(&device.stream)?;
        for row in 0..m {
            let scale = (input[row * k..(row + 1) * k]
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0, f32::max)
                / 448.0)
                .max(1e-12);
            let quantized: Vec<_> = input[row * k..(row + 1) * k]
                .iter()
                .map(|x| quantize(*x / scale))
                .collect();
            for col in 0..k {
                assert_eq!(
                    encoded[row * k + col].0,
                    quantized[col],
                    "row={row} k={col}"
                );
            }
            for col in 0..n {
                let expected = (0..k)
                    .map(|i| {
                        f64::from(decode(quantized[i])) * f64::from(decode(codes[col * k + i]))
                    })
                    .sum::<f64>()
                    * f64::from(scale)
                    * f64::from(scales[col]);
                assert!(
                    (f64::from(actual[row * n + col]) - expected).abs()
                        < expected.abs().mul_add(0.00002, 0.0002),
                    "{m}x{n}x{k} [{row},{col}] {} != {expected}",
                    actual[row * n + col]
                );
            }
        }
    }
    Ok(())
}
