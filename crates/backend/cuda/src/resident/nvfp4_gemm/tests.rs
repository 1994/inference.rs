use super::kernels;
use crate::device::CudaDevice;
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::prelude::*;

fn fp8(code: u8) -> f32 {
    let exponent = code >> 3;
    let mantissa = f32::from(code & 7);
    if exponent == 0 {
        mantissa / 512.0
    } else {
        (1.0 + mantissa / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    }
}
fn nearest(value: f32, table: &[f32]) -> usize {
    (0..table.len())
        .min_by(|&a, &b| {
            (table[a] - value)
                .abs()
                .total_cmp(&(table[b] - value).abs())
                .then((a % 2).cmp(&(b % 2)))
        })
        .expect("nonempty codebook")
}

#[test]
#[ignore = "requires a CUDA device; run inside safe-run"]
fn nvfp4_mma_matches_independent_block_quantization() -> Result<(), Box<dyn std::error::Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let f4 = [0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let f8: Vec<_> = (0..=126).map(fp8).collect();
    for (m, n, k, global) in [
        (1, 65, 144, 1.0_f32),
        (3, 64, 256, 16.0),
        (17, 129, 128, 0.25),
        (32, 64, 512, 1024.0),
    ] {
        let input: Vec<_> = (0..m * k)
            .map(|i| {
                if i < 16 {
                    0.0
                } else {
                    (f32::from(u8::try_from(i * 17 % 131).expect("bounded value")) - 65.0) / 37.0
                }
            })
            .collect();
        let codes: Vec<u8> = (0..n * k)
            .map(|i| u8::try_from((i * 7 + i / 9) % 16).expect("FP4 code"))
            .collect();
        let packed: Vec<_> = codes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f4e2m1fnx2::from_bits(c[0] | c[1] << 4))
            .collect();
        let scale_codes: Vec<u8> = (0..n * k / 16)
            .map(|i| u8::try_from(48 + i % 16).expect("FP8 code"))
            .collect();
        let scales: Vec<_> = scale_codes.iter().copied().map(f8e4m3fn).collect();
        let x = device.upload(input.clone(), &[m, k])?;
        let w = device.upload(packed, &[n, k / 2])?;
        let s = device.upload(scales, &[n, k / 16])?;
        let mut quantized_input = api::zeros::<f4e2m1fnx2>(&[m, k / 2]).sync_on(&device.stream)?;
        let mut input_scales = api::zeros::<f8e4m3fn>(&[m, k / 16]).sync_on(&device.stream)?;
        kernels::quantize(
            (&mut quantized_input).partition([1, 256]),
            (&mut input_scales).partition([1, 32]),
            &x,
            global,
        )
        .generics(vec![k.to_string()])
        .sync_on(&device.stream)?;
        let output = kernels::packed(
            api::zeros::<f32>(&[m, n]).partition([16, 64]),
            &quantized_input,
            &input_scales,
            &w,
            &s,
            1.0 / global,
        )
        .generics(vec![k.to_string()])
        .first()
        .unpartition()
        .sync_on(&device.stream)?;
        let output = output.to_host_vec().sync_on(&device.stream)?;
        let mut quantized = input.clone();
        for values in quantized.as_chunks_mut::<16>().0 {
            let maximum = values.iter().copied().map(f32::abs).fold(0.0, f32::max);
            let scale = f8[nearest((maximum * global / 6.0).min(448.0), &f8)];
            for value in values {
                *value = if scale == 0.0 {
                    0.0
                } else {
                    value.signum()
                        * f4[nearest((value.abs() * global / scale).min(6.0), &f4)]
                        * scale
                        / global
                };
            }
        }
        for row in 0..m {
            for col in 0..n {
                let expected: f64 = (0..k)
                    .map(|i| {
                        let code = codes[col * k + i];
                        let weight =
                            f4[usize::from(code & 7)] * if code & 8 == 0 { 1.0 } else { -1.0 };
                        f64::from(quantized[row * k + i])
                            * f64::from(weight * fp8(scale_codes[col * k / 16 + i / 16]))
                    })
                    .sum();
                let actual = f64::from(output[row * n + col]);
                assert!(
                    (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                    "{m}x{n}x{k} [{row},{col}]: {actual} != {expected}"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device; run inside safe-run"]
fn nvfp4_quantization_split_performance() -> Result<(), Box<dyn std::error::Error>> {
    use cutile::bench::{BenchOptions, do_bench_paired};
    use std::time::Duration;
    if cfg!(debug_assertions) {
        return Err("performance measurements require cargo test --release".into());
    }
    let compile = cutile::cutile_compiler::hints::CompileOptions::default();
    if compile.device_debug || compile.opt_level.is_some_and(|level| level != 3) {
        return Err("performance measurements require CUDA JIT O3 without device debug".into());
    }
    println!("build_profile=release cuda_jit_opt_level=3 device_debug=false");
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let options = BenchOptions {
        warmup: Duration::from_millis(50),
        rep: Duration::from_millis(100),
        min_reps: 10,
        max_reps: 100,
        clear_l2: true,
    };
    for (n, k) in [(14336, 4096), (4096, 14336)] {
        let weights = device.upload(vec![f4e2m1fnx2::from_bits(0x21); n * k / 2], &[n, k / 2])?;
        let scales = device.upload(vec![f8e4m3fn(56); n * k / 16], &[n, k / 16])?;
        for m in [1, 3, 32] {
            let input = device.upload(vec![0.25_f32; m * k], &[m, k])?;
            let mut quantized = api::zeros::<f4e2m1fnx2>(&[m, k / 2]).sync_on(&device.stream)?;
            let mut input_scales = api::zeros::<f8e4m3fn>(&[m, k / 16]).sync_on(&device.stream)?;
            let mut output = api::zeros::<f32>(&[m, n]).sync_on(&device.stream)?;
            let fused = CudaGraph::scope(&device.stream, |scope| {
                scope.record(
                    kernels::matmul(
                        (&mut output).partition([16, 64]),
                        &input,
                        &weights,
                        &scales,
                        1.0,
                        1.0,
                    )
                    .generics(vec![k.to_string()]),
                )?;
                Ok(())
            })?;
            let separated = CudaGraph::scope(&device.stream, |scope| {
                scope.record(
                    kernels::quantize(
                        (&mut quantized).partition([1, 256]),
                        (&mut input_scales).partition([1, 32]),
                        &input,
                        1.0,
                    )
                    .generics(vec![k.to_string()]),
                )?;
                scope.record(
                    kernels::packed(
                        (&mut output).partition([16, 64]),
                        &quantized,
                        &input_scales,
                        &weights,
                        &scales,
                        1.0,
                    )
                    .generics(vec![k.to_string()]),
                )?;
                Ok(())
            })?;
            let (baseline, candidate) = do_bench_paired(
                &device.stream,
                &options,
                |_| {
                    fused
                        .launch()
                        .sync_on(&device.stream)
                        .map_err(|e| cutile::error::tensor_error(&e.to_string()))
                },
                |_| {
                    separated
                        .launch()
                        .sync_on(&device.stream)
                        .map_err(|e| cutile::error::tensor_error(&e.to_string()))
                },
            )?;
            println!(
                "{m}x{n}x{k}: fused_ms={} separated_ms={} speedup={}",
                baseline.median_ms(),
                candidate.median_ms(),
                baseline.median_ms() / candidate.median_ms()
            );
        }
    }
    Ok(())
}
