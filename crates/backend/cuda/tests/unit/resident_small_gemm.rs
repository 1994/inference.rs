use super::kernels;
use crate::device::CudaDevice;
use cutile::{half::bf16, prelude::*};

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn small_mma_matches_f64_oracle() -> Result<(), Box<dyn std::error::Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (m, n, k) in [(2, 65, 144), (4, 32, 256), (12, 129, 512), (17, 33, 128)] {
        let input: Vec<f32> = (0..m * k)
            .map(|i| (f32::from(u16::try_from(i * 17 % 131).unwrap()) - 65.0) / 37.0)
            .collect();
        let weights: Vec<bf16> = (0..n * k)
            .map(|i| bf16::from_f32((f32::from(u16::try_from(i * 7 % 127).unwrap()) - 63.0) / 41.0))
            .collect();
        let x = device.upload(input.clone(), &[m, k])?;
        let w = device.upload(weights.clone(), &[n, k])?;
        let output = kernels::dense(api::zeros::<f32>(&[m, n]).partition([16, 32]), &x, &w)
            .generics(vec![bf16::DTYPE.as_str().into(), k.to_string()])
            .first()
            .unpartition()
            .sync_on(&device.stream)?;
        let actual = output.to_host_vec().sync_on(&device.stream)?;
        for row in 0..m {
            for col in 0..n {
                let expected: f64 = (0..k)
                    .map(|i| {
                        f64::from(input[row * k + i]) * f64::from(weights[col * k + i].to_f32())
                    })
                    .sum();
                let error = (f64::from(actual[row * n + col]) - expected).abs();
                assert!(
                    error < expected.abs().mul_add(0.00002, 0.0005),
                    "{m}x{n}x{k} [{row},{col}] {} != {expected}",
                    actual[row * n + col]
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn small_mma_performance() -> Result<(), Box<dyn std::error::Error>> {
    use cutile::bench::{BenchOptions, do_bench_paired};
    use std::time::Duration;
    if cfg!(debug_assertions) {
        return Err("performance requires release".into());
    }
    let compile = cutile::cutile_compiler::hints::CompileOptions::default();
    if compile.device_debug || compile.opt_level.is_some_and(|x| x != 3) {
        return Err("performance requires CUDA O3".into());
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let options = BenchOptions {
        warmup: Duration::from_millis(50),
        rep: Duration::from_millis(100),
        min_reps: 10,
        max_reps: 100,
        clear_l2: true,
    };
    for (n, k) in [(4096, 4096), (14336, 4096), (4096, 14336), (151_936, 2048)] {
        let weights = device.upload(vec![bf16::from_f32(0.25); n * k], &[n, k])?;
        for m in [2, 4, 12] {
            let input = device.upload(vec![0.253_f32; m * k], &[m, k])?;
            let mut output = api::zeros::<f32>(&[m, n]).sync_on(&device.stream)?;
            let old = CudaGraph::scope(&device.stream, |scope| {
                scope.record(
                    super::super::linear_batch::batched::dense(
                        (&mut output).partition([4, 16]),
                        &input,
                        &weights,
                    )
                    .generics(vec![
                        bf16::DTYPE.as_str().into(),
                        "16".into(),
                        "256".into(),
                        k.to_string(),
                    ]),
                )?;
                Ok(())
            })?;
            let new = CudaGraph::scope(&device.stream, |scope| {
                scope.record(
                    kernels::dense((&mut output).partition([16, 32]), &input, &weights)
                        .generics(vec![bf16::DTYPE.as_str().into(), k.to_string()]),
                )?;
                Ok(())
            })?;
            let (baseline, candidate) = do_bench_paired(
                &device.stream,
                &options,
                |_| {
                    old.launch()
                        .sync_on(&device.stream)
                        .map_err(|e| cutile::error::tensor_error(&e.to_string()))
                },
                |_| {
                    new.launch()
                        .sync_on(&device.stream)
                        .map_err(|e| cutile::error::tensor_error(&e.to_string()))
                },
            )?;
            println!(
                "{m}x{n}x{k} old_ms={} new_ms={} speedup={}",
                baseline.median_ms(),
                candidate.median_ms(),
                baseline.median_ms() / candidate.median_ms()
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn small_scaled_mma_matches_f64_oracle() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_core::f8e4m3fn;
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (m, n, k) in [(2, 65, 144), (12, 129, 256)] {
        let input: Vec<f32> = (0..m * k)
            .map(|i| (f32::from(u16::try_from(i * 17 % 131).unwrap()) - 65.0) / 37.0)
            .collect();
        let codes: Vec<u8> = (0..n * k)
            .map(|i| 48 + u8::try_from(i % 16).unwrap() + if i % 3 == 0 { 128 } else { 0 })
            .collect();
        let decoded: Vec<f32> = codes
            .iter()
            .map(|c| {
                let magnitude = c & 127;
                let sign = if c & 128 == 0 { 1.0 } else { -1.0 };
                sign * (1.0 + f32::from(magnitude & 7) / 8.0)
                    * 2.0_f32.powi(i32::from(magnitude >> 3) - 7)
            })
            .collect();
        let scales: Vec<f32> = (0..n)
            .map(|i| f32::from(u16::try_from(i % 13 + 1).unwrap()) / 11.0)
            .collect();
        let x = device.upload(input.clone(), &[m, k])?;
        let w = device.upload(codes.into_iter().map(f8e4m3fn).collect(), &[n, k])?;
        let scale = device.upload(scales.clone(), &[n])?;
        let output = kernels::scaled(
            api::zeros::<f32>(&[m, n]).partition([16, 32]),
            &x,
            &w,
            &scale,
        )
        .generics(vec![f8e4m3fn::DTYPE.as_str().into(), k.to_string()])
        .first()
        .unpartition()
        .sync_on(&device.stream)?;
        let actual = output.to_host_vec().sync_on(&device.stream)?;
        for row in 0..m {
            for col in 0..n {
                let expected: f64 = (0..k)
                    .map(|i| f64::from(input[row * k + i]) * f64::from(decoded[col * k + i]))
                    .sum::<f64>()
                    * f64::from(scales[col]);
                assert!(
                    (f64::from(actual[row * n + col]) - expected).abs()
                        < expected.abs().mul_add(0.00002, 0.0005)
                );
            }
        }
    }
    Ok(())
}
