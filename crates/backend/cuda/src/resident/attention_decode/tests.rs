use super::kernels;
use crate::device::CudaDevice;
use cuda_core::f8e4m3fn;
use cutile::prelude::*;

fn decoded(code: u8) -> f32 {
    let magnitude = code & 127;
    let x = (1.0 + f32::from(magnitude & 7) / 8.0) * 2.0_f32.powi(i32::from(magnitude >> 3) - 7);
    if code & 128 == 0 { x } else { -x }
}
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn split_kv_matches_independent_causal_window_reference() -> Result<(), Box<dyn std::error::Error>>
{
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (heads, kv_heads, dim) in [(4, 2, 32), (8, 8, 64), (16, 8, 128), (24, 4, 256)] {
        check_geometry(&device, heads, kv_heads, dim)?;
    }
    Ok(())
}

fn check_geometry(
    device: &CudaDevice,
    heads: usize,
    kv_heads: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let capacity = 1024;
    let scale = f64::from(u32::try_from(dim)?).sqrt();
    let query: Vec<f32> = (0..heads * dim)
        .map(|i| (f32::from(u16::try_from(i % 15).unwrap()) - 7.0) / 8.0)
        .collect();
    let codes: Vec<u8> = (0..kv_heads * capacity * dim)
        .map(|i| 32 + u8::try_from(i % 31).unwrap() + if i % 3 == 0 { 128 } else { 0 })
        .collect();
    let value_codes: Vec<u8> = codes.iter().copied().rev().collect();
    let keys: Vec<f32> = codes.iter().copied().map(decoded).collect();
    let values: Vec<f32> = value_codes.iter().copied().map(decoded).collect();
    let q = device.upload(query.clone(), &[heads, dim])?;
    let k = device.upload(keys.clone(), &[kv_heads, capacity, dim])?;
    let v = device.upload(values.clone(), &[kv_heads, capacity, dim])?;
    let k8 = device.upload(
        codes.into_iter().map(f8e4m3fn).collect(),
        &[kv_heads, capacity, dim],
    )?;
    let v8 = device.upload(
        value_codes.into_iter().map(f8e4m3fn).collect(),
        &[kv_heads, capacity, dim],
    )?;
    for (position, window) in [
        (-1, 0),
        (0, 0),
        (31, 0),
        (32, 17),
        (511, 0),
        (1023, 0),
        (1023, 17),
    ] {
        let mut expected = vec![0.0_f64; heads * dim];
        if position >= 0 {
            let end = usize::try_from(position + 1)?;
            let start = if window == 0 {
                0
            } else {
                end.saturating_sub(usize::try_from(window)?)
            };
            for head in 0..heads {
                let offset = head / (heads / kv_heads) * capacity * dim;
                let mut scores: Vec<f64> = (start..end)
                    .map(|row| {
                        (0..dim)
                            .map(|d| {
                                f64::from(query[head * dim + d])
                                    * f64::from(keys[offset + row * dim + d])
                                    * 0.5
                            })
                            .sum::<f64>()
                            / scale
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for score in &mut scores {
                    *score = (*score - maximum).exp();
                }
                let total: f64 = scores.iter().sum();
                for d in 0..dim {
                    expected[head * dim + d] = scores
                        .iter()
                        .enumerate()
                        .map(|(i, s)| s * f64::from(values[offset + (start + i) * dim + d]) * 2.0)
                        .sum::<f64>()
                        / total;
                }
            }
        }
        for parts in [1, 2, 8, 16] {
            for actual in [
                run(device, &q, &k, &v, position, window, parts)?,
                run(device, &q, &k8, &v8, position, window, parts)?,
            ] {
                for (a, b) in actual.iter().zip(&expected) {
                    assert!(
                        (f64::from(*a) - b).abs() < 2e-5,
                        "position={position} window={window} parts={parts}: {a} != {b}"
                    );
                }
            }
        }
    }
    Ok(())
}
fn run<E: DType>(
    device: &CudaDevice,
    q: &Tensor<f32>,
    k: &Tensor<E>,
    v: &Tensor<E>,
    position: i32,
    window: i32,
    parts: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let heads = usize::try_from(q.shape()[0])?;
    let dim = usize::try_from(q.shape()[1])?;
    let kv_heads = usize::try_from(k.shape()[0])?;
    let meta = device.upload(vec![position, 0, position, 0], &[4])?;
    let mut numerator = api::zeros::<f32>(&[heads * parts, dim]).sync_on(&device.stream)?;
    let mut maxima = api::zeros::<f32>(&[heads * parts, 1]).sync_on(&device.stream)?;
    let mut sums = api::zeros::<f32>(&[heads * parts, 1]).sync_on(&device.stream)?;
    kernels::partial(
        (&mut numerator).partition([1, dim]),
        (&mut maxima).partition([1, 1]),
        (&mut sums).partition([1, 1]),
        q,
        k,
        v,
        &meta,
        window,
        0.5,
        2.0,
    )
    .generics(vec![
        E::DTYPE.as_str().into(),
        dim.to_string(),
        (heads / kv_heads).to_string(),
        parts.to_string(),
    ])
    .sync_on(&device.stream)?;
    let output = kernels::merge(
        api::zeros::<f32>(&[heads, dim]).partition([1, dim]),
        &numerator.view(&[heads, parts, dim])?,
        &maxima.view(&[heads, parts])?,
        &sums.view(&[heads, parts])?,
    )
    .generics(vec![dim.to_string(), parts.to_string()])
    .first()
    .unpartition()
    .sync_on(&device.stream)?;
    Ok(output.to_host_vec().sync_on(&device.stream)?)
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn split_kv_performance() -> Result<(), Box<dyn std::error::Error>> {
    use cutile::bench::BenchOptions;
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
    for (heads, kv_heads, dim) in [(16, 8, 128), (32, 4, 128), (8, 2, 64)] {
        bench_case(&device, &options, heads, kv_heads, dim)?;
    }
    Ok(())
}

fn bench_case(
    device: &CudaDevice,
    options: &cutile::bench::BenchOptions,
    heads: usize,
    kv_heads: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use cutile::bench::do_bench_paired;
    for capacity in [128, 512, 4096, 16384] {
        let q = device.upload(vec![0.125_f32; heads * dim], &[heads, dim])?;
        let k = device.upload(
            vec![0.25_f32; kv_heads * capacity * dim],
            &[kv_heads, capacity, dim],
        )?;
        let v = device.upload(
            vec![0.375_f32; kv_heads * capacity * dim],
            &[kv_heads, capacity, dim],
        )?;
        let position = i32::try_from(capacity - 1)?;
        let meta = device.upload(vec![position, 0, position, 0], &[4])?;
        let mut output = api::zeros::<f32>(&[heads, dim]).sync_on(&device.stream)?;
        let baseline = CudaGraph::scope(&device.stream, |scope| {
            scope.record(
                super::super::attention::attention::decode(
                    (&mut output).partition([1, dim]),
                    &q,
                    &k,
                    &v,
                    &meta,
                    0,
                    1.0,
                    1.0,
                )
                .generics(vec![
                    f32::DTYPE.as_str().into(),
                    dim.to_string(),
                    (heads / kv_heads).to_string(),
                ]),
            )?;
            Ok(())
        })?;
        for parts in [2, 8, 16] {
            let mut numerator = api::zeros::<f32>(&[heads * parts, dim]).sync_on(&device.stream)?;
            let mut maxima = api::zeros::<f32>(&[heads * parts, 1]).sync_on(&device.stream)?;
            let mut sums = api::zeros::<f32>(&[heads * parts, 1]).sync_on(&device.stream)?;
            let candidate = CudaGraph::scope(&device.stream, |scope| {
                scope.record(
                    kernels::partial(
                        (&mut numerator).partition([1, dim]),
                        (&mut maxima).partition([1, 1]),
                        (&mut sums).partition([1, 1]),
                        &q,
                        &k,
                        &v,
                        &meta,
                        0,
                        1.0,
                        1.0,
                    )
                    .generics(vec![
                        f32::DTYPE.as_str().into(),
                        dim.to_string(),
                        (heads / kv_heads).to_string(),
                        parts.to_string(),
                    ]),
                )?;
                scope.record(
                    kernels::merge(
                        (&mut output).partition([1, dim]),
                        &numerator.view(&[heads, parts, dim])?,
                        &maxima.view(&[heads, parts])?,
                        &sums.view(&[heads, parts])?,
                    )
                    .generics(vec![dim.to_string(), parts.to_string()]),
                )?;
                Ok(())
            })?;
            let (old, new) = do_bench_paired(
                &device.stream,
                options,
                |_| {
                    baseline
                        .launch()
                        .sync_on(&device.stream)
                        .map_err(|e| cutile::error::tensor_error(&e.to_string()))
                },
                |_| {
                    candidate
                        .launch()
                        .sync_on(&device.stream)
                        .map_err(|e| cutile::error::tensor_error(&e.to_string()))
                },
            )?;
            println!(
                "heads={heads} kv_heads={kv_heads} dim={dim} tokens={capacity} parts={parts} old_ms={} new_ms={} speedup={}",
                old.median_ms(),
                new.median_ms(),
                old.median_ms() / new.median_ms()
            );
        }
    }
    Ok(())
}
