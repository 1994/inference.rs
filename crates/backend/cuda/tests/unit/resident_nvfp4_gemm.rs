use super::kernels;
use crate::device::CudaDevice;
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::prelude::*;

/// Tile these tests construct their output partition with; the GEMM's `M` and `N` generics and
/// the partition shape must agree. They stay on the narrow tile the correctness cases were
/// written against; the shipped tile is exercised end to end by the serving benchmark.
const QUANT_TILE_ROWS: usize = 16;
const QUANT_TILE_COLUMNS: usize = 64;

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
        .generics(vec![
            k.to_string(),
            QUANT_TILE_ROWS.to_string(),
            QUANT_TILE_COLUMNS.to_string(),
        ])
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
                    .generics(vec![
                        k.to_string(),
                        QUANT_TILE_ROWS.to_string(),
                        QUANT_TILE_COLUMNS.to_string(),
                    ]),
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

/// Rows every sweep variant is benchmarked against, so all tiles read one allocation size.
const SWEEP_PAD_ROWS: usize = 64;
/// Production projection shapes the sweep covers: the 27B MLP up/gate and down projections at
/// single-token, verification and prompt widths.
const SWEEP_SHAPES: [(usize, usize, usize); 8] = [
    (1, 17408, 5120),
    (1, 5120, 5120),
    (1, 5120, 17408),
    (4, 17408, 5120),
    (4, 5120, 17408),
    (12, 17408, 5120),
    (12, 5120, 17408),
    (64, 17408, 5120),
];
/// Output tiles the sweep compares against the shipped `[16, 64]`.
const SWEEP_TILES: [(usize, usize); 4] = [(16, 64), (16, 128), (64, 128), (64, 64)];

/// Tile sweep for the production W4A4 kernel: does a wider output column tile or a taller row
/// tile lift the weight bandwidth the prompt and slot-verify replays are bound by?
///
/// Measured per shape against the shipped `[16, 64]` tile with paired alternation, so a
/// throttled host cannot flip the ranking. Informational: it prints, it does not assert.
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn nvfp4_packed_tile_sweep() -> Result<(), Box<dyn std::error::Error>> {
    use cutile::bench::BenchOptions;
    use std::time::Duration;
    if cfg!(debug_assertions) {
        return Err("performance measurements require cargo test --release".into());
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
    for (m, n, k) in SWEEP_SHAPES {
        sweep_shape(&device, &options, (m, n, k))?;
    }
    Ok(())
}

/// One shape's tile comparison, alternating baseline and candidate under `do_bench_paired`.
fn sweep_shape(
    device: &CudaDevice,
    options: &cutile::bench::BenchOptions,
    (m, n, k): (usize, usize, usize),
) -> Result<(), Box<dyn std::error::Error>> {
    use cutile::bench::do_bench_paired;
    assert!(m <= SWEEP_PAD_ROWS);
    let input = device.upload(
        vec![f4e2m1fnx2::from_bits(0x21); SWEEP_PAD_ROWS * k / 2],
        &[SWEEP_PAD_ROWS, k / 2],
    )?;
    let input_scales = device.upload(
        vec![f8e4m3fn(48); SWEEP_PAD_ROWS * k / 16],
        &[SWEEP_PAD_ROWS, k / 16],
    )?;
    let weight = device.upload(vec![f4e2m1fnx2::from_bits(0x21); n * k / 2], &[n, k / 2])?;
    let scales = device.upload(vec![f8e4m3fn(48); n * k / 16], &[n, k / 16])?;
    let mut output = api::zeros::<f32>(&[SWEEP_PAD_ROWS, n]).sync_on(&device.stream)?;
    let mut tile_graph = |rows: usize, columns: usize| -> Result<CudaGraph<()>, Error> {
        Ok(CudaGraph::scope(&device.stream, |scope| {
            scope.record(
                kernels::packed(
                    (&mut output).partition([rows, columns]),
                    &input,
                    &input_scales,
                    &weight,
                    &scales,
                    1.0,
                )
                .generics(vec![
                    k.to_string(),
                    rows.to_string(),
                    columns.to_string(),
                ]),
            )?;
            Ok(())
        })?)
    };
    let baseline = tile_graph(QUANT_TILE_ROWS, QUANT_TILE_COLUMNS)?;
    for (rows, columns) in SWEEP_TILES {
        let candidate = tile_graph(rows, columns)?;
        let (base, cand) = do_bench_paired(
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
            "m{m} n{n} k{k} tile[{rows},{columns}]: base_ms={} cand_ms={} ratio={}",
            base.median_ms(),
            cand.median_ms(),
            cand.median_ms() / base.median_ms()
        );
    }
    Ok(())
}
