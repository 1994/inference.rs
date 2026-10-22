//! Exact scale/residual candidate, tested before enabling graph rewrites.
use super::{error, kernels as baseline};
use crate::{device::CudaDevice, resident::kernels::aux};
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    error::tensor_error,
    prelude::*,
};
use std::{error::Error, time::Duration};
type TestResult = Result<(), Box<dyn Error>>;

#[test]
#[ignore = "requires CUDA hardware; run release serially under safe-run"]
fn scale_residual_bits_and_cost() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (rows, columns) in [
        (1, 1),
        (3, 255),
        (3, 256),
        (3, 257),
        (64, 5120),
        (1024, 5120),
    ] {
        let mut raw = api::zeros::<f32>(&[rows, columns]).sync_on(&device.stream)?;
        let mut scaled = api::zeros::<f32>(&[rows, columns]).sync_on(&device.stream)?;
        let mut residual = api::zeros::<f32>(&[rows, columns]).sync_on(&device.stream)?;
        let mut xs = api::zeros::<f32>(&[rows, 1]).sync_on(&device.stream)?;
        let mut ws = api::zeros::<f32>(&[columns]).sync_on(&device.stream)?;
        let mut expected = api::zeros::<f32>(&[rows * columns]).sync_on(&device.stream)?;
        let mut actual = api::zeros::<f32>(&[rows, columns]).sync_on(&device.stream)?;
        for side in [0, 1] {
            let old = CudaGraph::scope(&device.stream, |scope| {
                scope.record(baseline::scale((&mut scaled).partition([1, 256]), &xs, &ws))?;
                let flat = scaled
                    .view(&[rows * columns])
                    .map_err(|e| error(e.to_string()))?;
                let other = residual
                    .view(&[rows * columns])
                    .map_err(|e| error(e.to_string()))?;
                let (a, b) = if side == 0 {
                    (&flat, &other)
                } else {
                    (&other, &flat)
                };
                scope.record(
                    aux::binary((&mut expected).partition([256]), a, b).generics(vec!["0".into()]),
                )?;
                Ok(())
            })?;
            let new = CudaGraph::scope(&device.stream, |scope| {
                scope.record(
                    baseline::scale_add(
                        (&mut actual).partition([1, 256]),
                        &raw,
                        &residual,
                        &xs,
                        &ws,
                    )
                    .generics(vec![side.to_string()]),
                )?;
                Ok(())
            })?;
            // Mutate every operand between graph replays, including signed zeros, subnormals,
            // NaN payloads, overflow and padding. Compare all bits, including NaNs.
            for fixture in 0..5 {
                let [values, other, row, column] = fixture_values(rows, columns, fixture);
                api::memcpy(&mut raw, &*device.upload(values, &[rows, columns])?)
                    .sync_on(&device.stream)?;
                api::memcpy(&mut scaled, &raw).sync_on(&device.stream)?;
                api::memcpy(&mut residual, &*device.upload(other, &[rows, columns])?)
                    .sync_on(&device.stream)?;
                api::memcpy(&mut xs, &*device.upload(row, &[rows, 1])?).sync_on(&device.stream)?;
                api::memcpy(&mut ws, &*device.upload(column, &[columns])?)
                    .sync_on(&device.stream)?;
                old.launch().sync_on(&device.stream)?;
                new.launch().sync_on(&device.stream)?;
                let a = device.read_borrowed(&expected)?;
                let b = device.read_borrowed(&actual)?;
                for (index, (a, b)) in a.iter().zip(&b).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "shape={rows}x{columns} side={side} fixture={fixture} index={index}"
                    );
                }
                if fixture == 0 && rows == 1024 {
                    measure(&device, &old, &new, side)?;
                }
            }
            println!("bits passed {rows}x{columns} side={side}");
        }
    }
    Ok(())
}

fn fixture_values(rows: usize, columns: usize, fixture: usize) -> [Vec<f32>; 4] {
    let patterns = [
        0_u32,
        0x8000_0000,
        1,
        0x8000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0001,
        0xffc0_0002,
        0x7f80_0001,
    ];
    let values: Vec<f32> = (0..rows * columns)
        .map(|i| {
            if fixture == 0 || fixture == 4 {
                f32::from(u16::try_from(i % 997).unwrap()) / 997.0 - 0.5
            } else {
                f32::from_bits(patterns[(i + fixture) % patterns.len()])
            }
        })
        .collect();
    let other: Vec<f32> = (0..rows * columns)
        .map(|i| {
            if fixture == 0 || fixture == 4 {
                f32::from(u16::try_from(i % 127).unwrap()) / 127.0
            } else {
                f32::from_bits(patterns[(i / patterns.len() + fixture) % patterns.len()])
            }
        })
        .collect();
    let row: Vec<f32> = (0..rows)
        .map(|i| {
            if fixture == 0 || fixture == 4 {
                if fixture == 0 { 1.0 } else { 0.173_205_08 }
            } else {
                f32::from_bits(patterns[(i + fixture * 3) % patterns.len()])
            }
        })
        .collect();
    let column: Vec<f32> = (0..columns)
        .map(|i| {
            if fixture == 0 || fixture == 4 {
                if fixture == 0 { 1.0 } else { 0.173_205_08 }
            } else {
                f32::from_bits(patterns[(i / 3 + fixture * 7) % patterns.len()])
            }
        })
        .collect();

    [values, other, row, column]
}

fn measure(device: &CudaDevice, old: &CudaGraph<()>, new: &CudaGraph<()>, side: i32) -> TestResult {
    for clear_l2 in [false, true] {
        let options = BenchOptions {
            warmup: Duration::from_millis(50),
            rep: Duration::from_millis(100),
            min_reps: 30,
            max_reps: 100,
            clear_l2,
        };
        let (a, b) = do_bench_paired(
            &device.stream,
            &options,
            |_| {
                old.launch()
                    .sync_on(&device.stream)
                    .map_err(|e| tensor_error(&e.to_string()))
            },
            |_| {
                new.launch()
                    .sync_on(&device.stream)
                    .map_err(|e| tensor_error(&e.to_string()))
            },
        )?;
        println!(
            "{}",
            serde_json::json!({"shape": [1024, 5120], "side": side, "clear_l2": clear_l2, "baseline_ms": a.median_ms(), "fused_ms": b.median_ms(), "baseline_samples": a.times_ms(), "fused_samples": b.times_ms()})
        );
    }
    Ok(())
}
