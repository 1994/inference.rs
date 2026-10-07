use super::kernels;
use crate::device::CudaDevice;
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    prelude::*,
};
use std::time::Duration;

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn recurrent_chunk_performance_gate() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("benchmark requires release".into());
    }
    let compile = cutile::cutile_compiler::hints::CompileOptions::default();
    if compile.device_debug || compile.opt_level.is_some_and(|x| x != 3) {
        return Err("benchmark requires CUDA O3".into());
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for (kh, vh, dim, lanes) in [(2, 4, 32, 32), (16, 48, 128, 32), (16, 48, 128, 128)] {
        bench_case(&device, (kh, vh, dim, lanes))?;
    }
    Ok(())
}
fn bench_case(
    device: &CudaDevice,
    geometry: (usize, usize, usize, usize),
) -> Result<(), Box<dyn std::error::Error>> {
    let (kh, vh, dim, lanes) = geometry;
    let cols = (2 * kh + vh) * dim;
    let qkv = device.upload(
        super::tests::values(lanes * cols, 17),
        &[lanes, 2 * kh + vh, dim],
    )?;
    let beta = device.upload(vec![0.2_f32; lanes * vh], &[lanes, vh])?;
    let alpha = device.upload(vec![-0.1_f32; lanes * vh], &[lanes, vh])?;
    let a_log = device.upload(vec![-1.0_f32; vh], &[vh])?;
    let bias = device.upload(vec![0.1_f32; vh], &[vh])?;
    let mut state = api::zeros::<f32>(&[vh, dim, dim]).sync_on(&device.stream)?;
    let mut old_outputs = Vec::new();
    let mut metadata = Vec::new();
    for lane in 0..lanes {
        old_outputs.push(api::zeros::<f32>(&[vh, dim]).sync_on(&device.stream)?);
        metadata.push(device.upload(vec![i32::try_from(lane)?, 0, i32::try_from(lane)?, 0], &[4])?);
    }
    let baseline = CudaGraph::scope(&device.stream, |scope| {
        for (lane, output) in old_outputs.iter_mut().enumerate() {
            scope.record(
                super::super::recurrent::recurrent::delta(
                    output.partition([1, dim]),
                    (&mut state).partition([1, dim, dim]),
                    &qkv.view(&[lanes * cols])?
                        .slice(std::slice::from_ref(&(lane * cols..(lane + 1) * cols)))?,
                    &beta
                        .view(&[lanes * vh])?
                        .slice(std::slice::from_ref(&(lane * vh..(lane + 1) * vh)))?,
                    &alpha
                        .view(&[lanes * vh])?
                        .slice(std::slice::from_ref(&(lane * vh..(lane + 1) * vh)))?,
                    &a_log,
                    &bias,
                    &metadata[lane],
                )
                .generics(vec![kh.to_string(), vh.to_string(), dim.to_string()]),
            )?;
        }
        Ok(())
    })?;
    let metadata = device.upload(vec![0, i32::try_from(lanes)?, 0, 0], &[4])?;
    let mut scratch = api::zeros::<f32>(&[vh, lanes, dim]).sync_on(&device.stream)?;
    let mut output = api::zeros::<f32>(&[lanes, vh, dim]).sync_on(&device.stream)?;
    let candidate = CudaGraph::scope(&device.stream, |scope| {
        scope.record(
            kernels::delta(
                (&mut state).partition([1, dim, dim]),
                (&mut scratch).partition([1, lanes, dim]),
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
            ]),
        )?;
        scope.record(
            kernels::transpose((&mut output).partition([1, 1, dim]), &scratch)
                .generics(vec![dim.to_string()]),
        )?;
        Ok(())
    })?;
    compare_graphs(device, &baseline, &candidate, geometry)
}

fn compare_graphs(
    device: &CudaDevice,
    baseline: &CudaGraph<()>,
    candidate: &CudaGraph<()>,
    geometry: (usize, usize, usize, usize),
) -> Result<(), Box<dyn std::error::Error>> {
    let options = BenchOptions {
        warmup: Duration::from_millis(50),
        rep: Duration::from_millis(100),
        min_reps: 10,
        max_reps: 100,
        clear_l2: true,
    };
    let (old, new) = do_bench_paired(
        &device.stream,
        &options,
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
        "geometry={geometry:?} old_ms={} new_ms={} speedup={}",
        old.median_ms(),
        new.median_ms(),
        old.median_ms() / new.median_ms()
    );
    assert!(
        new.median_ms() <= old.median_ms() * 1.05,
        "recurrent chunk regresses beyond 5% noise allowance"
    );
    Ok(())
}
