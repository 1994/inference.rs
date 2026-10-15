//! Native FP4 activation quantizer occupancy: full independent byte gates before timing.
use super::{
    kernels,
    quant_reuse_tests::{decode_fp8, nearest},
};
use crate::device::CudaDevice;
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::{prelude::*, tile_kernel::CompileOptions};
use std::time::Duration;
type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;
struct Buffers {
    input: Tensor<f32>,
    codes: Tensor<f4e2m1fnx2>,
    scales: Tensor<f8e4m3fn>,
    rows: usize,
    columns: usize,
}
impl Buffers {
    fn new(device: &CudaDevice, rows: usize, columns: usize) -> TestResult<Self> {
        Ok(Self {
            input: api::zeros::<f32>(&[rows, columns]).sync_on(&device.stream)?,
            codes: api::zeros::<f4e2m1fnx2>(&[rows, columns / 2]).sync_on(&device.stream)?,
            scales: api::zeros::<f8e4m3fn>(&[rows, columns / 16]).sync_on(&device.stream)?,
            rows,
            columns,
        })
    }
    fn capture(
        &mut self,
        device: &CudaDevice,
        global: f32,
        occupancy: i32,
    ) -> TestResult<CudaGraph<()>> {
        let input = self.input.view(&[self.rows, self.columns])?;
        let options = if occupancy == 0 {
            CompileOptions::default()
        } else {
            CompileOptions::default().occupancy(occupancy)
        };
        Ok(CudaGraph::scope(&device.stream, |scope| {
            scope.record(
                kernels::quantize(
                    (&mut self.codes).partition([1, 256]),
                    (&mut self.scales).partition([1, 32]),
                    &input,
                    global,
                )
                .generics(vec![self.columns.to_string()])
                .compile_options(options),
            )?;
            Ok(())
        })?)
    }
    fn update(&mut self, device: &CudaDevice, fixture: usize) -> TestResult<Vec<f32>> {
        let v = values(self.rows * self.columns, fixture)?;
        let upload = device.upload(v.clone(), &[self.rows, self.columns])?;
        api::memcpy(&mut self.input, &upload).sync_on(&device.stream)?;
        Ok(v)
    }
    fn read(&self, device: &CudaDevice, graph: &CudaGraph<()>) -> TestResult<(Vec<u8>, Vec<u8>)> {
        graph.launch().sync_on(&device.stream)?;
        Ok((
            device
                .read_borrowed(&self.codes)?
                .into_iter()
                .map(f4e2m1fnx2::to_bits)
                .collect(),
            device
                .read_borrowed(&self.scales)?
                .into_iter()
                .map(|s| s.0)
                .collect(),
        ))
    }
}
pub(super) fn values(n: usize, fixture: usize) -> TestResult<Vec<f32>> {
    (0..n)
        .map(|i| {
            let x = (f32::from(u16::try_from((i * 17 + i / 512 * 23) % 131)?) - 65.0) / 37.0;
            Ok(match fixture {
                0 => x,
                1 => {
                    if i.is_multiple_of(2) {
                        0.0
                    } else {
                        -0.0
                    }
                }
                2 => x * 1e-10,
                3 => {
                    if i % 47 < 45 {
                        0.0
                    } else {
                        x
                    }
                }
                4 => {
                    if i.is_multiple_of(2) {
                        1.75
                    } else {
                        -0.375
                    }
                }
                5 => x * 1000.0,
                6 => [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25][i % 8],
                _ => x.mul_add(-0.75, 0.125),
            })
        })
        .collect()
}
pub(super) fn reference(values: &[f32], global: f32) -> TestResult<(Vec<u8>, Vec<u8>)> {
    let table: Vec<_> = (0..=126).map(decode_fp8).collect();
    let levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut codes = Vec::with_capacity(values.len() / 2);
    let mut scales = Vec::with_capacity(values.len() / 16);
    for group in values.as_chunks::<16>().0 {
        let amax = group.iter().copied().map(f32::abs).fold(0.0, f32::max);
        let code = nearest((amax * (global / 6.0)).min(448.0), &table)?;
        scales.push(code);
        let denominator = if code == 0 {
            1.0
        } else {
            table[usize::from(code)]
        };
        let multiplier = global / denominator;
        let mut q = [0u8; 16];
        for (i, &v) in group.iter().enumerate() {
            let x = (v * multiplier).clamp(-6.0, 6.0);
            q[i] = nearest(x.abs(), &levels)? | if x.is_sign_negative() { 8 } else { 0 };
        }
        codes.extend(q.as_chunks::<2>().0.iter().map(|p| p[0] | p[1] << 4));
    }
    Ok((codes, scales))
}
#[test]
#[ignore = "requires CUDA/safe-run; native and independent CPU full codes/scales before any cost"]
fn nvfp4_quant_occupancy_full_bits_and_cost() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for rows in [1, 4, 64, 256] {
        for columns in [5120, 17408] {
            let mut b = Buffers::new(&device, rows, columns)?;
            for global in [0.125, 1.0, 64.0] {
                let baseline = b.capture(&device, global, 0)?;
                let candidates = [2, 4, 6]
                    .map(|o| b.capture(&device, global, o))
                    .into_iter()
                    .collect::<TestResult<Vec<_>>>()?;
                for fixture in 0..8 {
                    let v = b.update(&device, fixture)?;
                    let cpu = reference(&v, global)?;
                    assert_eq!(
                        b.read(&device, &baseline)?,
                        cpu,
                        "native CPU rows{rows} K{columns} fixture{fixture} global{global}"
                    );
                    for (occupancy, graph) in [2, 4, 6].into_iter().zip(&candidates) {
                        assert_eq!(b.read(&device, graph)?, cpu, "candidate {occupancy}");
                    }
                }
                println!(
                    "{}",
                    serde_json::json!({"kind":"fp4_quant_bits","rows":rows,"columns":columns,
    "global":global,"fixtures":8,"candidates":[2,4,6],"all_codes_and_scales_match_CPU":true,
    "same_addresses":true,"elements":rows*columns})
                );
            }
            let baseline = b.capture(&device, 1.0, 0)?;
            let v = b.update(&device, 0)?;
            let cpu = reference(&v, 1.0)?;
            for occupancy in [2, 4, 6] {
                let graph = b.capture(&device, 1.0, occupancy)?;
                assert_eq!(b.read(&device, &graph)?, cpu);
                cost(&device, &baseline, &graph, rows, columns, occupancy)?;
                assert_eq!(b.read(&device, &graph)?, cpu);
            }
        }
    }
    Ok(())
}

fn cost(
    device: &CudaDevice,
    baseline: &CudaGraph<()>,
    candidate: &CudaGraph<()>,
    rows: usize,
    columns: usize,
    occupancy: i32,
) -> TestResult {
    for reverse in [false, true] {
        for cold in [true, false] {
            let opts = cutile::bench::BenchOptions {
                warmup: Duration::from_millis(30),
                rep: Duration::from_millis(60),
                min_reps: 10,
                max_reps: 30,
                clear_l2: cold,
            };
            let (a, b) = if reverse {
                (candidate, baseline)
            } else {
                (baseline, candidate)
            };
            let run = |g: &CudaGraph<()>| {
                g.launch()
                    .sync_on(&device.stream)
                    .map_err(|e| cutile::error::tensor_error(&e.to_string()))
            };
            let (a, b) =
                cutile::bench::do_bench_paired(&device.stream, &opts, |_| run(a), |_| run(b))?;
            let (a, b) = if reverse { (b, a) } else { (a, b) };
            println!(
                "{}",
                serde_json::json!({"kind":"fp4_quant_cost","rows":rows,"columns":columns,
   "occupancy":occupancy,"clear_l2":cold,"candidate_first":reverse,"same_addresses":true,
   "baseline_ms":a.times_ms(),"candidate_ms":b.times_ms()})
            );
        }
    }
    Ok(())
}

thread_local! {
    static MODEL_BASELINE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static MODEL_SITES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
pub fn with_model_baseline<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            MODEL_BASELINE.with(|v| v.set(self.0));
        }
    }
    let previous = MODEL_BASELINE.with(|v| v.replace(true));
    let _restore = Restore(previous);
    f()
}
pub fn model_occupancy(occupancy: i32) -> i32 {
    if MODEL_BASELINE.with(std::cell::Cell::get) {
        0
    } else {
        if occupancy > 0 {
            MODEL_SITES.with(|v| v.set(v.get() + 1));
        }
        occupancy
    }
}
pub fn model_sites() -> usize {
    MODEL_SITES.with(std::cell::Cell::get)
}
