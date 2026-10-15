use super::{
    CANDIDATES, SCHEMA, compile_options,
    evidence::{Key, Shape, Times, Trial},
};
use crate::device::{CudaDevice, device_error};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    prelude::*,
};
use infer_core::{Error, Result};
use std::time::Duration;
pub(super) const JIT_OPT_LEVEL: u8 = 3;
pub(super) const GLOBAL_SCALES: [f32; 4] = [0.125, 1.0, 64.0, 1024.0];

// Each group independently exercises one quantization regime; no group mixes tiny and
// saturated values into the same amax. These finite samples are shared by every candidate.
const FIXTURE_GROUPS: [[f32; 16]; 8] = [
    [
        0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0,
    ],
    [
        1e-10, -1e-10, 2e-10, -2e-10, 3e-10, -3e-10, 0.0, -0.0, 1e-10, -1e-10, 2e-10, -2e-10,
        3e-10, -3e-10, 0.0, -0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0,
    ],
    [
        1.75, -0.375, 1.75, -0.375, 1.75, -0.375, 1.75, -0.375, 1.75, -0.375, 1.75, -0.375, 1.75,
        -0.375, 1.75, -0.375,
    ],
    [
        1000.0, -1200.0, 1500.0, -1700.0, 1300.0, -1100.0, 0.0, -0.0, 1000.0, -1200.0, 1500.0,
        -1700.0, 1300.0, -1100.0, 0.0, -0.0,
    ],
    [
        0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25,
    ],
    [
        -1.7, -1.4, -1.1, -0.8, -0.5, -0.2, 0.1, 0.4, 0.7, 1.0, 1.3, 1.6, -0.4, 0.3, -0.6, 0.5,
    ],
    [
        1.4, 1.1, 0.8, 0.5, 0.2, -0.1, -0.4, -0.7, -1.0, -1.3, -1.6, 0.4, -0.3, 0.6, -0.5, 0.7,
    ],
];
const MAX_FINITE_POSITIVE_SCALE: u8 = 126;

pub struct Buffers {
    shape: Shape,
    input: Tensor<f32>,
    codes: Tensor<f4e2m1fnx2>,
    scales: Tensor<f8e4m3fn>,
}
impl Buffers {
    pub fn metadata(device: &CudaDevice, shape: Shape) -> Result<Self> {
        Ok(Self {
            shape,
            input: api::meta::<f32>(&[shape.rows, shape.columns])
                .sync_on(&device.stream)
                .map_err(device_error)?,
            codes: api::meta::<f4e2m1fnx2>(&[shape.rows, shape.columns / 2])
                .sync_on(&device.stream)
                .map_err(device_error)?,
            scales: api::meta::<f8e4m3fn>(&[
                shape.rows,
                shape.columns / crate::constants::NVFP4_GROUP_SIZE,
            ])
            .sync_on(&device.stream)
            .map_err(device_error)?,
        })
    }
    pub fn new(device: &CudaDevice, shape: Shape) -> Result<Self> {
        let input = FIXTURE_GROUPS
            .into_iter()
            .flatten()
            .cycle()
            .take(shape.rows * shape.columns)
            .collect();
        Ok(Self {
            shape,
            input: Arc::try_unwrap(device.upload(input, &[shape.rows, shape.columns])?)
                .map_err(|_| Error::invariant("unique quant fixture"))?,
            codes: api::zeros::<f4e2m1fnx2>(&[shape.rows, shape.columns / 2])
                .sync_on(&device.stream)
                .map_err(device_error)?,
            scales: api::zeros::<f8e4m3fn>(&[
                shape.rows,
                shape.columns / crate::constants::NVFP4_GROUP_SIZE,
            ])
            .sync_on(&device.stream)
            .map_err(device_error)?,
        })
    }
    pub fn key(&mut self, device: &CudaDevice, shape: Shape) -> Result<Key> {
        use cutile::cutile_compiler::cuda_tile_runtime_utils as compiler;
        let input = self
            .input
            .view(&[shape.rows, shape.columns])
            .map_err(device_error)?;
        let mut ids = Vec::new();
        for occupancy in std::iter::once(0).chain(CANDIDATES) {
            ids.push(
                crate::resident::nvfp4_gemm::kernels::quantize(
                    (&mut self.codes).partition(crate::constants::NVFP4_QUANT_CODES_TILE),
                    (&mut self.scales).partition(crate::constants::NVFP4_QUANT_SCALES_TILE),
                    &input,
                    1.0f32,
                )
                .generics(vec![shape.columns.to_string()])
                .compile_options(compile_options(occupancy))
                .l2_cache_key_on(&device.stream)
                .map_err(device_error)?,
            );
        }
        Ok(Key {
            hardware: serde_json::to_string(device.profile()?).map_err(device_error)?,
            driver_version: CudaDevice::driver_version()?,
            driver_build: driver_build(),
            shape,
            kernel_fingerprint: format!(
                "{SCHEMA}:{}:{:?}:{:?}:{:?}:{}",
                compiler::tileiras_fingerprint(),
                compiler::bytecode_override(),
                compile_options(0),
                GLOBAL_SCALES.map(f32::to_bits),
                serde_json::to_string(&ids).map_err(device_error)?
            ),
        })
    }
    pub fn capture(
        &mut self,
        device: &CudaDevice,
        occupancy: i32,
        global: f32,
    ) -> Result<CudaGraph<()>> {
        let input = self
            .input
            .view(&[self.shape.rows, self.shape.columns])
            .map_err(device_error)?;
        CudaGraph::scope(&device.stream, |scope| {
            scope.record(
                crate::resident::nvfp4_gemm::kernels::quantize(
                    (&mut self.codes).partition(crate::constants::NVFP4_QUANT_CODES_TILE),
                    (&mut self.scales).partition(crate::constants::NVFP4_QUANT_SCALES_TILE),
                    &input,
                    global,
                )
                .generics(vec![self.shape.columns.to_string()])
                .compile_options(compile_options(occupancy)),
            )?;
            Ok(())
        })
        .map_err(device_error)
    }
    pub fn read(&self, device: &CudaDevice, graph: &CudaGraph<()>) -> Result<Vec<u8>> {
        graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let mut bytes: Vec<_> = device
            .read_borrowed(&self.codes)?
            .into_iter()
            .map(f4e2m1fnx2::to_bits)
            .collect();
        let scales = device.read_borrowed(&self.scales)?;
        if scales.iter().any(|s| s.0 > MAX_FINITE_POSITIVE_SCALE) {
            return Err(Error::invariant("nonfinite or negative quantization scale"));
        }
        bytes.extend(scales.into_iter().map(|s| s.0));
        Ok(bytes)
    }
    pub fn reference(&mut self, device: &CudaDevice) -> Result<Vec<Vec<u8>>> {
        GLOBAL_SCALES
            .into_iter()
            .map(|global| {
                let graph = self.capture(device, 0, global)?;
                self.read(device, &graph)
            })
            .collect()
    }
    pub fn verify(
        &mut self,
        device: &CudaDevice,
        occupancy: i32,
        expected: &[Vec<u8>],
    ) -> Result<()> {
        if expected.len() != GLOBAL_SCALES.len() {
            return Err(Error::invariant("incomplete quantizer fixture set"));
        }
        for (global, expected) in GLOBAL_SCALES.into_iter().zip(expected) {
            let graph = self.capture(device, occupancy, global)?;
            bits(&self.read(device, &graph)?, expected)?;
        }
        Ok(())
    }
}
fn driver_build() -> Option<String> {
    std::fs::read_to_string("/proc/driver/nvidia/version")
        .ok()?
        .lines()
        .find(|l| l.starts_with("NVRM version:"))
        .map(str::trim)
        .map(str::to_owned)
}
pub fn bits(actual: &[u8], expected: &[u8]) -> Result<()> {
    if actual.is_empty() || actual != expected {
        return Err(Error::invariant("quantizer changed native codes or scales"));
    }
    Ok(())
}
pub fn measure(
    device: &CudaDevice,
    baseline: &CudaGraph<()>,
    candidate: &CudaGraph<()>,
    occupancy: i32,
    reverse: bool,
    verified_bytes: usize,
) -> Result<Trial> {
    let (left, right) = if reverse {
        (candidate, baseline)
    } else {
        (baseline, candidate)
    };
    let measure = |cold| -> Result<Times> {
        let opts = BenchOptions {
            warmup: Duration::from_millis(super::super::WARMUP_MS),
            rep: Duration::from_millis(super::super::REP_MS),
            min_reps: super::super::MIN_REPS,
            max_reps: super::super::MAX_REPS,
            clear_l2: cold,
        };
        let run = |g: &CudaGraph<()>| {
            g.launch()
                .sync_on(&device.stream)
                .map_err(|e| cutile::error::tensor_error(&e.to_string()))
        };
        let (a, b) = do_bench_paired(&device.stream, &opts, |_| run(left), |_| run(right))
            .map_err(device_error)?;
        let (a, b) = if reverse { (b, a) } else { (a, b) };
        Ok(Times {
            baseline_ms: a.times_ms().to_vec(),
            candidate_ms: b.times_ms().to_vec(),
        })
    };
    Ok(Trial {
        occupancy,
        candidate_first: reverse,
        verified_bytes,
        cold: measure(true)?,
        warm: measure(false)?,
    })
}
