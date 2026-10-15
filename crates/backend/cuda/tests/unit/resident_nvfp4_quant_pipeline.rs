//! Same-allocation real-weight NVFP4 quantization plus mature GEMM cost.
use super::{Workspace, quant_discovery, record_with_input_pointer};
use crate::{
    device::CudaDevice,
    loading::{Projection, floats},
    resident::ProgramWeights,
};
use cutile::prelude::*;
use infer_core::{ModelId, OpId, TensorId};
use infer_ir::{DType as IrDType, DataflowGraph, TensorNode, TensorOp, TensorSpec, TensorStorage};
use infer_models::{DeviceWeight, QuantizedPackage, WeightEncoding};
use std::{sync::Arc, time::Duration};
type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

struct ProjectionCase {
    name: String,
    columns: usize,
    outputs: usize,
    global: f32,
    graph: DataflowGraph,
    weights: ProgramWeights,
}
impl ProjectionCase {
    fn load(
        device: &CudaDevice,
        package: &mut QuantizedPackage,
        name: String,
        source: &DeviceWeight,
    ) -> TestResult<Self> {
        let [outputs, columns] = source.shape.as_slice() else {
            return Err("projection rank".into());
        };
        let mut weights = super::quant_reuse_tests::fixture_weights();
        let projection = Projection::load(device, package, source)?.resident();
        projection.validate(*outputs, *columns)?;
        let scale = floats(
            package,
            source
                .input_global_scale
                .as_ref()
                .ok_or("activation scale missing")?,
        )?;
        let [global] = scale.as_slice() else {
            return Err("activation scale rank".into());
        };
        if !global.is_finite() || *global <= 0.0 {
            return Err("activation scale invalid".into());
        }
        let id = TensorId::new(10)?;
        weights.projections.insert(id, projection);
        weights.input_scales.insert(id, *global);
        let graph = DataflowGraph {
            tensors: [(1, *columns), (2, *outputs)]
                .into_iter()
                .map(|(id, size)| {
                    Ok(TensorSpec {
                        id: TensorId::new(id)?,
                        shape: vec![size],
                        dtype: IrDType::F32,
                        storage: TensorStorage::Activation,
                    })
                })
                .collect::<infer_core::Result<_>>()?,
            nodes: vec![TensorNode {
                id: OpId::new(1)?,
                layer: None,
                op: TensorOp::Linear,
                inputs: vec![TensorId::new(1)?, id],
                outputs: vec![TensorId::new(2)?],
                states: vec![],
            }],
            ..DataflowGraph::default()
        };
        println!(
            "{}",
            serde_json::json!({"kind":"quant_pipeline_checkpoint","name":name,"source":source,
            "global":global,"shape":[outputs,columns],"weight_requantized":false,"autotune":false})
        );
        Ok(Self {
            name,
            columns: *columns,
            outputs: *outputs,
            global: *global,
            graph,
            weights,
        })
    }
}
struct Case {
    baseline: CudaGraph<()>,
    candidate: CudaGraph<()>,
    workspace: Workspace,
    input: Tensor<f32>,
    output: Tensor<f32>,
    projection: Arc<ProjectionCase>,
    rows: usize,
    pointers: serde_json::Value,
    strategies: serde_json::Value,
}
impl Case {
    fn new(device: &CudaDevice, projection: Arc<ProjectionCase>, rows: usize) -> TestResult<Self> {
        let mut workspace =
            Workspace::new(device, &projection.graph, &projection.weights, &[rows])?;
        let input = api::zeros::<f32>(&[rows, projection.columns]).sync_on(&device.stream)?;
        let mut output = api::zeros::<f32>(&[rows, projection.outputs]).sync_on(&device.stream)?;
        workspace.set_quant_occupancy(0);
        let baseline = capture(
            device,
            &projection,
            &mut workspace,
            &input,
            &mut output,
            rows,
        )?;
        workspace.set_quant_occupancy(6);
        let candidate = capture(
            device,
            &projection,
            &mut workspace,
            &input,
            &mut output,
            rows,
        )?;
        workspace.set_quant_occupancy(0);
        let (q, qs) = workspace
            .quantized_fp4(rows, projection.columns)
            .ok_or("quant buffers missing")?;
        let pointers = serde_json::json!({"input":pointer(&input),"output":pointer(&output),
            "codes":pointer(q),"scales":pointer(qs),"weights":super::quant_reuse_tests::projection_pointers(&projection.weights),
            "same_all_allocations_both_arms":true});
        let strategies = workspace.prepared_configs();
        Ok(Self {
            baseline,
            candidate,
            workspace,
            input,
            output,
            projection,
            rows,
            pointers,
            strategies,
        })
    }
    fn update(&mut self, device: &CudaDevice, fixture: usize) -> TestResult<Vec<f32>> {
        let values = quant_discovery::values(self.rows * self.projection.columns, fixture)?;
        let upload = device.upload(values.clone(), &[self.rows, self.projection.columns])?;
        api::memcpy(&mut self.input, &upload).sync_on(&device.stream)?;
        Ok(values)
    }
    fn snapshot(
        &self,
        device: &CudaDevice,
        graph: &CudaGraph<()>,
    ) -> TestResult<(Vec<u8>, Vec<u8>, Vec<f32>)> {
        graph.launch().sync_on(&device.stream)?;
        let (q, qs) = self
            .workspace
            .quantized_fp4(self.rows, self.projection.columns)
            .ok_or("quant buffers missing")?;
        Ok((
            device
                .read_borrowed(q)?
                .into_iter()
                .map(cuda_core::f4e2m1fnx2::to_bits)
                .collect(),
            device.read_borrowed(qs)?.into_iter().map(|s| s.0).collect(),
            device.read_borrowed(&self.output)?,
        ))
    }
    fn validate(&mut self, device: &CudaDevice) -> TestResult {
        for fixture in 0..8 {
            let values = self.update(device, fixture)?;
            let cpu = quant_discovery::reference(&values, self.projection.global)?;
            for reverse in [false, true] {
                let (a, b) = if reverse {
                    (&self.candidate, &self.baseline)
                } else {
                    (&self.baseline, &self.candidate)
                };
                let expected = self.snapshot(device, a)?;
                let actual = self.snapshot(device, b)?;
                assert_eq!((&expected.0, &expected.1), (&cpu.0, &cpu.1));
                assert_eq!((&actual.0, &actual.1), (&cpu.0, &cpu.1));
                super::quant_reuse_tests::check_outputs(&expected.2, &actual.2, "projection")?;
            }
        }
        println!(
            "{}",
            serde_json::json!({"kind":"quant_pipeline_bits","name":self.projection.name,
            "shape":[self.rows,self.projection.outputs,self.projection.columns],"global":self.projection.global,
            "fixtures":8,"AB_BA":true,"CPU_codes_scales_match":true,"full_output_finite_bitwise":true,
            "same_addresses":self.pointers,"strategies":self.strategies,"occupancy":6})
        );
        Ok(())
    }
    fn cost(&self, device: &CudaDevice, round: usize, reverse: bool, cold: bool) -> TestResult {
        let opts = cutile::bench::BenchOptions {
            warmup: Duration::from_millis(15),
            rep: Duration::from_millis(30),
            min_reps: 16,
            max_reps: 32,
            clear_l2: cold,
        };
        let (a, b) = if reverse {
            (&self.candidate, &self.baseline)
        } else {
            (&self.baseline, &self.candidate)
        };
        let run = |g: &CudaGraph<()>| {
            g.launch()
                .sync_on(&device.stream)
                .map_err(|e| cutile::error::tensor_error(&e.to_string()))
        };
        let (a, b) = cutile::bench::do_bench_paired(&device.stream, &opts, |_| run(a), |_| run(b))?;
        let (a, b) = if reverse { (b, a) } else { (a, b) };
        println!(
            "{}",
            serde_json::json!({"kind":"quant_pipeline_cost","name":self.projection.name,
            "shape":[self.rows,self.projection.outputs,self.projection.columns],"occupancy":6,
            "round":round,"candidate_first":reverse,"clear_l2":cold,"baseline_ms":a.times_ms(),"candidate_ms":b.times_ms(),
            "strategies":self.strategies,"pointers":self.pointers,"scope":"full quantization + mature GEMM",
            "all_numerics_before_timing":true,"production_admission":false})
        );
        Ok(())
    }
}
fn pointer<T: DType>(tensor: &Tensor<T>) -> u64 {
    tensor.device_pointer().cu_deviceptr()
}
fn capture(
    device: &CudaDevice,
    p: &ProjectionCase,
    workspace: &mut Workspace,
    input: &Tensor<f32>,
    output: &mut Tensor<f32>,
    rows: usize,
) -> TestResult<CudaGraph<()>> {
    workspace.begin_quant_capture();
    workspace.enter_quant_node(0, None);
    let view = input.view(&[rows, p.columns])?;
    let id = TensorId::new(10)?;
    let weight = p.weights.projections.get(&id).ok_or("weight missing")?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        if !record_with_input_pointer(
            scope,
            workspace,
            weight,
            p.weights.activation_quantization(id),
            &view,
            output,
            p.columns,
            pointer(input),
        )? {
            return Err(DeviceError::Launch("projection not recorded".into()));
        }
        Ok(())
    });
    workspace.end_quant_capture();
    assert_eq!(workspace.quant_reuse_stats(), (1, 0));
    Ok(graph?)
}
fn selected(package: &QuantizedPackage) -> TestResult<Vec<(String, DeviceWeight)>> {
    let mut out = Vec::new();
    for kind in [".mlp.gate_proj.", ".mlp.down_proj."] {
        let mut weights = package
            .weights
            .iter()
            .filter(|(_, w)| w.encoding == WeightEncoding::Nvfp4 && w.data.name.contains(kind))
            .map(|(name, w)| {
                let layer = w
                    .data
                    .name
                    .split(".layers.")
                    .nth(1)
                    .and_then(|s| s.split('.').next())
                    .ok_or("layer coordinate")?
                    .parse::<usize>()?;
                Ok((layer, name.clone(), w.clone()))
            })
            .collect::<TestResult<Vec<_>>>()?;
        weights.sort_by_key(|(layer, _, _)| *layer);
        let first = weights.first().ok_or("first weight missing")?;
        let last = weights.last().ok_or("last weight missing")?;
        assert_ne!(first.0, last.0);
        out.push((first.1.clone(), first.2.clone()));
        out.push((last.1.clone(), last.2.clone()));
    }
    Ok(out)
}
#[test]
#[ignore = "requires CUDA/real checkpoint and safe-run, exact full outputs before coupled timing"]
fn nvfp4_quant_occupancy_real_projection_bits_and_cost() -> TestResult {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let mut package = QuantizedPackage::open(std::env::var("INFER_TEST_MODEL")?, ModelId::ONE)?;
    let selected = selected(&package)?;
    let mut cases = Vec::new();
    for (name, source) in selected {
        let projection = Arc::new(ProjectionCase::load(&device, &mut package, name, &source)?);
        for rows in [64, 256] {
            cases.push(Case::new(&device, Arc::clone(&projection), rows)?);
        }
    }
    for case in &mut cases {
        case.validate(&device)?;
    }
    println!(
        "{}",
        serde_json::json!({"kind":"quant_pipeline_all_bits_pass","cases":cases.len(),"timing_started":false})
    );
    for case in &mut cases {
        let _values = case.update(&device, 0)?;
    }
    device.reclaim_barrier()?;
    for round in 0..3 {
        for cold in [true, false] {
            for reverse in [false, true] {
                for case in &cases {
                    case.cost(&device, round, reverse, cold)?;
                }
            }
        }
    }
    Ok(())
}
