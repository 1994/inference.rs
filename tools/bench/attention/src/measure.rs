//! Warm CUDA graph timings, including layout and dtype adaptation.
use crate::{Inputs, Result};
use candle_core::cuda_backend::cudarc::driver::sys;
use candle_core::{DType, Device, Tensor};

const REPEATS: usize = 8;

pub fn run(
    device: &Device,
    inputs: &Inputs,
    dtype: DType,
    expected: &Tensor,
    shape: &serde_json::Value,
) -> Result<()> {
    let (samples, actual) = time(device, || inputs.execute(dtype))?;
    let (q, k, v) = inputs.prepare(dtype)?;
    let (core_samples, core_values) = time(device, || inputs.core(&q, &k, &v))?;
    if !core_values.iter().all(|v| v.is_finite()) {
        return Err("nonfinite core output".into());
    }
    let reference = expected.flatten_all()?.to_vec1::<f32>()?;
    if actual.len() != reference.len() {
        return Err("output shape mismatch".into());
    }
    let finite = actual.iter().chain(&reference).all(|v| v.is_finite());
    if !finite {
        return Err("candidate produced nonfinite values".into());
    }
    let mut error = 0.0f32;
    let mut scale = 0.0f32;
    for (index, (a, b)) in actual.iter().zip(&reference).enumerate() {
        if index % inputs.half < inputs.head / 2 {
            error = error.max((a - b).abs());
            scale = scale.max(b.abs());
        } else if a.abs() > 1e-6 {
            return Err("candidate padding nonzero".into());
        }
    }
    println!(
        "ATTENTION_GATE {}",
        serde_json::json!({
            "case": shape["id"], "shape": shape, "warmup": 100, "samples_ms": samples, "core_samples_ms": core_samples, "replays_per_sample": REPEATS,
            "max_abs_error": error, "reference_max_abs": scale,
            "elements": shape["elements"], "finite": finite,
        })
    );
    Ok(())
}

fn time(
    device: &Device,
    execute: impl Fn() -> candle_core::Result<Tensor>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let cuda = device.as_cuda_device()?;
    let stream = cuda.cuda_stream();
    let _parameter_cache = cuda.enable_cuda_graph_htod_cache();
    for _ in 0..10 {
        let _output = execute()?;
    }
    stream.synchronize()?;
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let output_result = (|| {
        let mut output = execute()?;
        for _ in 1..REPEATS {
            output = execute()?;
        }
        Ok::<_, candle_core::Error>(output)
    })();
    // Always end capture, even if a candidate operation is unsupported.
    let graph_result = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    let output = output_result?;
    let graph = graph_result?.ok_or("empty captured graph")?;
    for _ in 0..100 {
        graph.launch()?;
    }
    stream.synchronize()?;
    let mut samples = Vec::new();
    for _ in 0..60 {
        let start = stream
            .context()
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let end = stream
            .context()
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        start.record(&stream)?;
        graph.launch()?;
        end.record(&stream)?;
        end.synchronize()?;
        samples.push(start.elapsed_ms(&end)? / 8.0);
    }
    let actual = output
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok((samples, actual))
}
