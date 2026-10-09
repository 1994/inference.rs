//! Same-weight, same-width comparison isolates recurrent capture from GEMM policy.
use crate::{
    device::CudaDevice,
    loading::{LoadOptions, LoadedModel},
};
use infer_core::{Error, ModelId, Result};

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL and INFER_TEST_TOKENS (JSON token array)"]
fn checkpoint_chunk_recurrence_matches_legacy_logits() -> Result<()> {
    let path = std::env::var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens_path =
        std::env::var("INFER_TEST_TOKENS").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens: Vec<u32> = serde_json::from_slice(
        &std::fs::read(tokens_path).map_err(|e| Error::invalid(e.to_string()))?,
    )
    .map_err(|e| Error::invalid(e.to_string()))?;
    if tokens.is_empty() {
        return Err(Error::invalid("empty checkpoint test prompt"));
    }
    let width = std::env::var("INFER_TEST_PREFILL_WIDTH")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|e| Error::invalid(e.to_string()))
        })
        .transpose()?
        .unwrap_or(32);
    if ![32, 64, 128].contains(&width) {
        return Err(Error::invalid("checkpoint test prefill width"));
    }
    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::new(1)?,
        LoadOptions {
            prefill_width: width,
            verification_width: 3,
            autotune: false,
            ..LoadOptions::default()
        },
    )?;
    let capacity = (tokens.len() + 32).next_power_of_two();
    let mut candidate = loaded.sequence(capacity)?;
    super::LEGACY_CAPTURE.set(true);
    let legacy = loaded.sequence(capacity);
    super::LEGACY_CAPTURE.set(false);
    let mut legacy = legacy?;
    for (chunk, tokens) in tokens.chunks(width).enumerate() {
        let actual = candidate.prefill_batch(tokens, chunk * width, true)?;
        let expected = legacy.prefill_batch(tokens, chunk * width, true)?;
        // The two captures must agree on state, not only on the readout: when they did not, the
        // divergence was already in the recurrence and not in whatever consumes it.
        if chunk == 0 {
            let mut worst = 0.0_f32;
            let mut source = "none";
            for (id, a) in candidate.states() {
                let Some(b) = legacy.states().get(id) else {
                    continue;
                };
                for (ta, tb) in a.iter().zip(b) {
                    let va = loaded.device().read_borrowed(ta)?;
                    let vb = loaded.device().read_borrowed(tb)?;
                    let error = va
                        .iter()
                        .zip(&vb)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0_f32, f32::max);
                    if error > worst {
                        worst = error;
                        source = match loaded
                            .graph()
                            .nodes
                            .iter()
                            .find(|n| n.states.contains(id))
                            .map(|n| &n.op)
                        {
                            Some(infer_ir::TensorOp::Delta { .. }) => "delta",
                            Some(infer_ir::TensorOp::Conv { .. }) => "conv",
                            _ => "other",
                        };
                    }
                }
            }
            eprintln!("state agreement after chunk 0: worst {worst} ({source})");
        }
        for (a, b) in actual.iter().zip(&expected) {
            compare(&a.0, &b.0, "hidden");
            compare(&a.1, &b.1, "logits");
        }
    }
    // Also exercise captured verification, including rollback and continued decode.
    let position = tokens.len();
    let verify: Vec<_> = tokens
        .iter()
        .copied()
        .cycle()
        .take(candidate.batch_width())
        .collect();
    let actual = candidate.step_batch(&verify, position)?;
    let expected = legacy.step_batch(&verify, position)?;
    for (a, b) in actual.iter().zip(&expected) {
        compare(&a.0, &b.0, "verify hidden");
        compare(&a.1, &b.1, "verify logits");
    }
    candidate.commit_batch(1)?;
    legacy.commit_batch(1)?;
    for step in 0..16 {
        let position = tokens.len() + 1 + step;
        let token = tokens[step % tokens.len()];
        let a = candidate.step(token, position, position, None, true)?;
        let b = legacy.step(token, position, position, None, true)?;
        compare(&a.0, &b.0, "decode hidden");
        compare(&a.1, &b.1, "decode logits");
    }
    Ok(())
}

fn compare(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return;
    }
    let error = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>();
    let norm = b.iter().map(|&b| f64::from(b).powi(2)).sum::<f64>();
    let relative = (error / norm.max(1e-20)).sqrt();
    let maximum = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("{label}: relative_l2={relative:.8}, max_abs={maximum:.8}");
    assert!(relative <= 0.0001 && maximum <= 0.01, "{label} drift");
}
