//! Actual checkpoint activations: compare full hidden/logit/state bits, not just greedy tokens.
use super::{
    LoadOptions, LoadedModel,
    quant_reuse_model_tests::{outputs, program_states},
};
use crate::{
    AttentionProfile,
    device::CudaDevice,
    resident::{norm_maximum_captures, with_separate_norm_maximum},
};
use infer_core::{Error, ModelId, Result};
use std::{env::var, fs::read};

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL and frozen 1025-token INFER_TEST_TOKENS; safe-run"]
fn norm_maximum_preserves_checkpoint_model_bits() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let token_path = var("INFER_TEST_TOKENS").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens: Vec<u32> =
        serde_json::from_slice(&read(token_path).map_err(|e| Error::invalid(e.to_string()))?)
            .map_err(|e| Error::invalid(e.to_string()))?;
    assert_eq!(tokens.len(), 1025);
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 1024,
            verification_width: 0,
            mtp_depth: 0,
            autotune: false,
            chunked_recurrent: true,
            attention_profile: AttentionProfile::Bf16,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    loaded.weights.narrow_prefill_width = 0;
    let before = norm_maximum_captures();
    let mut fused = loaded.sequence(2048)?;
    let captured = norm_maximum_captures();
    assert!(captured > before, "must execute the fused model capture");
    let mut separate = with_separate_norm_maximum(|| loaded.sequence(2048))?;
    assert_eq!(
        norm_maximum_captures(),
        captured,
        "reference must use separate norm and maximum"
    );
    for length in [1025, 511, 1024, 1] {
        fused.reset()?;
        separate.reset()?;
        for (chunk, input) in tokens[..length].chunks(1024).enumerate() {
            outputs(
                &fused.prefill_batch(input, chunk * 1024, true)?,
                &separate.prefill_batch(input, chunk * 1024, true)?,
            );
        }
        program_states(&loaded, &fused, &separate)?;
        for step in 0..4 {
            let pos = fused.position();
            let token = tokens[(length + step) % tokens.len()];
            outputs(
                &vec![fused.step(token, pos, pos, None, true)?],
                &vec![separate.step(token, pos, pos, None, true)?],
            );
        }
        program_states(&loaded, &fused, &separate)?;
        println!(
            "checkpoint bits passed: prompt={length}, fused captures={}",
            captured - before
        );
    }
    Ok(())
}
