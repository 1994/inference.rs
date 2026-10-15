//! Actual native versus selected Lt model capture; no production policy.
use super::{LoadOptions, LoadedModel, quant_reuse_model_tests as reference};
use crate::device::CudaDevice;
use infer_core::{Error, ModelId, Result};
use std::cell::Cell;
thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static NATIVE: Cell<bool> = const { Cell::new(false) };
    static SITES: Cell<usize> = const { Cell::new(0) };
}
pub fn enabled() -> bool {
    ENABLED.with(Cell::get)
}
pub fn eligible(rows: usize, outputs: i32, columns: usize) -> bool {
    !enabled()
        || matches!(
            (rows, outputs, columns),
            (64, 5120, 17408)
                | (64, 12288, 5120)
                | (256, 6144 | 10240 | 12288 | 17408 | 248_320, 5120)
        )
}
pub fn recorded() {
    if enabled() {
        SITES.with(|s| s.set(s.get() + 1));
    }
}
fn with_vendor<T>(f: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            ENABLED.with(|v| v.set(self.0));
        }
    }
    let _reset = Reset(ENABLED.with(|v| v.replace(true)));
    f()
}
fn compare(
    actual: &crate::resident::BatchOutput,
    expected: &crate::resident::BatchOutput,
    width: usize,
    length: usize,
    chunk: usize,
) {
    let mut hidden = 0;
    let mut logits = 0;
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.0.len(), b.0.len());
        assert_eq!(a.1.len(), b.1.len());
        hidden +=
            a.0.iter()
                .zip(&b.0)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
        logits +=
            a.1.iter()
                .zip(&b.1)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
    }
    println!(
        "{}",
        serde_json::json!({"kind":"fp8_vendor_real_model_outputs","width":width,"length":length,"chunk":chunk,"hidden_bit_differences":hidden,"logit_bit_differences":logits,"vendor_capture_sites":SITES.with(Cell::get),"production_admission":false})
    );
    reference::outputs(actual, expected);
}
#[test]
#[ignore = "requires CUDA/checkpoint/safe-run; full states/logits/hidden before service timing"]
fn preselected_fp8_lt_preserves_full_model_bits() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let path = std::env::var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens = reference::prompt()?;
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 256,
            verification_width: 3,
            mtp_depth: 0,
            autotune: false,
            ..LoadOptions::default()
        },
    )?;
    for width in [64, 256] {
        loaded.weights.prefill_width = width;
        loaded.weights.narrow_prefill_width = 0;
        let mut control = loaded.sequence(1024)?;
        SITES.with(|s| s.set(0));
        let mut tuned = with_vendor(|| loaded.sequence(1024))?;
        assert!(
            SITES.with(Cell::get) > 0,
            "candidate must record actual Lt sites"
        );
        for length in [51, 64, 255, 256, 511] {
            for (chunk, input) in tokens[..length].chunks(width).enumerate() {
                let last = (chunk + 1) * width >= length;
                compare(
                    &tuned.prefill_batch_readout(input, chunk * width, last, last)?,
                    &control.prefill_batch_readout(input, chunk * width, last, last)?,
                    width,
                    length,
                    chunk,
                );
                reference::program_states(&loaded, &tuned, &control)?;
            }
            let mut next = tokens[0];
            for _ in 0..8 {
                let position = control.position();
                let a = control.step(next, position, position, None, true)?;
                let b = tuned.step(next, position, position, None, true)?;
                next = reference::greedy(&a.1)?;
                assert_eq!(next, reference::greedy(&b.1)?);
                reference::outputs(&vec![a], &vec![b]);
                reference::program_states(&loaded, &tuned, &control)?;
            }
            println!(
                "{}",
                serde_json::json!({"fp8_vendor_model_bits":true,"prefill_width":width,
                "length":length,"decode_steps":8,"full_hidden_logits_F32_FP8":true,
                "candidate_capture_sites":SITES.with(Cell::get)})
            );
        }
        for (count, commit) in [(3, 1), (2, 2), (1, 1), (3, 2), (3, 3)] {
            let position = control.position();
            reference::outputs(
                &control.step_batch(&tokens[..count], position)?,
                &tuned.step_batch(&tokens[..count], position)?,
            );
            reference::program_states(&loaded, &tuned, &control)?;
            control.commit_batch(commit)?;
            tuned.commit_batch(commit)?;
            reference::program_states(&loaded, &tuned, &control)?;
        }
    }
    Ok(())
}

pub fn native_baseline() -> bool {
    NATIVE.with(Cell::get)
}
fn with_native<T>(f: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            NATIVE.with(|v| v.set(self.0));
        }
    }
    let _reset = Reset(NATIVE.with(|v| v.replace(true)));
    f()
}

#[test]
#[ignore = "requires CUDA/checkpoint/safe-run; full states/logits/hidden before service timing"]
fn default_fp8_lt_autotune_preserves_full_model_bits() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let path = std::env::var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens = reference::prompt()?;
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 256,
            verification_width: 3,
            mtp_depth: 0,
            autotune: true,
            ..LoadOptions::default()
        },
    )?;
    for width in [64, 256] {
        loaded.weights.prefill_width = width;
        loaded.weights.narrow_prefill_width = 0;
        // Each arm models a fresh preparation session. The shared immutable weights stay
        // identical; baseline capture time must not spend the candidate model's allowance.
        loaded.weights.resident_tuning =
            std::sync::Arc::new(crate::tuning::resident_gemm::ResidentTuningSession::default());
        SITES.with(|s| s.set(0));
        let mut tuned = loaded.sequence(1024)?;
        loaded.weights.resident_tuning =
            std::sync::Arc::new(crate::tuning::resident_gemm::ResidentTuningSession::default());
        let mut control = with_native(|| loaded.sequence(1024))?;
        assert!(
            SITES.with(Cell::get) > 0,
            "candidate must record actual Lt sites"
        );
        for length in [51, 64, 255, 256, 511] {
            for (chunk, input) in tokens[..length].chunks(width).enumerate() {
                let last = (chunk + 1) * width >= length;
                compare(
                    &tuned.prefill_batch_readout(input, chunk * width, last, last)?,
                    &control.prefill_batch_readout(input, chunk * width, last, last)?,
                    width,
                    length,
                    chunk,
                );
                reference::program_states(&loaded, &tuned, &control)?;
            }
            let mut next = tokens[0];
            for _ in 0..8 {
                let position = control.position();
                let a = control.step(next, position, position, None, true)?;
                let b = tuned.step(next, position, position, None, true)?;
                next = reference::greedy(&a.1)?;
                assert_eq!(next, reference::greedy(&b.1)?);
                reference::outputs(&vec![a], &vec![b]);
                reference::program_states(&loaded, &tuned, &control)?;
            }
            println!(
                "{}",
                serde_json::json!({"fp8_default_vendor_model_bits":true,"prefill_width":width,
                "length":length,"decode_steps":8,"full_hidden_logits_F32_FP8":true,
                "candidate_capture_sites":SITES.with(Cell::get)})
            );
        }
        for (count, commit) in [(3, 1), (2, 2), (1, 1), (3, 2), (3, 3)] {
            let position = control.position();
            reference::outputs(
                &control.step_batch(&tokens[..count], position)?,
                &tuned.step_batch(&tokens[..count], position)?,
            );
            reference::program_states(&loaded, &tuned, &control)?;
            control.commit_batch(commit)?;
            tuned.commit_batch(commit)?;
            reference::program_states(&loaded, &tuned, &control)?;
        }
    }
    Ok(())
}

pub fn selected() {
    SITES.with(|s| s.set(s.get() + 1));
}
