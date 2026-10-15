//! Full model and private-state equivalence before testing node-major cache reuse.
use super::{LoadOptions, LoadedModel, quant_reuse_model_tests as reference};
use crate::{device::CudaDevice, resident::GroupedPrefillTest};
use infer_core::{Error, ModelId, Result};
use std::time::Instant;

#[test]
#[ignore = "requires CUDA/checkpoint/frozen inputs/safe-run; all hidden/logits/state bits before timing"]
fn grouped_prefill_node_major_full_model_and_cost() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let model = std::env::var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens = reference::prompt()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        model,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 64,
            verification_width: 0,
            mtp_depth: 0,
            autotune: false,
            ..LoadOptions::default()
        },
    )?;
    for count in [2, 3, 4] {
        let mut controls = (0..count)
            .map(|_| loaded.sequence(256))
            .collect::<Result<Vec<_>>>()?;
        let mut candidates = (0..count)
            .map(|_| loaded.sequence(256))
            .collect::<Result<Vec<_>>>()?;
        let capture_started = Instant::now();
        let mut grouped = GroupedPrefillTest::new(
            loaded.device(),
            loaded.graph(),
            &loaded.weights,
            &mut candidates,
            64,
        )?;
        println!(
            "{}",
            serde_json::json!({"kind":"group_capture","requests":count,
            "capture_ms":capture_started.elapsed().as_secs_f64()*1000.0})
        );
        for (position, length) in [(0, 52), (52, 51), (103, 63)] {
            let inputs: Vec<_> = (0..count)
                .map(|i| (&tokens[i * 4..i * 4 + length], position))
                .collect();
            if position == 0 {
                for p in &mut candidates {
                    p.reset()?;
                }
            }
            let mut expected = Vec::new();
            for (p, (input, base)) in controls.iter_mut().zip(&inputs) {
                expected.push(p.prefill_batch(input, *base, true)?);
            }
            let actual = grouped.run(loaded.device(), &inputs, true)?;
            for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                reference::outputs(a, b);
                candidates[i].grouped_test_advance(length, position)?;
                reference::program_states(&loaded, &candidates[i], &controls[i])?;
            }
            println!(
                "{}",
                serde_json::json!({"kind":"group_model_bits","requests":count,
                "position":position,"tokens_each":length,"hidden_logits_bits":true,
                "all_private_F32_and_FP8_KV_bits":true})
            );
        }
        shared_graph_gate(
            &loaded,
            &tokens,
            &mut controls,
            &mut candidates,
            &mut grouped,
        )?;
        time_group(&loaded, &tokens, &mut candidates, &mut grouped)?;
        drop(grouped);
        drop(candidates);
        drop(controls);
        loaded.device().reclaim_barrier()?;
    }
    Ok(())
}

fn time_group(
    loaded: &LoadedModel,
    tokens: &[u32],
    candidates: &mut [crate::resident::DeviceProgram],
    grouped: &mut GroupedPrefillTest,
) -> Result<()> {
    let count = candidates.len();
    for reverse in [false, true] {
        let mut baseline = Vec::new();
        let mut candidate = Vec::new();
        for _ in 0..5 {
            let inputs: Vec<_> = (0..count)
                .map(|i| (&tokens[i * 4..i * 4 + 52], 0usize))
                .collect();
            let run = |node_major,
                       candidates: &mut [crate::resident::DeviceProgram],
                       grouped: &mut GroupedPrefillTest|
             -> Result<f64> {
                let started = Instant::now();
                for p in candidates.iter_mut() {
                    p.reset()?;
                }
                if node_major {
                    grouped.run(loaded.device(), &inputs, false)?;
                } else {
                    grouped.run_sequential(loaded.device(), &inputs, false)?;
                }
                for p in candidates.iter_mut() {
                    p.grouped_test_advance(52, 0)?;
                }
                Ok(started.elapsed().as_secs_f64() * 1000.0)
            };
            let (a, b) = if reverse {
                let b = run(true, candidates, grouped)?;
                let a = run(false, candidates, grouped)?;
                (a, b)
            } else {
                let a = run(false, candidates, grouped)?;
                let b = run(true, candidates, grouped)?;
                (a, b)
            };
            baseline.push(a);
            candidate.push(b);
        }
        println!(
            "{}",
            serde_json::json!({"kind":"group_complete_cost","requests":count,
            "candidate_first":reverse,"baseline_ms":baseline,"candidate_ms":candidate,
            "includes_metadata_and_logit_readback":true,"kernel_arithmetic_unchanged":true,"same_addresses":true,"same_graph_replays":true})
        );
    }
    Ok(())
}

fn shared_graph_gate(
    loaded: &LoadedModel,
    tokens: &[u32],
    controls: &mut [crate::resident::DeviceProgram],
    candidates: &mut [crate::resident::DeviceProgram],
    grouped: &mut GroupedPrefillTest,
) -> Result<()> {
    let inputs: Vec<_> = (0..controls.len())
        .map(|i| (&tokens[i * 4..i * 4 + 52], 0usize))
        .collect();
    let expected = controls
        .iter_mut()
        .zip(&inputs)
        .map(|(p, (t, b))| p.prefill_batch(t, *b, true))
        .collect::<Result<Vec<_>>>()?;
    for node_major in [false, true] {
        for p in candidates.iter_mut() {
            p.reset()?;
        }
        let actual = if node_major {
            grouped.run(loaded.device(), &inputs, true)?
        } else {
            grouped.run_sequential(loaded.device(), &inputs, true)?
        };
        for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
            candidates[i].grouped_test_advance(52, 0)?;
            reference::outputs(a, b);
            reference::program_states(loaded, &candidates[i], &controls[i])?;
        }
        println!(
            "{}",
            serde_json::json!({"kind":"group_same_address_bits","requests":controls.len(),
            "node_major":node_major,"all_hidden_logits_F32_FP8_bits":true})
        );
    }
    Ok(())
}
