//! Actual slot-backed prefill/decode and inactive-slot equivalence.
use super::{LoadOptions, LoadedModel, quant_reuse_model_tests as reference};
use crate::device::CudaDevice;
use infer_core::{Error, ModelId, Result};

#[test]
#[ignore = "requires CUDA/checkpoint/safe-run; every captured subset, private states and pooled decode"]
fn grouped_prefill_slots_preserve_full_model_bits() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let path = std::env::var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens = reference::prompt()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 64,
            verification_width: 0,
            mtp_depth: 0,
            autotune: false,
            ..LoadOptions::default()
        },
    )?;
    for capacity in [128, crate::constants::CB_SLOT_TOKENS] {
        check_slots(&loaded, &tokens, capacity)?;
    }
    Ok(())
}

fn check_slots(loaded: &LoadedModel, tokens: &[u32], capacity: usize) -> Result<()> {
    let mut controls = (0..4)
        .map(|_| loaded.sequence(capacity))
        .collect::<Result<Vec<_>>>()?;
    let mut pool = loaded.slot_pool(4, capacity)?;
    assert!(pool.has_prefill(), "production prefill graph must exist");
    for slot in 0..4 {
        assert_eq!(pool.take_free(), Some(slot));
    }
    for mask in 1u32..16 {
        if mask.count_ones() < 2 {
            continue;
        }
        let active: Vec<_> = (0..4).filter(|i| mask & (1u32 << i) != 0).collect();
        let mut expected = Vec::new();
        for &slot in &active {
            controls[slot].reset()?;
            pool.bind_prefill(
                slot,
                controls[slot].states(),
                controls[slot].fp8_states(),
                loaded.device(),
            )?;
            expected.push(controls[slot].prefill_batch_readout(
                &tokens[slot * 4..slot * 4 + 52],
                0,
                true,
                false,
            )?);
        }
        let inputs: Vec<_> = active
            .iter()
            .map(|&slot| (slot, &tokens[slot * 4..slot * 4 + 52]))
            .collect();
        let actual = pool.run_prefill(loaded.device(), &inputs)?;
        let mut next = Vec::new();
        for (index, row) in expected.iter().enumerate() {
            let last = row
                .last()
                .ok_or_else(|| Error::invariant("prefill reference"))?;
            reference::bits(&actual[index].1, &last.1, "slot prefill logits");
            next.push(reference::greedy(&last.1)?);
        }
        compare_states(loaded, &pool, &controls)?;
        for _ in 0..8 {
            let mut reference_rows = Vec::new();
            let mut lanes = Vec::new();
            for (index, &slot) in active.iter().enumerate() {
                let position = controls[slot].position();
                lanes.push((slot, next[index], position, position));
                reference_rows.push(controls[slot].step(
                    next[index],
                    position,
                    position,
                    None,
                    true,
                )?);
            }
            let actual = pool.run_lanes(loaded.device(), &lanes)?;
            for (index, &slot) in active.iter().enumerate() {
                reference::bits(&actual[index].1, &reference_rows[index].1, "pooled logits");
                reference::bits(
                    &pool.test_hidden(loaded.device(), slot)?,
                    &reference_rows[index].0,
                    "pooled hidden",
                );
                next[index] = reference::greedy(&reference_rows[index].1)?;
            }
            compare_states(loaded, &pool, &controls)?;
        }
        println!(
            "{}",
            serde_json::json!({"slot_prefill_bits":true,"mask":mask,"capacity":capacity,
            "prefill_requests":active.len(),"decode_steps":8,"full_F32_and_FP8_KV_bits":true,
            "inactive_slots_unchanged":true})
        );
    }
    Ok(())
}

fn compare_states(
    loaded: &LoadedModel,
    pool: &crate::resident::slot_batch::SlotPool,
    controls: &[crate::resident::DeviceProgram],
) -> Result<()> {
    for (slot, control) in controls.iter().enumerate() {
        let (actual, fp8) = pool.test_states(slot);
        reference::states(
            loaded.device(),
            actual,
            control.states(),
            fp8,
            control.fp8_states(),
        )?;
    }
    Ok(())
}
