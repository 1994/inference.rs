//! Real target arithmetic across private and pooled verification, including partial commits.
use super::{
    LoadOptions, LoadedModel,
    quant_reuse_model_tests::{bits, outputs, prompt},
    support::seed_prefix,
};
use crate::{
    constants::KV_BLOCK_TOKENS,
    device::CudaDevice,
    resident::{DeviceProgram, slot_batch::SlotPool},
};
use infer_core::{Error, ModelId, Result};
use serde_json::json;
use std::env::var;

fn live_states(
    loaded: &LoadedModel,
    control: &DeviceProgram,
    pool: &SlotPool,
    slot: usize,
) -> Result<()> {
    let (state, fp8) = pool.test_states(slot);
    for (id, tensors) in control.states() {
        for (a, b) in tensors.iter().zip(&state[id]) {
            bits(
                &loaded.device().read_borrowed(a)?,
                &loaded.device().read_borrowed(b)?,
                "committed recurrent state",
            );
        }
    }
    let table = pool.test_table(loaded.device(), slot)?;
    for (id, (keys, values)) in control.fp8_states() {
        let (pk, pv) = &fp8[id];
        for (a, b) in [(keys, pk), (values, pv)] {
            let heads = usize::try_from(a.shape()[0]).unwrap();
            let capacity = usize::try_from(a.shape()[1]).unwrap();
            let physical = usize::try_from(b.shape()[1]).unwrap();
            let dim = usize::try_from(a.shape()[2]).unwrap();
            let a = loaded.device().read_borrowed(a)?;
            let b = loaded.device().read_borrowed(b)?;
            for head in 0..heads {
                for pos in 0..control.position() {
                    let row = usize::try_from(table[pos / KV_BLOCK_TOKENS]).unwrap()
                        * KV_BLOCK_TOKENS
                        + pos % KV_BLOCK_TOKENS;
                    let left = (head * capacity + pos) * dim;
                    let right = (head * physical + row) * dim;
                    assert!(
                        a[left..left + dim]
                            .iter()
                            .zip(&b[right..right + dim])
                            .all(|(a, b)| a.0 == b.0),
                        "live KV slot={slot} position={pos}"
                    );
                }
            }
        }
    }
    Ok(())
}

fn bind(
    loaded: &LoadedModel,
    prefix: &DeviceProgram,
    pool: &mut SlotPool,
    slot: usize,
) -> Result<()> {
    pool.release(slot);
    assert!(pool.take_slot(slot));
    assert!(pool.assign(slot, 128, loaded.device())?);
    pool.bind(
        slot,
        prefix.states(),
        prefix.fp8_states(),
        prefix.position(),
        loaded.device(),
    )
}

fn replay(
    loaded: &LoadedModel,
    prefix: &DeviceProgram,
    pool: &mut SlotPool,
    tokens: &[u32],
    round: usize,
) -> Result<()> {
    let mut control = loaded.sequence(128)?;
    let active: Vec<_> = (0..pool.width())
        .filter(|&s| match round {
            0 => true,
            1 => s % 2 == 0,
            _ => s + 1 == pool.width(),
        })
        .collect();
    let position = prefix.position();
    for slot in 0..pool.width() {
        bind(loaded, prefix, pool, slot)?;
    }
    let token = |slot: usize, lane: usize| tokens[100 + slot * 3 + lane];
    let lanes: Vec<_> = active
        .iter()
        .flat_map(|&slot| {
            (0..3).map(move |lane| {
                (
                    slot,
                    lane,
                    token(slot, lane),
                    position + lane,
                    position + lane,
                )
            })
        })
        .collect();
    let rows = pool.run_verify(loaded.device(), &lanes)?;
    for &slot in &active {
        control.reset()?;
        seed_prefix(loaded, prefix, &mut control)?;
        for lane in 0..3 {
            let expected = control.step(
                token(slot, lane),
                position + lane,
                position + lane,
                None,
                true,
            )?;
            outputs(&vec![expected], &vec![rows[slot * 3 + lane].clone()]);
        }
    }
    let commits: Vec<_> = active
        .iter()
        .map(|&slot| (slot, (slot + round) % 3))
        .collect();
    pool.commit_verify_many(loaded.device(), &commits)?;
    for slot in 0..pool.width() {
        control.reset()?;
        seed_prefix(loaded, prefix, &mut control)?;
        if active.contains(&slot) {
            for lane in 0..=(slot + round) % 3 {
                control.step(
                    token(slot, lane),
                    position + lane,
                    position + lane,
                    None,
                    true,
                )?;
            }
        }
        assert_eq!(control.position(), pool.cursors()[slot]);
        live_states(loaded, &control, pool, slot)?;
    }
    Ok(())
}

#[test]
#[ignore = "CUDA/checkpoint/frozen tokens/safe-run; exact logits, partial commits and live state in actual speculation pools"]
fn checkpoint_verification_matches_single_step() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let model = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens = prompt()?;
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        model,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 64,
            verification_width: 3,
            mtp_depth: 2,
            autotune: false,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    let mut prefix = loaded.sequence_pooled(128)?;
    prefix.prefill_batch(&tokens[..51], 0, true)?;
    loaded.weights.prefill_width = 0;
    loaded.weights.narrow_prefill_width = 0;
    loaded.weights.batch_width = 3;
    for (slots, paged) in [(2, false), (4, true)] {
        let mut pool = loaded.slot_pool(slots, 128, paged.then_some(slots * 128))?;
        for round in 0..3 {
            replay(&loaded, &prefix, &mut pool, &tokens, round)?;
        }
        drop(pool);
        loaded.device().reclaim_barrier()?;
    }
    Ok(())
}

#[test]
#[ignore = "CUDA/checkpoint/frozen tokens/safe-run; private M1 versus M3 full target outputs and state"]
fn checkpoint_private_verification_matches_single_step() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let model = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens = prompt()?;
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        model,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 64,
            verification_width: 3,
            mtp_depth: 2,
            autotune: false,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    let mut prefix = loaded.sequence_pooled(256)?;
    for (index, part) in tokens[..127].chunks(64).enumerate() {
        prefix.prefill_batch(part, index * 64, index == 1)?;
    }
    loaded.weights.prefill_width = 0;
    loaded.weights.narrow_prefill_width = 0;
    let mut scalar = loaded.sequence(256)?;
    let mut verify = loaded.sequence(256)?;
    seed_prefix(&loaded, &prefix, &mut scalar)?;
    seed_prefix(&loaded, &prefix, &mut verify)?;
    let mut offset = 200;
    for count in [1, 2, 3] {
        let position = scalar.position();
        let inputs = &tokens[offset..offset + count];
        let rows = verify.step_batch(inputs, position)?;
        for (lane, &token) in inputs.iter().enumerate() {
            let expected = scalar.step(token, position + lane, position + lane, None, true)?;
            outputs(&vec![expected], &vec![rows[lane].clone()]);
        }
        private_committed_states(&loaded, &scalar, &verify)?;
        offset += count;
    }
    Ok(())
}

/// KV beyond the committed cursor is speculative scratch, not part of the observable state.
fn private_committed_states(
    loaded: &LoadedModel,
    control: &DeviceProgram,
    verify: &DeviceProgram,
) -> Result<()> {
    assert_eq!(control.position(), verify.position());
    assert_eq!(control.states().len(), verify.states().len());
    for (id, tensors) in control.states() {
        assert_eq!(tensors.len(), verify.states()[id].len());
        for (a, b) in tensors.iter().zip(&verify.states()[id]) {
            bits(
                &loaded.device().read_borrowed(a)?,
                &loaded.device().read_borrowed(b)?,
                "private committed F32 state",
            );
        }
    }
    assert!(!control.fp8_states().is_empty());
    assert_eq!(control.fp8_states().len(), verify.fp8_states().len());
    let mut discarded = 0;
    let mut live = 0;
    for (id, (keys, values)) in control.fp8_states() {
        let (vk, vv) = &verify.fp8_states()[id];
        for (a, b) in [(keys, vk), (values, vv)] {
            assert_eq!(a.shape(), b.shape());
            let capacity = usize::try_from(a.shape()[1]).unwrap();
            let dim = usize::try_from(a.shape()[2]).unwrap();
            let a = loaded.device().read_borrowed(a)?;
            let b = loaded.device().read_borrowed(b)?;
            for (index, (a, b)) in a.iter().zip(&b).enumerate() {
                let position = (index / dim) % capacity;
                if position < control.position() {
                    assert!(a.0 & 127 != 127 && b.0 & 127 != 127, "nonfinite live KV");
                    assert_eq!(a.0, b.0, "live private KV at position {position}");
                    live += 1;
                } else if a.0 != b.0 {
                    discarded += 1;
                }
            }
        }
    }
    println!(
        "{}",
        json!({"private_committed_state_exact":true,"position":control.position(),"live_FP8_bytes":live,"different_discarded_future_bytes":discarded})
    );
    Ok(())
}
