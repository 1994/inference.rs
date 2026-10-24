//! Ordinary serving is the verification oracle, including its logical-slot normalization.
use super::{
    LoadOptions, LoadedModel,
    quant_reuse_model_tests::{bits, prompt},
};
use crate::{
    constants::KV_BLOCK_TOKENS,
    device::CudaDevice,
    resident::{DeviceProgram, slot_batch::SlotPool},
};
use infer_core::{Error, ModelId, Result};
use std::env::var;
const CANDIDATES: usize = 3;
const fn token(tokens: &[u32], slot: usize, lane: usize) -> u32 {
    tokens[100 + slot * CANDIDATES + lane]
}

fn bind_all(loaded: &LoadedModel, prefix: &DeviceProgram, pool: &mut SlotPool) -> Result<()> {
    for slot in 0..pool.width() {
        pool.release(slot);
        assert!(pool.take_slot(slot));
        assert!(pool.assign(slot, 256, loaded.device())?);
        pool.bind(
            slot,
            prefix.states(),
            prefix.fp8_states(),
            prefix.position(),
            loaded.device(),
        )?;
    }
    Ok(())
}
fn live_states(
    loaded: &LoadedModel,
    reference: &SlotPool,
    target: &SlotPool,
    slot: usize,
) -> Result<()> {
    let (rs, rf) = reference.test_states(slot);
    assert!(!rs.is_empty() && !rf.is_empty());
    let (ts, tf) = target.test_states(slot);
    assert_eq!(rs.keys().collect::<Vec<_>>(), ts.keys().collect::<Vec<_>>());
    for (id, tensors) in rs {
        assert_eq!(tensors.len(), ts[id].len());
        for (a, b) in tensors.iter().zip(&ts[id]) {
            bits(
                &loaded.device().read_borrowed(a)?,
                &loaded.device().read_borrowed(b)?,
                "ordinary/verify recurrent state",
            );
        }
    }
    let rt = reference.test_table(loaded.device(), slot)?;
    let tt = target.test_table(loaded.device(), slot)?;
    assert_eq!(rf.keys().collect::<Vec<_>>(), tf.keys().collect::<Vec<_>>());
    let position = reference.cursors()[slot];
    assert_eq!(position, target.cursors()[slot]);
    for (id, (rk, rv)) in rf {
        let (tk, tv) = &tf[id];
        for (a, b) in [(rk, tk), (rv, tv)] {
            let heads = usize::try_from(a.shape()[0]).unwrap();
            let rd = usize::try_from(a.shape()[1]).unwrap();
            let td = usize::try_from(b.shape()[1]).unwrap();
            let dim = usize::try_from(a.shape()[2]).unwrap();
            let a = loaded.device().read_borrowed(a)?;
            let b = loaded.device().read_borrowed(b)?;
            for head in 0..heads {
                for pos in 0..position {
                    let row = |table: &[i32]| {
                        usize::try_from(table[pos / KV_BLOCK_TOKENS]).unwrap() * KV_BLOCK_TOKENS
                            + pos % KV_BLOCK_TOKENS
                    };
                    let left = (head * rd + row(&rt)) * dim;
                    let right = (head * td + row(&tt)) * dim;
                    assert!(
                        a[left..left + dim]
                            .iter()
                            .zip(&b[right..right + dim])
                            .all(|(a, b)| a.0 == b.0),
                        "ordinary/verify live KV"
                    );
                }
            }
        }
    }
    Ok(())
}
fn compare_scores(
    loaded: &LoadedModel,
    reference: &mut SlotPool,
    target: &mut SlotPool,
    active: &[usize],
    tokens: &[u32],
    position: usize,
) -> Result<()> {
    let lanes: Vec<_> = active
        .iter()
        .flat_map(|&slot| {
            (0..CANDIDATES).map(move |lane| {
                (
                    slot,
                    lane,
                    token(tokens, slot, lane),
                    position + lane,
                    position + lane,
                )
            })
        })
        .collect();
    let rows = target.run_verify(loaded.device(), &lanes)?;
    assert_eq!(rows.len(), target.width() * CANDIDATES);
    // Scoring may append future KV, but committed prefixes and recurrent states stay untouched.
    for slot in 0..target.width() {
        live_states(loaded, reference, target, slot)?;
    }
    for lane in 0..CANDIDATES {
        let lanes: Vec<_> = active
            .iter()
            .map(|&slot| {
                (
                    slot,
                    token(tokens, slot, lane),
                    position + lane,
                    position + lane,
                )
            })
            .collect();
        let expected = reference.run_lanes(loaded.device(), &lanes)?;
        assert_eq!(expected.len(), active.len());
        for (index, &slot) in active.iter().enumerate() {
            bits(
                &reference.test_hidden(loaded.device(), slot)?,
                &rows[slot * CANDIDATES + lane].0,
                "ordinary/verify hidden",
            );
            bits(
                &expected[index].1,
                &rows[slot * CANDIDATES + lane].1,
                "ordinary/verify full logits",
            );
        }
    }
    Ok(())
}
fn replay(
    loaded: &LoadedModel,
    prefix: &DeviceProgram,
    reference: &mut SlotPool,
    target: &mut SlotPool,
    tokens: &[u32],
    round: usize,
) -> Result<()> {
    bind_all(loaded, prefix, reference)?;
    bind_all(loaded, prefix, target)?;
    let active: Vec<_> = (0..target.width())
        .filter(|&slot| match round {
            0 => true,
            1 => slot % 2 == 0,
            _ => slot + 1 == target.width(),
        })
        .collect();
    let position = prefix.position();
    compare_scores(loaded, reference, target, &active, tokens, position)?;
    let commits: Vec<_> = active
        .iter()
        .map(|&slot| (slot, (slot + round) % CANDIDATES))
        .collect();
    target.commit_verify_many(loaded.device(), &commits)?;
    bind_all(loaded, prefix, reference)?;
    for lane in 0..CANDIDATES {
        let lanes: Vec<_> = active
            .iter()
            .filter(|&&slot| (slot + round) % CANDIDATES >= lane)
            .map(|&slot| {
                (
                    slot,
                    token(tokens, slot, lane),
                    position + lane,
                    position + lane,
                )
            })
            .collect();
        if !lanes.is_empty() {
            reference.run_lanes(loaded.device(), &lanes)?;
        }
    }
    for slot in 0..target.width() {
        live_states(loaded, reference, target, slot)?;
    }
    Ok(())
}
#[test]
#[ignore = "CUDA/checkpoint/frozen tokens/safe-run; ordinary-pool versus speculative-pool full target arithmetic"]
fn checkpoint_verification_matches_ordinary_pool() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let model = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens = prompt()?;
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        model,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 64,
            verification_width: CANDIDATES,
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
    for (reference_width, verify_width) in [(6, 6), (12, 12), (16, 12), (16, 16)] {
        loaded.weights.batch_width = 0;
        let mut reference = SlotPool::new(
            loaded.device(),
            loaded.graph(),
            &loaded.weights,
            reference_width,
            8192,
            loaded.model.hidden_size,
            (loaded.model.vocab_size, 0),
            Some(8704),
        )?;
        loaded.weights.batch_width = CANDIDATES;
        let mut target = loaded.slot_pool(verify_width, 8192, Some(8448))?;
        for round in 0..CANDIDATES {
            replay(
                &loaded,
                &prefix,
                &mut reference,
                &mut target,
                &tokens,
                round,
            )?;
        }
        drop((reference, target));
        loaded.device().reclaim_barrier()?;
    }
    Ok(())
}

#[test]
#[ignore = "CUDA/checkpoint/frozen tokens/safe-run; ordinary-pool versus speculative-pool full target arithmetic"]
fn checkpoint_bf16_verification_keeps_funded_pool_target() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let model = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens = prompt()?;
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        model,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 64,
            verification_width: CANDIDATES,
            mtp_depth: 2,
            autotune: false,
            vision: false,
            attention_profile: crate::AttentionProfile::Bf16,
            ..LoadOptions::default()
        },
    )?;
    let mut prefix = loaded.sequence_pooled(256)?;
    for (index, part) in tokens[..127].chunks(64).enumerate() {
        prefix.prefill_batch(part, index * 64, index == 1)?;
    }
    loaded.weights.prefill_width = 0;
    loaded.weights.narrow_prefill_width = 0;
    for (reference_width, verify_width, profile) in [
        (7, 7, crate::AttentionProfile::Bf16),
        (16, 7, crate::AttentionProfile::Canonical),
        (10, 10, crate::AttentionProfile::Bf16),
        (16, 10, crate::AttentionProfile::Canonical),
    ] {
        loaded.weights.batch_width = 0;
        loaded.weights.attention_profile = profile;
        let mut reference = SlotPool::new(
            loaded.device(),
            loaded.graph(),
            &loaded.weights,
            reference_width,
            8192,
            loaded.model.hidden_size,
            (loaded.model.vocab_size, 0),
            Some(8704),
        )?;
        loaded.weights.batch_width = CANDIDATES;
        loaded.weights.attention_profile = crate::AttentionProfile::Bf16;
        let mut target = loaded.slot_pool(verify_width, 8192, Some(8448))?;
        for round in 0..CANDIDATES {
            replay(
                &loaded,
                &prefix,
                &mut reference,
                &mut target,
                &tokens,
                round,
            )?;
        }
        drop((reference, target));
        loaded.device().reclaim_barrier()?;
    }
    Ok(())
}

fn teacher_forced_prefix() -> Result<(LoadedModel, DeviceProgram, Vec<u32>)> {
    use super::quant_reuse_model_tests::{outputs, program_states};
    use crate::AttentionProfile;
    use std::fs::read;
    CudaDevice::enable_kernel_cache()?;
    let model = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let path = var("INFER_TEST_REGRESSION").map_err(|e| Error::invalid(e.to_string()))?;
    let fixture: serde_json::Value = serde_json::from_slice(&read(path).unwrap()).unwrap();
    let prompt: Vec<u32> = serde_json::from_value(fixture["prompt"].clone()).unwrap();
    let decode: Vec<u32> = serde_json::from_value(fixture["decode"].clone()).unwrap();
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        model,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 1024,
            verification_width: 3,
            mtp_depth: 2,
            autotune: false,
            attention_profile: AttentionProfile::Bf16,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    let mut fast_prefix = loaded.sequence(2048)?;
    loaded.weights.batch_width = 0;
    loaded.weights.attention_profile = AttentionProfile::Canonical;
    let mut control_prefix = loaded.sequence(2048)?;
    let f = fast_prefix.prefill_batch(&prompt, 0, true)?;
    let c = control_prefix.prefill_batch(&prompt, 0, true)?;
    outputs(&f, &c);
    program_states(&loaded, &fast_prefix, &control_prefix)?;
    println!("ACTUAL DEFAULT prefix bits exact: prompt={}", prompt.len());
    Ok((loaded, control_prefix, decode))
}

#[test]
#[ignore = "CUDA/checkpoint/actual regression580/safe-run; diagnostic only"]
fn checkpoint_actual_default_teacher_forced_target() -> Result<()> {
    use crate::AttentionProfile;
    let (mut loaded, control_prefix, decode) = teacher_forced_prefix()?;
    loaded.weights.prefill_width = 0;
    loaded.weights.narrow_prefill_width = 0;
    let mut reference = SlotPool::new(
        loaded.device(),
        loaded.graph(),
        &loaded.weights,
        16,
        8192,
        loaded.model.hidden_size,
        (loaded.model.vocab_size, 0),
        Some(8704),
    )?;
    loaded.weights.batch_width = 3;
    loaded.weights.attention_profile = AttentionProfile::Bf16;
    let mut target = loaded.slot_pool(10, 8192, Some(8448))?;
    for pool in [&mut reference, &mut target] {
        for slot in 0..pool.width() {
            assert!(pool.take_slot(slot));
            assert!(pool.assign(slot, 512, loaded.device())?);
            pool.bind(
                slot,
                control_prefix.states(),
                control_prefix.fp8_states(),
                control_prefix.position(),
                loaded.device(),
            )?;
        }
    }
    for (round, input) in decode.as_chunks::<3>().0.iter().enumerate() {
        let pos = control_prefix.position() + round * 3;
        let lanes: Vec<_> = (0..10)
            .flat_map(|slot| {
                (0..3).map(move |lane| (slot, lane, input[lane], pos + lane, pos + lane))
            })
            .collect();
        let selected = target.run_verify_readout(loaded.device(), &lanes, true)?;
        let mut actual = selected.rows;
        let logits = target.test_verify_logit_rows(loaded.device())?;
        for (row, logits) in actual.iter_mut().zip(logits) {
            row.1 = logits;
        }
        let selected = selected
            .tokens
            .expect("actual default device readout must execute");
        for row in 0..30 {
            let expected = actual[row]
                .1
                .iter()
                .copied()
                .enumerate()
                .max_by(|(ai, a), (bi, b)| a.total_cmp(b).then_with(|| bi.cmp(ai)))
                .unwrap()
                .0;
            let expected = u32::try_from(expected).unwrap();
            assert_eq!(
                selected[row],
                Some(expected),
                "actual regression greedy round={round} row={row}"
            );
        }
        for lane in 0..3 {
            let ordinary: Vec<_> = (0..10)
                .map(|slot| (slot, input[lane], pos + lane, pos + lane))
                .collect();
            let expected = reference.run_lanes(loaded.device(), &ordinary)?;
            for slot in 0..10 {
                bits(
                    &reference.test_hidden(loaded.device(), slot)?,
                    &actual[slot * 3 + lane].0,
                    "actual default canonical16/BF16MTP10 hidden",
                );
                bits(
                    &expected[slot].1,
                    &actual[slot * 3 + lane].1,
                    "actual default canonical16/BF16MTP10 logits",
                );
            }
        }
        let commits: Vec<_> = (0..10).map(|slot| (slot, 2)).collect();
        target.commit_verify_many(loaded.device(), &commits)?;
        for slot in 0..10 {
            live_states(&loaded, &reference, &target, slot)?;
        }
        println!(
            "actual default target exact through decode {}",
            (round + 1) * 3
        );
    }
    Ok(())
}
