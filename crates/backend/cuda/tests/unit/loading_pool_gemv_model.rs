use super::{LoadOptions, LoadedModel};
use crate::{
    device::CudaDevice,
    resident::{DeviceProgram, pool_gemv, slot_batch::SlotPool},
};
use infer_core::{Error, ModelId, Result};
use serde_json::{Value, json};

fn agreement(left: &[f32], right: &[f32], field: &str) -> Value {
    assert_eq!(left.len(), right.len(), "{field}");
    assert!(!left.is_empty(), "{field}");
    assert!(left.iter().chain(right).all(|x| x.is_finite()), "{field}");
    let mut different = 0usize;
    let mut maximum = 0.0_f64;
    let mut first = None;
    for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
        maximum = maximum.max((f64::from(left) - f64::from(right)).abs());
        if left.to_bits() != right.to_bits() {
            different += 1;
            first.get_or_insert_with(|| json!({"element":index,"left":left,"right":right,"left_bits":left.to_bits(),"right_bits":right.to_bits()}));
        }
    }
    json!({"field":field,"elements":left.len(),"different":different,"maximum_absolute":maximum,"first":first})
}
fn differences(value: &Value) -> usize {
    usize::try_from(value["different"].as_u64().expect("diagnostic count"))
        .expect("count fits usize")
}
fn state_census_slot(
    loaded: &LoadedModel,
    program: &DeviceProgram,
    pool: &SlotPool,
    slot: usize,
) -> Result<(usize, Vec<Value>)> {
    let (state, fp8) = pool.test_states(slot);
    assert_eq!(program.states().len(), state.len());
    let mut changed = 0;
    let mut details = Vec::new();
    for (id, tensors) in program.states() {
        let reference = &state[id];
        assert_eq!(tensors.len(), reference.len());
        let spec = loaded
            .graph
            .tensors
            .iter()
            .find(|s| s.id == *id)
            .ok_or_else(|| Error::invariant("state spec"))?;
        for (part, (left, right)) in tensors.iter().zip(reference).enumerate() {
            let count = agreement(
                &loaded.device().read_borrowed(left)?,
                &loaded.device().read_borrowed(right)?,
                &format!("{id:?} {:?} part={part}", spec.storage),
            );
            changed += differences(&count);
            if differences(&count) > 0 {
                details.push(count);
            }
        }
    }
    assert_eq!(program.fp8_states().len(), fp8.len());
    for (id, (keys, values)) in program.fp8_states() {
        let (rkeys, rvalues) = &fp8[id];
        for (field, left, right) in [("keys", keys, rkeys), ("values", values, rvalues)] {
            let a = loaded.device().read_borrowed(left)?;
            let b = loaded.device().read_borrowed(right)?;
            assert_eq!(a.len(), b.len());
            assert!(a.iter().chain(&b).all(|x| x.0 & 127 != 127));
            let different = a.iter().zip(&b).filter(|(a, b)| a.0 != b.0).count();
            changed += different;
            if different > 0 {
                details.push(json!({"field":format!("{id:?} {field}"),"elements":a.len(),"different":different}));
            }
        }
    }
    Ok((changed, details))
}
fn seed_prefix(
    loaded: &LoadedModel,
    aligned: &DeviceProgram,
    original: &mut DeviceProgram,
) -> Result<()> {
    // Seed every program with the same prefix before isolating decode arithmetic.
    // Copy into captured allocations without changing their addresses or F32/FP8 storage.
    for (id, source) in aligned.states() {
        let target = original
            .states_mut()
            .get_mut(id)
            .ok_or_else(|| Error::invariant("control state"))?;
        assert_eq!(target.len(), source.len());
        for (target, source) in target.iter_mut().zip(source) {
            assert_eq!(target.size(), source.size());
            loaded.device().copy_d2d(target, source, source.size())?;
        }
    }
    for (id, (keys, values)) in aligned.fp8_states() {
        let (target_keys, target_values) = original
            .fp8_states_mut()
            .get_mut(id)
            .ok_or_else(|| Error::invariant("control FP8 cache"))?;
        for (target, source) in [(target_keys, keys), (target_values, values)] {
            assert_eq!(target.size(), source.size());
            loaded.device().copy_d2d(target, source, source.size())?;
        }
    }
    loaded.device().drain()?;
    original.set_position(aligned.position());
    Ok(())
}

fn greedy(logits: &[f32]) -> Result<u32> {
    assert_ne!(logits.len(), 0);
    let mut best = 0;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    u32::try_from(best).map_err(|e| Error::invalid(e.to_string()))
}
struct Case {
    controls: Vec<DeviceProgram>,
    pool: SlotPool,
    next: Vec<u32>,
    vocab: u32,
}
fn make_case(loaded: &LoadedModel, prefix: &DeviceProgram, slots: usize) -> Result<Case> {
    let before = pool_gemv::captures();
    let mut pool = loaded.slot_pool(slots, 128)?;
    assert!(pool_gemv::captures() - before >= 96, "actual GEMV captures");
    let mut controls = Vec::new();
    for slot in 0..slots {
        let mut control = pool_gemv::with_pool_gemv(false, || loaded.sequence_pooled(128))?;
        seed_prefix(loaded, prefix, &mut control)?;
        assert_eq!(pool.take_free(), Some(slot));
        pool.bind(
            slot,
            control.states(),
            control.fp8_states(),
            control.position(),
            loaded.device(),
        )?;
        controls.push(control);
    }
    let vocab =
        u32::try_from(loaded.model.vocab_size).map_err(|e| Error::invalid(e.to_string()))?;
    let next = (0..slots)
        .map(|slot| u32::try_from(100 + slot * 11).expect("small slot") % vocab)
        .collect();
    Ok(Case {
        controls,
        pool,
        next,
        vocab,
    })
}
fn check_states(loaded: &LoadedModel, case: &Case, step: usize) -> Result<usize> {
    let mut changed = 0;
    for (slot, control) in case.controls.iter().enumerate() {
        let (count, details) = state_census_slot(loaded, control, &case.pool, slot)?;
        changed += count;
        println!(
            "{}",
            json!({"gemv_pool_state":step,"slot":slot,"different":count,"details":details})
        );
    }
    Ok(changed)
}
fn replay(loaded: &LoadedModel, case: &mut Case, slots: usize) -> Result<()> {
    let all: Vec<_> = (0..slots).collect();
    let even: Vec<_> = (0..slots).filter(|slot| slot % 2 == 0).collect();
    let patterns = [
        all.clone(),
        even,
        vec![slots - 1],
        vec![0],
        all.clone(),
        all.clone(),
        vec![0, slots - 1],
        all,
    ];
    let mut changed = 0;
    for (step, active) in patterns.iter().enumerate() {
        if step == 5 {
            let slot = slots - 1;
            case.pool.release(slot);
            assert_eq!(case.pool.take_free(), Some(slot));
            let control = &case.controls[slot];
            case.pool.bind(
                slot,
                control.states(),
                control.fp8_states(),
                control.position(),
                loaded.device(),
            )?;
        }
        let lanes: Vec<_> = active
            .iter()
            .map(|&slot| {
                let pos = case.controls[slot].position();
                (slot, case.next[slot], pos, pos)
            })
            .collect();
        let private = lanes
            .iter()
            .map(|&(slot, token, pos, _)| case.controls[slot].step(token, pos, pos, None, true))
            .collect::<Result<Vec<_>>>()?;
        let pooled = case.pool.run_lanes(loaded.device(), &lanes)?;
        assert_eq!(pooled.len(), active.len());
        for ((&slot, reference), output) in active.iter().zip(&private).zip(&pooled) {
            let hidden = agreement(
                &reference.0,
                &case.pool.test_hidden(loaded.device(), slot)?,
                "private/pool hidden",
            );
            let logits = agreement(&reference.1, &output.1, "private/pool full vocabulary");
            changed += differences(&hidden) + differences(&logits);
            println!(
                "{}",
                json!({"gemv_pool_step":step,"slot":slot,"hidden":hidden,"logits":logits})
            );
            case.next[slot] =
                (greedy(&reference.1)? + u32::try_from(slot * 3).expect("slot")) % case.vocab;
        }
        if step == 0 || step == 7 {
            changed += check_states(loaded, case, step)?;
        }
    }
    assert_eq!(changed, 0, "complete private/pool arithmetic");
    println!(
        "{}",
        json!({"gemv_pool_model_summary":true,"slots":slots,"steps":8,"different":0,
        "inactive_reactivate_rebind":true,"full_F32_FP8_states":true,"private_arithmetic_unchanged":true})
    );
    Ok(())
}
#[test]
#[ignore = "requires CUDA/checkpoint/frozen511tokens/safe-run; original private versus default pool full-model bits"]
fn checkpoint_pool_gemv_private_model_bits() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let path = std::env::var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens = super::quant_reuse_model_tests::prompt()?;
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 64,
            autotune: false,
            mtp_depth: 0,
            ..LoadOptions::default()
        },
    )?;
    let mut prefix = loaded.sequence(128)?;
    prefix.prefill_batch(&tokens[..51], 0, true)?;
    loaded.weights.prefill_width = 0;
    loaded.weights.narrow_prefill_width = 0;
    for slots in [2, 3, 4] {
        let mut case = make_case(&loaded, &prefix, slots)?;
        replay(&loaded, &mut case, slots)?;
        drop(case);
        loaded.device().reclaim_barrier()?;
    }
    Ok(())
}
