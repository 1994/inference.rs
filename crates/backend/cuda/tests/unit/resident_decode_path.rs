//! Full checkpoint census of private/pooled arithmetic with a common forced trajectory.
//! Original GEMV is a diagnostic control; the experimental private lowering must match pool.
use super::{LoadOptions, LoadedModel};
use crate::{
    device::CudaDevice,
    resident::{
        DeviceProgram, pool_f32_capture_records, slot_batch::SlotPool, with_pool_f32,
        with_precise_dense,
    },
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
fn state_census(
    loaded: &LoadedModel,
    program: &DeviceProgram,
    pool: &SlotPool,
) -> Result<(usize, Vec<Value>)> {
    state_census_slot(loaded, program, pool, 0)
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

fn common_prefix(
    loaded: &LoadedModel,
    original: &mut DeviceProgram,
    aligned: &mut DeviceProgram,
    tokens: &[u32],
) -> Result<Vec<f32>> {
    let control = original.prefill_batch(&tokens[..179], 0, true)?;
    let precise = aligned.prefill_batch(&tokens[..179], 0, true)?;
    assert_eq!(control.len(), precise.len());
    let mut changed = 0;
    for (control, precise) in control.iter().zip(&precise) {
        changed += differences(&agreement(
            &control.0,
            &precise.0,
            "prefill arithmetic change",
        ));
    }
    seed_prefix(loaded, aligned, original)?;
    println!(
        "{}",
        json!({"private_pool_common_prefix":true,"original_prefill_hidden_differences":changed,"control_state_seeded_from_aligned":true,"F32_state":true,"FP8_KV":true})
    );
    Ok(precise
        .last()
        .ok_or_else(|| Error::invariant("prompt logits"))?
        .1
        .clone())
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
#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL, frozen INFER_TEST_TOKENS/table; full private/pool arithmetic diagnostic, safe-run"]
fn checkpoint_private_pool_first_step_arithmetic_census() -> Result<()> {
    let path = std::env::var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens_path =
        std::env::var("INFER_TEST_TOKENS").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens: Vec<u32> = serde_json::from_slice(
        &std::fs::read(tokens_path).map_err(|e| Error::invalid(e.to_string()))?,
    )
    .map_err(|e| Error::invalid(e.to_string()))?;
    assert_eq!(tokens.len(), 511);
    let cache_path =
        crate::tuning::table_path().ok_or_else(|| Error::invalid("frozen table required"))?;
    let cache = std::fs::read(&cache_path).map_err(|e| Error::invalid(e.to_string()))?;
    CudaDevice::enable_kernel_cache()?;
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::new(1)?,
        LoadOptions {
            prefill_width: 256,
            autotune: false,
            chunked_recurrent: true,
            mtp_depth: 0,
            ..LoadOptions::default()
        },
    )?;
    loaded.weights.narrow_prefill_width = 0;
    let mut original = with_precise_dense(false, || loaded.sequence(1024))?;
    let mut aligned = with_precise_dense(true, || loaded.sequence(1024))?;
    let mut pool = with_precise_dense(true, || loaded.slot_pool(4, 1024))?;
    let logits = common_prefix(&loaded, &mut original, &mut aligned, &tokens)?;
    let mut token = greedy(&logits)?;
    assert_eq!(pool.take_free(), Some(0));
    pool.bind(
        0,
        aligned.states(),
        aligned.fp8_states(),
        aligned.position(),
        loaded.device(),
    )?;
    let (initial, details) = state_census(&loaded, &aligned, &pool)?;
    assert_eq!(initial, 0, "initial state {details:?}");
    let (control_initial, details) = state_census(&loaded, &original, &pool)?;
    assert_eq!(control_initial, 0, "initial control state {details:?}");
    let mut aligned_differences = 0;
    let mut control_differences = 0;
    for step in 0..8 {
        let position = aligned.position();
        assert_eq!(original.position(), position);
        let original_output = original.step(token, position, position, None, true)?;
        let output = aligned.step(token, position, position, None, true)?;
        let pooled = pool.run_lanes(loaded.device(), &[(0, token, position, position)])?;
        assert_eq!(pooled.len(), 1);
        let reference = &pooled[0];
        let pooled_hidden = pool.test_hidden(loaded.device(), 0)?;
        let hidden = agreement(&output.0, &pooled_hidden, "aligned hidden");
        let logits = agreement(&output.1, &reference.1, "aligned full vocabulary");
        let old_hidden = agreement(&original_output.0, &pooled_hidden, "original GEMV hidden");
        let old_logits = agreement(
            &original_output.1,
            &reference.1,
            "original GEMV full vocabulary",
        );
        aligned_differences += differences(&hidden) + differences(&logits);
        control_differences += differences(&old_hidden) + differences(&old_logits);
        let (state, state_details) = if step == 0 || step == 7 {
            state_census(&loaded, &aligned, &pool)?
        } else {
            (0, Vec::new())
        };
        aligned_differences += state;
        let (old_state, old_state_details) = if step == 0 {
            state_census(&loaded, &original, &pool)?
        } else {
            (0, Vec::new())
        };
        println!(
            "{}",
            json!({"private_pool_arithmetic_step":step,"position":position,"input_token":token,"aligned_hidden":hidden,"aligned_logits":logits,"original_hidden":old_hidden,"original_logits":old_logits,"aligned_state_differences":state,"aligned_state_details":state_details,"original_state_differences":old_state,"original_state_details":old_state_details,"MTP":0,"pool_width":4,"capacity":1024})
        );
        token = greedy(&output.1)?;
    }
    println!(
        "{}",
        json!({"private_pool_arithmetic_summary":true,"aligned_differences":aligned_differences,"original_control_output_differences":control_differences,"F32_state":true,"FP8_KV_complete":true,"steps":8})
    );
    assert_eq!(
        cache,
        std::fs::read(cache_path).map_err(|e| Error::invalid(e.to_string()))?
    );
    assert_eq!(
        aligned_differences, 0,
        "private small dense lowering does not fully align pooled arithmetic"
    );
    Ok(())
}

struct PoolCheckpoint {
    controls: Vec<DeviceProgram>,
    pool: SlotPool,
    loaded: LoadedModel,
    next: Vec<u32>,
    vocabulary: u32,
    cache_path: std::path::PathBuf,
    cache: Vec<u8>,
}
fn pool_checkpoint() -> Result<PoolCheckpoint> {
    let path = std::env::var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens_path =
        std::env::var("INFER_TEST_TOKENS").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens: Vec<u32> = serde_json::from_slice(
        &std::fs::read(tokens_path).map_err(|e| Error::invalid(e.to_string()))?,
    )
    .map_err(|e| Error::invalid(e.to_string()))?;
    assert_eq!(tokens.len(), 511);
    let cache_path =
        crate::tuning::table_path().ok_or_else(|| Error::invalid("frozen table required"))?;
    let cache = std::fs::read(&cache_path).map_err(|e| Error::invalid(e.to_string()))?;
    CudaDevice::enable_kernel_cache()?;
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::new(1)?,
        LoadOptions {
            prefill_width: 256,
            autotune: false,
            chunked_recurrent: true,
            mtp_depth: 0,
            ..LoadOptions::default()
        },
    )?;
    loaded.weights.narrow_prefill_width = 0;
    let mut first = with_pool_f32(false, || loaded.sequence_pooled(1024))?;
    let prompt = first.prefill_batch(&tokens[..179], 0, true)?;
    let logits = &prompt
        .last()
        .ok_or_else(|| Error::invariant("prompt output"))?
        .1;
    let vocabulary = u32::try_from(logits.len()).map_err(|e| Error::invalid(e.to_string()))?;
    let initial = greedy(logits)?;
    let mut controls = vec![first];
    // Other controls only replay private decode, so avoid unused wide prompt/verify captures.
    loaded.weights.prefill_width = 0;
    for _ in 1..4 {
        let mut program = with_pool_f32(false, || loaded.sequence_pooled(1024))?;
        seed_prefix(&loaded, &controls[0], &mut program)?;
        controls.push(program);
    }
    let before = pool_f32_capture_records();
    let mut pool = with_pool_f32(true, || loaded.slot_pool(4, 1024))?;
    let records = pool_f32_capture_records().saturating_sub(before);
    assert!(
        records >= 96,
        "F32 pool capture was not used for all real small projections: {records}"
    );
    println!(
        "{}",
        json!({"pool_four_capture_records":records,"expected_small_projections":96})
    );
    for (slot, program) in controls.iter().enumerate() {
        assert_eq!(pool.take_free(), Some(slot));
        pool.bind(
            slot,
            program.states(),
            program.fp8_states(),
            program.position(),
            loaded.device(),
        )?;
        let (count, details) = state_census_slot(&loaded, program, &pool, slot)?;
        assert_eq!(count, 0, "initial slot {slot}: {details:?}");
    }
    let next: Vec<u32> = (0..4)
        .map(|slot| (initial + u32::try_from(slot * 11).expect("four slots")) % vocabulary)
        .collect();
    Ok(PoolCheckpoint {
        controls,
        pool,
        loaded,
        next,
        vocabulary,
        cache_path,
        cache,
    })
}

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL/frozen tokens/table, safe-run; four distinct trajectories, inactive/rebind, complete private F32/pool equality"]
fn checkpoint_pool_four_slots_private_f32_census() -> Result<()> {
    let PoolCheckpoint {
        mut controls,
        mut pool,
        loaded,
        mut next,
        vocabulary,
        cache_path,
        cache,
    } = pool_checkpoint()?;
    let mut changed = 0;
    let mut total = 0;
    let patterns = [
        vec![0, 1, 2, 3],
        vec![0, 2],
        vec![1, 3],
        vec![3],
        vec![0, 1, 3],
        vec![0, 1, 2, 3],
        vec![1, 2],
        vec![0, 1, 2, 3],
    ];
    for (step, active) in patterns.iter().enumerate() {
        if step == 5 {
            pool.release(2);
            assert_eq!(pool.take_free(), Some(2));
            pool.bind(
                2,
                controls[2].states(),
                controls[2].fp8_states(),
                controls[2].position(),
                loaded.device(),
            )?;
        }
        let lanes: Vec<_> = active
            .iter()
            .map(|&slot| {
                (
                    slot,
                    next[slot],
                    controls[slot].position(),
                    controls[slot].position(),
                )
            })
            .collect();
        let private = lanes
            .iter()
            .map(|&(slot, token, position, _)| {
                controls[slot].step(token, position, position, None, true)
            })
            .collect::<Result<Vec<_>>>()?;
        let pooled = pool.run_lanes(loaded.device(), &lanes)?;
        assert_eq!(pooled.len(), active.len());
        for ((&slot, original), output) in active.iter().zip(&private).zip(&pooled) {
            let hidden = agreement(
                &original.0,
                &pool.test_hidden(loaded.device(), slot)?,
                "pool four hidden",
            );
            let logits = agreement(&original.1, &output.1, "pool four vocabulary");
            changed += differences(&hidden) + differences(&logits);
            total += original.0.len() + original.1.len();
            println!(
                "{}",
                json!({"pool_four_model_step":step,"slot":slot,"hidden":hidden,"logits":logits})
            );
            next[slot] =
                (greedy(&original.1)? + u32::try_from(slot * 3).expect("four slots")) % vocabulary;
        }
        if step == 0 || step + 1 == patterns.len() {
            for (slot, program) in controls.iter().enumerate() {
                let (count, details) = state_census_slot(&loaded, program, &pool, slot)?;
                changed += count;
                println!(
                    "{}",
                    json!({"pool_four_model_state":step,"slot":slot,"different":count,"details":details})
                );
            }
        }
    }
    println!(
        "{}",
        json!({"pool_four_model_summary":true,"steps":8,"output_elements":total,"different":changed,"F32_state":true,"FP8_KV_complete":true,"private_arithmetic_unchanged":true})
    );
    assert_eq!(
        cache,
        std::fs::read(cache_path).map_err(|e| Error::invalid(e.to_string()))?
    );
    assert_eq!(
        changed, 0,
        "pool four slots must match original private F32 computation"
    );
    Ok(())
}
