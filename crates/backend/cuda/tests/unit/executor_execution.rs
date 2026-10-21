use super::pooled_envelope;
use std::{env::var, error::Error};

#[test]
fn lazy_pool_preserves_live_private_reservations_and_scratch() {
    // 40 reserved by live sequences and the shared scratch is still owned by them.
    let lazy = pooled_envelope(1_000, 100, 40, 20);
    assert_eq!(lazy, 60);
    assert_eq!(lazy + 40 + 20, 120);
    // At load time that same scratch reservation can instead fund the worker.
    assert_eq!(pooled_envelope(1_000, 100, 0, 20), 120);
}

#[test]
fn pool_envelope_is_capped_by_physical_storage_and_saturates() {
    assert_eq!(pooled_envelope(50, 100, 0, 20), 50);
    assert_eq!(pooled_envelope(50, 100, 40, 20), 50);
    assert_eq!(pooled_envelope(1_000, 100, 101, 20), 0);
    assert_eq!(pooled_envelope(u64::MAX, u64::MAX, 0, 20), u64::MAX);
}

#[test]
#[ignore = "real MTP2 checkpoint and CUDA; run inside safe-run"]
fn lazy_pool_retries_after_a_live_private_reservation_is_released() -> Result<(), Box<dyn Error>> {
    use crate::{
        device::CudaDevice,
        executor::CudaBackend,
        loading::{LoadOptions, LoadedModel},
    };
    use infer_core::{ModelId, StateId};
    use infer_ir::OutputReadout;

    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        var("INFER_TEST_MODEL")?,
        ModelId::new(1)?,
        LoadOptions {
            prefill_width: 256,
            mtp_depth: 2,
            autotune: false,
            ..LoadOptions::default()
        },
    )?;
    assert!(loaded.model().max_sequence >= 32768);
    let budget = loaded.device().allocatable_memory_bytes()?;
    // Start without a pool, then admit one long and two short private programs.
    let mut backend = CudaBackend::new(loaded, budget, 1)?;
    backend.maximum_states = 16;
    let long = StateId::new(1)?;
    backend.reserve(long, 32768, OutputReadout::Logits)?;
    backend.reserve(StateId::new(2)?, 512, OutputReadout::Logits)?;
    backend.reserve(StateId::new(3)?, 512, OutputReadout::Logits)?;
    let mut lower = 0;
    let mut upper = budget;
    assert!(backend.pooled_plan(512, 16, upper)?.is_some());
    while lower < upper {
        let middle = lower + (upper - lower) / 2;
        if backend.pooled_plan(512, 16, middle)?.is_some() {
            upper = middle;
        } else {
            lower = middle + 1;
        }
    }
    // Existing reservations are legal; the additional pool is one byte short.
    let active = backend.active_budget()?;
    backend.budget = active.checked_add(lower - 1).ok_or("budget overflow")?;
    backend.ensure_slot_pool(&[true, true]);
    assert!(backend.slots.is_none());
    assert!(!backend.slots_disabled);
    assert_eq!(backend.deferred_pool_plan, Some((512, active)));
    backend.ensure_slot_pool(&[true, true]);
    assert_eq!(backend.deferred_pool_plan, Some((512, active)));

    backend.release(long)?;
    while backend.pool.evict() {}
    backend.ensure_slot_pool(&[true, true]);
    assert!(backend.slots.is_some(), "{:?}", backend.pool_failure_reason);
    assert!(backend.worker.is_some());
    assert!(!backend.slots_disabled);
    assert!(backend.deferred_pool_plan.is_none());
    assert!(backend.homes.iter().all(Option::is_some));
    Ok(())
}
