use super::*;
use std::time::Instant;
fn entry() -> Entry {
    let times = Times {
        baseline_ms: vec![2.0; MIN_REPS],
        candidate_ms: vec![1.0; MIN_REPS],
    };
    let search = Trial {
        value_block: 64,
        candidate_first: false,
        max_f64_error: 1e-7,
        cold: times.clone(),
        warm: times,
    };
    let mut confirmation = search.clone();
    confirmation.candidate_first = true;
    Entry {
        key: Key {
            hardware: "test-device".into(),
            driver_version: 13000,
            driver_build: Some("test-driver".into()),
            kernel_fingerprint: "test-ir".into(),
            schema: SCHEMA.into(),
            shape: [16, 48, 128, 64],
        },
        selected: 64,
        search: vec![search],
        confirmation,
    }
}
#[test]
fn rejected_cache_does_not_encode_a_valid_hardware_decision() {
    let good = entry();
    assert!(good.valid());
    let mut bad = good.clone();
    bad.confirmation.candidate_first = false;
    assert!(!bad.valid());
    let mut bad = good.clone();
    bad.search.clear();
    assert!(!bad.valid());
    let mut bad = good.clone();
    bad.confirmation.max_f64_error = 2e-5;
    assert!(!bad.valid());
    let mut bad = good.clone();
    bad.confirmation.cold.candidate_ms[0] = f32::NAN;
    assert!(!bad.valid());
    let mut bad = good.clone();
    bad.confirmation.cold.baseline_ms.pop();
    assert!(!bad.valid());
    let mut bad = good.clone();
    bad.selected = 32;
    assert!(!bad.valid());
    let mut bad = good.clone();
    bad.selected = 128;
    bad.confirmation.value_block = 128;
    bad.search.clear();
    assert!(!bad.valid());
    let mut bad = good.clone();
    bad.search.push(bad.search[0].clone());
    assert!(!bad.valid());
    let mut bad = good;
    bad.key.schema = "old-neumann".into();
    assert!(!bad.valid());
}
#[test]
fn supports_algorithm_geometry_without_device_names() {
    assert!(supported([2, 4, 32, 32]));
    assert!(supported([16, 48, 128, 256]));
    assert!(!supported([0, 48, 128, 64]));
    assert!(!supported([16, 47, 128, 64]));
    assert!(!supported([16, 48, 128, 3]));
    assert!(!supported([16, 48, 64, 64]));
}
#[test]
fn numerical_gate_rejects_nonfinite_or_partial_results() {
    assert!(reference::maximum_error(&[1.0], &[1.0]).is_ok());
    assert!(reference::maximum_error(&[1.0], &[1.0, 1.0]).is_err());
    assert!(reference::maximum_error(&[f32::NAN], &[1.0]).is_err());
    assert!(reference::maximum_error(&[1.0], &[1.1]).is_err());
}
#[test]
#[ignore = "requires CUDA hardware; finite hardware tuning with independent F64 references"]
fn hardware_search_passes_independent_reference_and_reverse_confirmation() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let shape = [16, 48, 128, 64];
    let selected = select(&device, shape, true, &ResidentTuningSession::default())
        .ok_or_else(|| Error::invalid("hardware selection unexpectedly rejected"))?;
    let entry = memo()
        .lock()
        .map_err(|_| Error::invariant("test tuning cache poisoned"))?
        .values()
        .find(|x| x.key.shape == shape)
        .cloned()
        .ok_or_else(|| Error::invalid("hardware search did not complete evidence"))?;
    assert_eq!(entry.selected, selected);
    assert!(entry.valid());
    let started = Instant::now();
    for _ in 0..48 {
        assert_eq!(
            select(&device, shape, false, &ResidentTuningSession::default()),
            Some(selected)
        );
    }
    eprintln!(
        "RECURRENT_IDENTITY_48_CACHED_MS {}",
        started.elapsed().as_secs_f64() * 1000.0
    );
    eprintln!(
        "RECURRENT_TUNING {}",
        serde_json::to_string(&entry).map_err(device_error)?
    );
    Ok(())
}
