use super::*;
fn entry() -> Entry {
    let shape = Shape {
        heads: 24,
        kv_heads: 4,
        dim: 256,
        lanes: 256,
        window: 0,
    };
    let case = |history| Case {
        history,
        cold: Times {
            baseline_ms: vec![1.0; 10],
            candidate_ms: vec![0.5; 10],
        },
        warm: Times {
            baseline_ms: vec![1.0; 10],
            candidate_ms: vec![0.5; 10],
        },
    };
    let trial = |variant| Trial {
        variant,
        fixtures: reference::FIXTURES,
        elements: reference::FIXTURES * shape.lanes * shape.heads * shape.dim,
        max_f64_error: 0.002,
        forward: histories(shape, capacity(shape))
            .into_iter()
            .map(case)
            .collect(),
        reverse: histories(shape, capacity(shape))
            .into_iter()
            .map(case)
            .collect(),
    };
    Entry {
        key: Key {
            hardware: r#"{"architecture":"sm_100"}"#.into(),
            driver_api: 13040,
            driver_build: Some("test-driver".into()),
            fingerprint: "a".repeat(64),
            schema: SCHEMA.into(),
            shape,
        },
        architecture: "sm_100".into(),
        profile: catalog::PROFILE.into(),
        baseline: Choice::Simt,
        selected: Choice::Mma {
            variant: Variant::Sm100,
        },
        trials: vec![trial(Variant::Sm100), trial(Variant::Sm100Pair)],
    }
}
#[test]
fn cluster_selection_requires_complete_independent_profile_evidence() {
    assert!(entry().valid());
    let mut e = entry();
    e.trials.pop();
    assert!(!e.valid());
    let mut e = entry();
    e.trials[1].variant = Variant::Sm103Pair;
    assert!(!e.valid());
    let mut e = entry();
    e.trials[0].reverse.pop();
    assert!(!e.valid());
    let mut e = entry();
    e.trials[0].fixtures -= 1;
    assert!(!e.valid());
    let mut e = entry();
    e.trials[0].max_f64_error = f64::NAN;
    assert!(!e.valid());
    let mut e = entry();
    e.profile = "canonical-f32".into();
    assert!(!e.valid());
}
#[test]
fn fast_long_case_cannot_hide_a_short_regression() {
    let mut e = entry();
    e.trials[0].forward[0].cold.candidate_ms.fill(1.2);
    assert!(!e.valid());
    e.selected = Choice::Mma {
        variant: Variant::Sm100Pair,
    };
    assert!(e.valid());
}

#[test]
#[ignore = "requires CUDA and safe-run; AOT ABI, full BF16-profile F64 sanity and measured architecture/cluster selection"]
fn hardware_mma_aot_graph_and_tuning() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let shape = Shape {
        heads: 24,
        kv_heads: 4,
        dim: 256,
        lanes: 256,
        window: 0,
    };
    let session = ResidentTuningSession::default();
    let selected = select(&device, shape, true, &session, Choice::Simt);
    let entry = memo()
        .lock()
        .map_err(|_| Error::invariant("test MMA memo poisoned"))?
        .values()
        .find(|e| e.key.shape == shape)
        .cloned()
        .ok_or_else(|| Error::invalid("MMA numerical and cluster search did not complete"))?;
    assert!(entry.valid());
    assert_eq!(
        select(&device, shape, false, &session, Choice::Simt),
        selected
    );
    println!(
        "MMA_ATTENTION_TUNING {}",
        serde_json::to_string(&entry).map_err(crate::device::device_error)?
    );
    Ok(())
}
