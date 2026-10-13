use super::*;
fn shape() -> Shape {
    Shape {
        heads: 24,
        kv_heads: 4,
        dim: 256,
        lanes: 64,
        window: 0,
    }
}
fn entry() -> Entry {
    let trial = |query_tile| Trial {
        choice: Choice::Tiled { query_tile },
        candidate_first: false,
        max_f64_error: 1e-7,
        cold: Times {
            baseline_ms: vec![2.0; MIN_REPS],
            candidate_ms: vec![1.0; MIN_REPS],
        },
        warm: Times {
            baseline_ms: vec![2.0; MIN_REPS],
            candidate_ms: vec![1.0; MIN_REPS],
        },
    };
    let search = vec![trial(16), trial(32)];
    let mut confirmation = search[0].clone();
    confirmation.candidate_first = true;
    Entry {
        key: Key {
            hardware: "test-device".into(),
            driver_api: 13000,
            driver_build: Some("test-driver".into()),
            fingerprint: "test-kernel".into(),
            schema: SCHEMA.into(),
            shape: shape(),
        },
        selected: confirmation.choice,
        search,
        confirmation,
        fixtures: reference::FIXTURES,
    }
}
#[test]
fn cache_requires_numerical_proof_all_candidates_and_reverse_confirmation() {
    let good = entry();
    assert!(good.valid());
    for variant in 0..12 {
        let mut bad = good.clone();
        match variant {
            0 => bad.key.schema = "stale-bound".into(),
            1 => bad.key.fingerprint.clear(),
            2 => bad.key.hardware.clear(),
            3 => bad.fixtures -= 1,
            4 => bad.search.pop().map_or((), |_| ()),
            5 => bad.search[1] = bad.search[0].clone(),
            6 => bad.confirmation.candidate_first = false,
            7 => bad.confirmation.max_f64_error = MAX_ERROR,
            8 => bad.search[0].max_f64_error = f64::NAN,
            9 => bad.confirmation.cold.candidate_ms = vec![2.0; MIN_REPS],
            10 => bad.confirmation.warm.baseline_ms.pop().map_or((), |_| ()),
            11 => bad.selected = Choice::Tiled { query_tile: 64 },
            _ => unreachable!(),
        }
        assert!(!bad.valid(), "variant {variant}");
    }
}
#[test]
fn fallback_can_have_complete_evidence_without_a_speedup() {
    let mut value = entry();
    value.selected = Choice::Simt;
    value.confirmation.choice = Choice::Simt;
    value.confirmation.cold.candidate_ms = value.confirmation.cold.baseline_ms.clone();
    value.confirmation.warm.candidate_ms = value.confirmation.warm.baseline_ms.clone();
    assert!(value.valid());
    value.search.clear();
    assert!(!value.valid());
}
#[test]
fn unsupported_geometry_never_selects_a_tensor_core_configuration() {
    assert!(shape().supported());
    for s in [
        Shape {
            heads: 0,
            ..shape()
        },
        Shape {
            kv_heads: 0,
            ..shape()
        },
        Shape {
            heads: 23,
            ..shape()
        },
        Shape {
            dim: 128,
            ..shape()
        },
        Shape {
            lanes: 51,
            ..shape()
        },
        Shape {
            window: usize::MAX,
            ..shape()
        },
    ] {
        assert!(!s.supported());
    }
}
#[test]
fn independent_reference_preserves_inactive_rows_and_quantized_append() -> Result<()> {
    let f = Fixture::new(shape(), 2)?;
    assert!(f.expected.iter().all(|v| *v == 0.0));
    assert!(f.keys.iter().zip(&f.expected_keys).all(|(a, b)| a.0 == b.0));
    assert!(
        f.values
            .iter()
            .zip(&f.expected_values)
            .all(|(a, b)| a.0 == b.0)
    );
    let f = Fixture::new(shape(), 0)?;
    let inactive = (f.shape.lanes - 1) * f.shape.heads * f.shape.dim;
    assert!(f.expected[inactive..].iter().all(|v| *v == 0.0));
    assert!(f.expected[..inactive].iter().all(|v| v.is_finite()));
    Ok(())
}
#[test]
fn attention_cache_round_trip_preserves_hardware_scope_and_old_tables() -> Result<()> {
    let table = super::super::TilingTable {
        resident_attention: vec![entry()],
        ..Default::default()
    };
    let json = serde_json::to_string(&table).map_err(device_error)?;
    let decoded: super::super::TilingTable = serde_json::from_str(&json).map_err(device_error)?;
    assert_eq!(decoded.resident_attention.len(), 1);
    assert!(decoded.resident_attention[0].valid());
    let legacy: super::super::TilingTable = serde_json::from_str("{}").map_err(device_error)?;
    assert!(legacy.resident_attention.is_empty());
    Ok(())
}
#[test]
#[ignore = "requires CUDA; full F64/cache numerical gate and hardware AB/BA tuning"]
fn hardware_query_sharing_search_and_cached_reuse() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let session = ResidentTuningSession::default();
    for lanes in [64, 256] {
        let shape = Shape { lanes, ..shape() };
        let selected = select(&device, shape, true, &session);
        let entry = memo()
            .lock()
            .map_err(|_| Error::invariant("test attention memo poisoned"))?
            .values()
            .find(|e| e.key.shape == shape)
            .cloned()
            .ok_or_else(|| Error::invalid("attention hardware proof did not complete"))?;
        assert!(entry.valid());
        assert_eq!(entry.selected, selected);
        let start = std::time::Instant::now();
        for _ in 0..16 {
            assert_eq!(select(&device, shape, false, &session), selected);
        }
        eprintln!(
            "ATTENTION_CACHED_16_MS {}",
            start.elapsed().as_secs_f64() * 1000.0
        );
        eprintln!(
            "ATTENTION_TUNING {}",
            serde_json::to_string(&entry).map_err(device_error)?
        );
    }
    Ok(())
}
