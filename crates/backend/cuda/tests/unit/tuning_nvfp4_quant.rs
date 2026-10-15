//! Cache integrity and real automatic native quantization selection.
use super::*;
#[test]
#[ignore = "requires CUDA/safe-run and private original tuning table"]
fn quant_profile_preserves_all_other_entries() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let path = super::super::table_path()
        .ok_or_else(|| Error::invalid("private tuning table required"))?;
    let before: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).map_err(crate::device::device_error)?)
            .map_err(crate::device::device_error)?;
    let device = CudaDevice::new(0)?;
    let session = ResidentTuningSession::default();
    for (rows, columns) in [(64, 5120), (64, 17408), (256, 5120), (256, 17408)] {
        assert_eq!(select(&device, rows, columns, false, &session), BASELINE);
        let chosen = select(&device, rows, columns, true, &session);
        assert_ne!(chosen, BASELINE, "known measured prefill should improve");
        assert_eq!(select(&device, rows, columns, false, &session), chosen);
    }
    for rows in [1, 4] {
        assert_eq!(select(&device, rows, 5120, true, &session), BASELINE);
    }
    let after: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).map_err(crate::device::device_error)?)
            .map_err(crate::device::device_error)?;
    for (key, value) in before
        .as_object()
        .ok_or_else(|| Error::invariant("table object"))?
    {
        if key != "resident_nvfp4_quant" {
            assert_eq!(after[key], *value, "old field {key}");
        }
    }
    let table = super::super::TilingTable::load(&path)?;
    assert_eq!(table.resident_nvfp4_quant.len(), 4);
    for entry in table.resident_nvfp4_quant {
        assert!(entry.valid());
        let mut bad = entry.clone();
        bad.confirmation.verified_bytes -= 1;
        assert!(!bad.valid());
        let mut bad = entry.clone();
        bad.confirmation.candidate_first = false;
        assert!(!bad.valid());
        let mut bad = entry.clone();
        bad.search.pop();
        assert!(!bad.valid());
        let mut bad = entry.clone();
        bad.key.shape.columns += 1;
        assert!(!bad.valid());
        let mut bad = entry.clone();
        bad.confirmation.cold.candidate_ms[0] = f32::NAN;
        assert!(!bad.valid());
        println!(
            "{}",
            serde_json::json!({"quant_profile":true,"shape":entry.key.shape,
            "selected_occupancy":entry.selected,"cold_speedup":entry.confirmation.cold.speedup(),
            "warm_speedup":entry.confirmation.warm.speedup(),"complete_four_global_byte_proof":true,
            "other_tuning_fields_unchanged":true})
        );
    }
    Ok(())
}
#[test]
fn invalid_quant_geometry_and_incomplete_proof_rejected() {
    for (m, k) in [
        (0, 5120),
        (1, 5120),
        (4, 5120),
        (64, 0),
        (64, 5119),
        (usize::MAX, 5120),
    ] {
        assert!(Shape::new(m, k).is_none());
    }
    let shape = Shape::new(64, 5120).expect("fixture");
    assert_eq!(shape.verified_bytes(), 64 * 5120 / 16 * 9 * 4);
    let t = evidence::Trial {
        occupancy: 6,
        candidate_first: true,
        verified_bytes: shape.verified_bytes(),
        cold: evidence::Times {
            baseline_ms: vec![2.0; 10],
            candidate_ms: vec![1.0; 10],
        },
        warm: evidence::Times {
            baseline_ms: vec![2.0; 10],
            candidate_ms: vec![1.0; 10],
        },
    };
    assert!(t.confirmed());
    let mut bad = t.clone();
    bad.cold.baseline_ms.pop();
    assert!(!bad.confirmed());
    let mut bad = t.clone();
    bad.warm.candidate_ms[0] = 0.0;
    assert!(!bad.confirmed());
    let mut bad = t;
    bad.occupancy = 32;
    assert!(!bad.confirmed());
}
