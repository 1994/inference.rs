use super::*;

fn entry(hardware: &str, tile: usize) -> Entry {
    let times = Times {
        baseline_ms: vec![2.0; MIN_REPS],
        candidate_ms: vec![1.0; MIN_REPS],
    };
    let trial = Trial {
        tile,
        candidate_first: false,
        cold: times.clone(),
        warm: times,
    };
    Entry {
        key: Key {
            hardware: hardware.into(),
            driver_version: 13020,
            driver_build: Some("driver-build".into()),
            kernel_fingerprint: "kernel-A".into(),
            rows: 1,
            vocabulary: 248_320,
        },
        selected: tile,
        search: vec![trial.clone()],
        confirmation: Trial {
            candidate_first: true,
            ..trial
        },
    }
}
#[test]
fn persisted_greedy_decisions_require_exact_device_driver_kernel_and_geometry() {
    let first = entry("GPU-A", TILE_SEARCH[1]);
    let second = entry("GPU-B", TILE_SEARCH[2]);
    let entries = [first.clone(), second.clone()];
    assert_eq!(
        cached_entry(&entries, &first.key).unwrap().selected,
        TILE_SEARCH[1]
    );
    assert_eq!(
        cached_entry(&entries, &second.key).unwrap().selected,
        TILE_SEARCH[2]
    );
    for changed in 0..5 {
        let mut key = first.key.clone();
        match changed {
            0 => key.driver_version += 1,
            1 => key.driver_build = Some("other-build".into()),
            2 => key.kernel_fingerprint = "other-kernel".into(),
            3 => key.rows += 1,
            _ => key.vocabulary += 1,
        }
        assert!(cached_entry(&entries, &key).is_none());
    }
}
#[test]
fn persisted_greedy_decisions_reject_missing_confirmation_or_bad_cold_warm_samples() {
    let good = entry("GPU-A", TILE_SEARCH[1]);
    assert!(good.valid());
    let encoded = serde_json::to_string(&good).unwrap();
    let restored: Entry = serde_json::from_str(&encoded).unwrap();
    assert!(restored.valid());
    for bad in 0..5 {
        let mut entry = good.clone();
        match bad {
            0 => entry.confirmation.candidate_first = false,
            1 => entry.confirmation.cold.candidate_ms[0] = f32::NAN,
            2 => entry.confirmation.warm.candidate_ms = vec![3.0; MIN_REPS],
            3 => entry.search.clear(),
            _ => entry
                .confirmation
                .cold
                .candidate_ms
                .pop()
                .map(|_| ())
                .unwrap(),
        }
        assert!(cached_entry(&[entry], &good.key).is_none());
    }
}

#[test]
#[ignore = "requires CUDA hardware and a writable tuning cache; run under safe-run"]
fn greedy_real_vocabulary_search_records_verified_cold_and_warm_confirmation() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let session = ResidentTuningSession::default();
    for rows in [1, 4, 12] {
        let key = key(&device, rows, 248_320)?;
        let entry = tune(&device, key, &session)?
            .ok_or_else(|| Error::invalid("greedy search did not complete"))?;
        assert!(entry.valid());
        let path = super::super::table_path()
            .ok_or_else(|| Error::invalid("greedy test requires tuning table path"))?;
        assert!(super::super::update_table(Some(&path), |table| {
            table.greedy.retain(|old| old.key != entry.key);
            table.greedy.push(entry.clone());
        }));
        println!("{}", serde_json::to_string(&entry).map_err(device_error)?);
    }
    Ok(())
}
