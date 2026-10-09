use super::*;

const TABLE: &str = r#"{
        "device": "NVIDIA GeForce RTX 5090|sm_120|32768MiB",
        "entries": [
            {"dtype": "nvfp4", "rows": 17408, "columns": 5120, "tile_rows": 16, "tile_columns": 512},
            {"dtype": "bf16", "rows": 48, "columns": 5120, "tile_rows": 1, "tile_columns": 512}
        ]
    }"#;

#[test]
fn table_round_trips_and_scopes_entries_to_the_measured_device() -> Result<()> {
    let table: TilingTable = serde_json::from_str(TABLE)
        .map_err(|error| Error::invalid(format!("test table: {error}")))?;
    assert_eq!(
        table.entry("nvfp4", 17408, 5120)?,
        Some(LinearTiling::new(16, 512)?)
    );
    assert_eq!(
        table.entry("bf16", 48, 5120)?,
        Some(LinearTiling::new(1, 512)?)
    );
    // Unknown geometry and dtype stay unmeasured instead of guessing.
    assert_eq!(table.entry("nvfp4", 5120, 5120)?, None);
    assert_eq!(table.entry("fp8-channel", 48, 5120)?, None);
    // A table measured elsewhere is discarded rather than trusted.
    let foreign = TilingTable {
        device: "NVIDIA H100|sm_90|81920MiB".to_owned(),
        entries: table.entries.clone(),
    };
    assert_eq!(
        foreign.tiling_for("NVIDIA H100|sm_90|81920MiB", "nvfp4", 17408, 5120)?,
        Some(LinearTiling::new(16, 512)?)
    );
    assert_eq!(
        foreign.tiling_for(
            "NVIDIA GeForce RTX 5090|sm_120|32768MiB",
            "nvfp4",
            17408,
            5120
        )?,
        None,
        "device scope keeps foreign entries out of the lookup"
    );
    let encoded = serde_json::to_string(&table)
        .map_err(|error| Error::invalid(format!("test table: {error}")))?;
    let decoded: TilingTable = serde_json::from_str(&encoded)
        .map_err(|error| Error::invalid(format!("test table: {error}")))?;
    assert_eq!(decoded.entries, table.entries);
    Ok(())
}

#[test]
fn shipped_baseline_answers_only_the_device_it_was_measured_on() -> Result<()> {
    let table = shipped_table()?;
    assert!(
        !table.device.is_empty(),
        "a shipped table must record the device it was measured on"
    );
    assert!(
        !table.entries.is_empty(),
        "a shipped table must carry entries"
    );
    for entry in &table.entries {
        // Every shipped tile must be a legal geometry and must be reachable through the same
        // lookup the loader uses: bare dtype plus geometry, scoped to the measured device.
        let expected = LinearTiling::new(entry.tile_rows, entry.tile_columns)?;
        assert_eq!(
            table.tiling_for(&table.device, &entry.dtype, entry.rows, entry.columns)?,
            Some(expected)
        );
        assert_eq!(
            table.tiling_for(
                "unknown|sm_000|0MiB",
                &entry.dtype,
                entry.rows,
                entry.columns
            )?,
            None
        );
    }
    Ok(())
}
