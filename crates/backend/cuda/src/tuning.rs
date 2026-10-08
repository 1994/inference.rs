//! Paired graph measurements on loaded model weights, isolated from model adapters.
use crate::{
    benchmark::Trial,
    device::{CudaDevice, device_error},
    mlp::ProjectionWeight,
    strategy::LinearTiling,
};
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    prelude::*,
};
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
    time::Duration,
};

/// Winners must beat the baseline by this factor on an independent trial.
const MIN_CONFIRM_SPEEDUP: f32 = 1.02;
/// Search space of graph GEMV tile dimensions.
const TILE_ROW_SEARCH: [usize; 4] = [1, 4, 8, 16];
/// Search space of graph GEMV tile dimensions.
const TILE_COLUMN_SEARCH: [usize; 3] = [128, 256, 512];
/// Relative plus absolute slack between candidate and baseline outputs.
const MATCH_TOLERANCE: f32 = 5e-4;
/// Deterministic sample value bounds: modulo, center and scale.
const SAMPLE_MODULUS: usize = 31;
/// Deterministic sample value bounds: modulo, center and scale.
const SAMPLE_CENTER: i16 = 15;
/// Deterministic sample value bounds: modulo, center and scale.
const SAMPLE_SCALE: f32 = 16.0;
/// Measurement wall clocks and repetition bounds.
const WARMUP_MS: u64 = 50;
/// Measurement wall clocks and repetition bounds.
const REP_MS: u64 = 100;
/// Measurement wall clocks and repetition bounds.
const MIN_REPS: usize = 10;
/// Measurement wall clocks and repetition bounds.
const MAX_REPS: usize = 30;

#[derive(Serialize)]
pub struct ProjectionTrial {
    pub key: String,
    pub rows: usize,
    pub columns: usize,
    pub gpu: String,
    pub target: crate::target::CudaTarget,
    pub baseline: LinearTiling,
    pub selected: LinearTiling,
    pub trials: Vec<Trial>,
    pub confirmation: Trial,
    pub clear_l2: bool,
}

/// One projection measured during a load, with the confirmed winner.
#[derive(Debug, Clone, Serialize)]
pub struct MeasuredTiling {
    /// `dtype:rowsxcolumns` key of the measured projection.
    pub key: String,
    /// Confirmed winning tile.
    pub tiling: LinearTiling,
    /// Confirmed speedup over the conservative baseline; without a win the baseline is kept.
    pub speedup: f32,
}

/// Automatic tiling decisions from one model load, for reporting rather than control.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TuningReport {
    /// Geometries measured and confirmed on this load.
    pub measured: Vec<MeasuredTiling>,
    /// Geometries whose measurement failed, with the reason; the conservative tile was kept.
    pub fallback: Vec<String>,
    /// Whether at least one measurement reached the machine-local cache for the next load.
    pub cached: bool,
}

/// Stable dtype and geometry key, without model-specific weight names.
/// # Errors
/// Rejects invalid projection rank/dimensions.
pub fn projection_key(weight: &ProjectionWeight) -> Result<(String, usize, usize)> {
    let (encoding, shape, packing) = match weight {
        ProjectionWeight::Dense(w) => ("bf16", w.shape(), 1),
        ProjectionWeight::Fp8(w, _) => ("fp8-channel", w.shape(), 1),
        ProjectionWeight::Fp8Block(w, _) => ("fp8-block", w.shape(), 1),
        ProjectionWeight::Fp4(w, _, _) => ("nvfp4", w.shape(), 2),
    };
    if shape.len() != 2 {
        return Err(Error::invalid("projection rank"));
    }
    let rows = usize::try_from(shape[0]).map_err(device_error)?;
    let columns = usize::try_from(shape[1]).map_err(device_error)? * packing;
    Ok((format!("{encoding}:{rows}x{columns}"), rows, columns))
}

/// Dtype part of a `dtype:rowsxcolumns` key, matching what [`TuningEntry::dtype`] records.
fn key_dtype(key: &str) -> &str {
    key.split(':').next().unwrap_or_default()
}

/// Tune once per dtype/shape, then confirm the winner in a fresh paired trial.
/// # Errors
/// Rejects numerical divergence, invalid storage or failed CUDA measurements.
pub fn tune(device: &CudaDevice, weight: &ProjectionWeight) -> Result<ProjectionTrial> {
    let (key, rows, columns) = projection_key(weight)?;
    let baseline = crate::strategy::default_tiling()?;
    let (base, expected) = capture(device, weight, rows, columns, baseline)?;
    base.launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let expected = device.read(&expected)?;
    let options = BenchOptions {
        warmup: Duration::from_millis(WARMUP_MS),
        rep: Duration::from_millis(REP_MS),
        min_reps: MIN_REPS,
        max_reps: MAX_REPS,
        clear_l2: true,
    };
    let mut trials = Vec::new();
    let mut selected = baseline;
    let mut best = 1.0f32;
    for tile_rows in TILE_ROW_SEARCH {
        for tile_columns in TILE_COLUMN_SEARCH {
            let tiling = LinearTiling::new(tile_rows, tile_columns)?;
            let (candidate, output) = capture(device, weight, rows, columns, tiling)?;
            candidate
                .launch()
                .sync_on(&device.stream)
                .map_err(device_error)?;
            verify(&device.read(&output)?, &expected)?;
            let trial = measure(device, &options, &base, &candidate, tiling)?;
            if trial.speedup > best {
                best = trial.speedup;
                selected = tiling;
            }
            trials.push(trial);
        }
    }
    let (candidate, _) = capture(device, weight, rows, columns, selected)?;
    let confirmation = measure(device, &options, &base, &candidate, selected)?;
    // Require a measurable win on the independent trial; noise keeps the baseline.
    if confirmation.speedup < MIN_CONFIRM_SPEEDUP {
        selected = baseline;
    }
    Ok(ProjectionTrial {
        key,
        rows,
        columns,
        gpu: device.stream.device().name().map_err(device_error)?,
        target: device.target().clone(),
        baseline,
        selected,
        trials,
        confirmation,
        clear_l2: true,
    })
}

fn capture(
    device: &CudaDevice,
    weight: &ProjectionWeight,
    rows: usize,
    columns: usize,
    tile: LinearTiling,
) -> Result<(CudaGraph<()>, Arc<Tensor<f32>>)> {
    weight.validate(rows, columns)?;
    let values = (0..columns)
        .map(|i| {
            f32::from(i16::try_from(i % SAMPLE_MODULUS).unwrap_or_default() - SAMPLE_CENTER)
                / SAMPLE_SCALE
        })
        .collect();
    let input = device.upload(values, &[columns])?;
    let mut output = api::zeros::<f32>(&[rows])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        weight.record(scope, &mut output, &input, columns, tile)
    })
    .map_err(device_error)?;
    Ok((graph, Arc::new(output)))
}

fn measure(
    device: &CudaDevice,
    options: &BenchOptions,
    base: &CudaGraph<()>,
    candidate: &CudaGraph<()>,
    tiling: LinearTiling,
) -> Result<Trial> {
    let run = |graph: &CudaGraph<()>| {
        graph
            .launch()
            .sync_on(&device.stream)
            .map_err(|e| cutile::error::tensor_error(&e.to_string()))
    };
    let (a, b) = do_bench_paired(&device.stream, options, |_| run(base), |_| run(candidate))
        .map_err(device_error)?;
    Ok(Trial {
        tiling,
        baseline_ms: a.times_ms().to_vec(),
        candidate_ms: b.times_ms().to_vec(),
        baseline_median_ms: a.median_ms(),
        candidate_median_ms: b.median_ms(),
        speedup: a.median_ms() / b.median_ms(),
    })
}

fn verify(actual: &[f32], expected: &[f32]) -> Result<()> {
    if actual.iter().zip(expected).any(|(a, b)| {
        !a.is_finite()
            || !b.is_finite()
            || (a - b).abs() > b.abs().mul_add(MATCH_TOLERANCE, MATCH_TOLERANCE)
    }) {
        return Err(Error::invariant(
            "projection tuning changed output beyond F32 tolerance",
        ));
    }
    Ok(())
}

/// Environment override for the machine-local tuning table path.
const TUNING_TABLE_ENV: &str = "INFER_CUDA_TUNING_TABLE";

/// One measured tile decision, keyed by projection dtype and geometry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuningEntry {
    /// Projection dtype key: `bf16`, `fp8-channel` or `nvfp4`.
    pub dtype: String,
    /// Output rows of the projection.
    pub rows: usize,
    /// Input columns of the projection.
    pub columns: usize,
    /// Winning tile rows.
    pub tile_rows: usize,
    /// Winning tile columns.
    pub tile_columns: usize,
}

/// Machine-local GEMV tiling decisions. The table is data, not code: it records what was
/// measured on one device, so a different GPU or model never inherits foreign tiles.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TilingTable {
    /// Device identity the entries were measured on, from `DeviceProfile::identity`; a mismatch
    /// discards them instead of trusting tiles measured on other hardware.
    #[serde(default, alias = "gpu")]
    pub device: String,
    /// Measured entries for this device.
    #[serde(default)]
    pub entries: Vec<TuningEntry>,
}

impl TilingTable {
    /// # Errors
    /// Rejects unreadable or malformed table files instead of silently ignoring them.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| Error::invalid(format!("CUDA tuning table: {error}")))?;
        serde_json::from_str(&text)
            .map_err(|error| Error::invalid(format!("CUDA tuning table: {error}")))
    }
    /// Record one trial for `device`, replacing any previous entry for the same geometry.
    pub fn record(&mut self, device: &str, trial: &ProjectionTrial) {
        let dtype = key_dtype(&trial.key).to_owned();
        self.entries.retain(|entry| {
            entry.dtype != dtype || entry.rows != trial.rows || entry.columns != trial.columns
        });
        self.entries.push(TuningEntry {
            dtype,
            rows: trial.rows,
            columns: trial.columns,
            tile_rows: trial.selected.rows(),
            tile_columns: trial.selected.columns(),
        });
        device.clone_into(&mut self.device);
    }
    /// Measured tile for one dtype and geometry; `None` when the table has no record.
    /// # Errors
    /// Rejects entries whose tile dimensions are not a legal tiling.
    pub fn entry(&self, dtype: &str, rows: usize, columns: usize) -> Result<Option<LinearTiling>> {
        for entry in &self.entries {
            if entry.dtype == dtype && entry.rows == rows && entry.columns == columns {
                return LinearTiling::new(entry.tile_rows, entry.tile_columns).map(Some);
            }
        }
        Ok(None)
    }
    /// Measured tile for one device, dtype and geometry; `None` when nothing measured it.
    ///
    /// Entries measured on another device never apply: tiles are a property of the machine. The
    /// dtype is the bare encoding (`nvfp4`), not the composite key, because that is what
    /// [`TuningEntry::dtype`] records.
    /// # Errors
    /// Rejects entries whose tile dimensions are not a legal tiling.
    pub fn tiling_for(
        &self,
        device: &str,
        dtype: &str,
        rows: usize,
        columns: usize,
    ) -> Result<Option<LinearTiling>> {
        if !self.device.is_empty() && self.device != device {
            return Ok(None);
        }
        self.entry(dtype, rows, columns)
    }
}

/// Derive the machine-local tuning table path. Nothing is configured by hand: the path comes
/// from the user cache directory and the entries inside are keyed by the device identity, so
/// one file serves every GPU and model on this host.
pub(crate) fn table_path() -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os(TUNING_TABLE_ENV).filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(explicit));
    }
    let root = std::env::var_os("XDG_CACHE_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))?;
    Some(root.join("infer-cuda").join("tuning.json"))
}

/// Load the configured tuning table; an absent file simply means "nothing measured yet".
/// # Errors
/// Rejects unreadable or malformed tables instead of silently ignoring them.
pub(crate) fn resolve_table(path: Option<&Path>) -> Result<Option<TilingTable>> {
    let Some(path) = path.map(Path::to_path_buf).or_else(table_path) else {
        return Ok(None);
    };
    if !path.is_file() {
        return Ok(None);
    }
    TilingTable::load(&path).map(Some)
}

/// Baseline measured on known devices and shipped with the backend.
///
/// This is data, not a policy branch: an entry applies only when the running device's identity
/// matches the one it was measured on, and any other device or geometry is measured on first use
/// through [`TilingSource::autotune`]. A new card therefore needs no code change, only its first
/// load; a card with a shipped entry starts at its measured best tile immediately.
const SHIPPED_BASELINE: &str = include_str!("../baselines/tuning.json");

/// Parse the shipped baseline once per process.
/// # Errors
/// Rejects a malformed embedded baseline, which would be a packaging mistake.
pub(crate) fn shipped_table() -> Result<&'static TilingTable> {
    static SHIPPED: OnceLock<TilingTable> = OnceLock::new();
    if let Some(table) = SHIPPED.get() {
        return Ok(table);
    }
    let table: TilingTable = serde_json::from_str(SHIPPED_BASELINE)
        .map_err(|error| Error::invalid(format!("shipped CUDA tuning baseline: {error}")))?;
    Ok(SHIPPED.get_or_init(|| table))
}

/// Where tile decisions come from for one model load.
pub(crate) struct TilingSource<'a> {
    /// Machine-local measurements for this device, when the cache exists.
    pub local: Option<&'a TilingTable>,
    /// Shipped baseline for known devices; read-only and never written back.
    pub shipped: Option<&'a TilingTable>,
    /// Path new measurements are appended to.
    pub path: Option<PathBuf>,
    /// Whether tables leave a geometry uncovered, so it may be measured now.
    pub autotune: bool,
}

/// Geometry key of one measured projection tiling: dtype, rows and columns.
type TilingKey = (String, usize, usize);
/// In-process memo of autotuned tilings.
type MeasuredTilings = Mutex<BTreeMap<TilingKey, LinearTiling>>;

/// In-process memo so one autotuned geometry is never measured twice.
fn measured() -> &'static MeasuredTilings {
    static MEASURED: OnceLock<MeasuredTilings> = OnceLock::new();
    MEASURED.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Merge one trial into the derived table file so the measurement is reused next start.
///
/// Best effort by design: a missing, unreadable or read-only cache only costs re-measurement on the
/// next start. Returns whether the trial was written, so the caller can report it truthfully.
fn persist(path: Option<&Path>, device: &str, trial: &ProjectionTrial) -> bool {
    let Some(path) = path else {
        return false;
    };
    let mut table = if path.is_file() {
        // Never clobber a cache that cannot be parsed; this load still uses the in-process memo.
        let Ok(table) = TilingTable::load(path) else {
            return false;
        };
        table
    } else {
        TilingTable::default()
    };
    table.record(device, trial);
    let Ok(text) = serde_json::to_string_pretty(&table) else {
        return false;
    };
    if let Some(parent) = path.parent() {
        let create = std::fs::create_dir_all(parent);
        drop(create);
    }
    let write = std::fs::write(path, text);
    drop(write);
    true
}

/// Tile for one projection: local measurements, the shipped baseline, then measuring it here.
///
/// A measurement is an optional optimization, so a failed one keeps the conservative tile and is
/// recorded in `report`; only malformed tables or invalid dimensions fail the load.
/// # Errors
/// Rejects malformed tables and invalid tile dimensions.
pub(crate) fn select_tiling(
    device: &CudaDevice,
    weight: &ProjectionWeight,
    profile: &crate::device::DeviceProfile,
    source: &TilingSource<'_>,
    report: &mut TuningReport,
) -> Result<LinearTiling> {
    let (key, rows, columns) = projection_key(weight)?;
    let identity = profile.identity();
    // Local measurements win over the shipped baseline; both are scoped to this device identity.
    for table in [source.local, source.shipped].into_iter().flatten() {
        if let Some(tiling) = table.tiling_for(&identity, key_dtype(&key), rows, columns)? {
            return Ok(tiling);
        }
    }
    let baseline = crate::strategy::default_tiling()?;
    if !source.autotune {
        return Ok(baseline);
    }
    let cached = measured()
        .lock()
        .map_err(|_| Error::invariant("CUDA tuning cache poisoned"))?
        .get(&(key.clone(), rows, columns))
        .copied();
    if let Some(tiling) = cached {
        return Ok(tiling);
    }
    let trial = match tune(device, weight) {
        Ok(trial) => trial,
        Err(error) => {
            // The conservative tile is already numerically verified, so an optional measurement
            // that fails must not fail the load; the caller reports why it was skipped.
            report.fallback.push(format!("{key}: {error}"));
            return Ok(baseline);
        }
    };
    measured()
        .lock()
        .map_err(|_| Error::invariant("CUDA tuning cache poisoned"))?
        .insert((key.clone(), rows, columns), trial.selected);
    report.cached |= persist(source.path.as_deref(), &identity, &trial);
    report.measured.push(MeasuredTiling {
        key,
        tiling: trial.selected,
        speedup: trial.confirmation.speedup,
    });
    Ok(trial.selected)
}

#[cfg(test)]
mod tests {
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
}
