//! Native NVFP4 activation quantizer compiler occupancy, selected before capture from measured evidence.
mod capture;
mod evidence;
use self::{
    capture::Buffers,
    evidence::{Entry, Key, Shape},
};
use super::resident_gemm::ResidentTuningSession;
use crate::device::CudaDevice;
use cutile::tile_kernel::CompileOptions;
pub use evidence::Entry as QuantEntry;
use infer_core::{Error, Result};
use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
};

/// Native quantization works on complete 512-value tiles.
const QUANT_ALIGNMENT: usize = 512;
/// Zero denotes the original compiler policy, rather than an occupancy hint.
const BASELINE: i32 = 0;
const CANDIDATES: [i32; 3] = [2, 4, 6];
const SCHEMA: &str =
    "native-nvfp4-quant-occupancy:same-address:four-global-mixed-all-bytes:cold-warm-AB-BA-v1";
type ProcessKey = (usize, Shape);
#[derive(Clone, Copy)]
struct Prepared {
    occupancy: i32,
    measured: bool,
}
type Memo = Mutex<BTreeMap<ProcessKey, Prepared>>;
fn memo() -> &'static Memo {
    static MEMO: OnceLock<Memo> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(BTreeMap::new()))
}
fn process_key(device: &CudaDevice, shape: Shape) -> ProcessKey {
    (device.stream.device().ordinal(), shape)
}
pub(crate) fn compile_options(occupancy: i32) -> CompileOptions {
    if occupancy == BASELINE {
        CompileOptions::default()
    } else {
        CompileOptions::default().occupancy(occupancy)
    }
}
/// Geometry is prepared before CUDA capture; decode keeps the original compiler policy.
/// The global scale is a runtime scalar, not a specialization: all candidates are gated across
/// four scales and mixed zero, tiny, sparse, sign, saturation and rounding fixtures.
pub(crate) fn select(
    device: &CudaDevice,
    rows: usize,
    columns: usize,
    autotune: bool,
    session: &ResidentTuningSession,
) -> i32 {
    let Some(shape) = Shape::new(rows, columns) else {
        return BASELINE;
    };
    if let Err(error) = resolve(device, shape, autotune, session) {
        tracing::warn!(target:"infer::tuning",?shape,%error,"NVFP4 quantizer original compiler fallback");
        return BASELINE;
    }
    memo()
        .lock()
        .ok()
        .and_then(|m| m.get(&process_key(device, shape)).map(|p| p.occupancy))
        .unwrap_or(BASELINE)
}
fn resolve(
    device: &CudaDevice,
    shape: Shape,
    autotune: bool,
    session: &ResidentTuningSession,
) -> Result<()> {
    let local = process_key(device, shape);
    if let Some(prepared) = memo()
        .lock()
        .map_err(|_| Error::invariant("quantizer memo poisoned"))?
        .get(&local)
        && (prepared.measured || !autotune)
    {
        return Ok(());
    }
    let mut buffers = Buffers::metadata(device, shape)?;
    let key = buffers.key(device, shape)?;
    let path = super::table_path();
    let table = if key.driver_build.is_some() {
        super::resolve_table(path.as_deref())?
    } else {
        None
    };
    let cached = table
        .as_ref()
        .and_then(|t| {
            t.resident_nvfp4_quant
                .iter()
                .find(|e| e.key == key && e.valid())
        })
        .cloned();
    let entry = if cached.is_some() {
        cached
    } else if autotune && session.permits_quant_search() {
        tune(device, key.clone(), session)?
    } else {
        None
    };
    let selected = entry.as_ref().map_or(BASELINE, |e| e.selected);
    if let Some(entry) = &entry {
        let from_table = table.as_ref().is_some_and(|t| {
            t.resident_nvfp4_quant
                .iter()
                .any(|e| e.key == key && e.valid())
        });
        if !from_table && key.driver_build.is_some() {
            super::update_table(path.as_deref(), |table| {
                table.resident_nvfp4_quant.retain(|old| old.key != key);
                table.resident_nvfp4_quant.push(entry.clone());
            });
        }
        tracing::info!(target:"infer::tuning",?shape,occupancy=selected,from_table,
            cold_speedup=entry.confirmation.cold.speedup(),warm_speedup=entry.confirmation.warm.speedup(),
            "NVFP4 quantizer compiler selected");
    }
    memo()
        .lock()
        .map_err(|_| Error::invariant("quantizer memo poisoned"))?
        .insert(
            local,
            Prepared {
                occupancy: selected,
                measured: entry.is_some(),
            },
        );
    Ok(())
}
fn tune(device: &CudaDevice, key: Key, session: &ResidentTuningSession) -> Result<Option<Entry>> {
    let compile = CompileOptions::default();
    if cfg!(debug_assertions)
        || compile.device_debug
        || compile
            .opt_level
            .is_some_and(|n| n != capture::JIT_OPT_LEVEL)
    {
        return Err(Error::invalid(
            "NVFP4 quantizer timing requires release CUDA JIT O3",
        ));
    }
    let mut buffers = Buffers::new(device, key.shape)?;
    let baseline = buffers.capture(device, BASELINE, 1.0)?;
    let expected = buffers.reference(device)?;
    buffers.verify(device, BASELINE, &expected)?;
    let verified_bytes = key.shape.verified_bytes();
    let mut search = Vec::new();
    let mut selected = BASELINE;
    let mut best = 1.0f32;
    for occupancy in CANDIDATES {
        if !session.permits_quant_search() {
            return Ok(None);
        }
        let candidate = buffers.capture(device, occupancy, 1.0)?;
        buffers.verify(device, occupancy, &expected)?;
        let trial = capture::measure(
            device,
            &baseline,
            &candidate,
            occupancy,
            false,
            verified_bytes,
        )?;
        buffers.verify(device, occupancy, &expected)?;
        if trial.confirmed() && trial.score() > best {
            selected = occupancy;
            best = trial.score();
        }
        search.push(trial);
    }
    if !session.permits_quant_search() {
        return Ok(None);
    }
    let candidate = buffers.capture(device, selected, 1.0)?;
    buffers.verify(device, selected, &expected)?;
    let mut confirmation = capture::measure(
        device,
        &baseline,
        &candidate,
        selected,
        true,
        verified_bytes,
    )?;
    buffers.verify(device, selected, &expected)?;
    if selected != BASELINE && !confirmation.confirmed() {
        selected = BASELINE;
        confirmation =
            capture::measure(device, &baseline, &baseline, BASELINE, true, verified_bytes)?;
    }
    let entry = Entry {
        key,
        selected,
        search,
        confirmation,
    };
    if !entry.valid() {
        return Err(Error::invariant("incomplete NVFP4 quantizer evidence"));
    }
    Ok(Some(entry))
}

#[cfg(test)]
#[path = "../../tests/unit/tuning_nvfp4_quant.rs"]
mod tests;
