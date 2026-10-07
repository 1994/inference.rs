use super::{SelectedBackend, Selection};
use infer_core::{Error, Result};
use std::path::Path;

pub fn load(path: &Path, memory_bytes: u64, selection: &Selection) -> Result<SelectedBackend> {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        use infer_backend_cuda::{
            device::CudaDevice,
            executor::CudaBackend,
            loading::{LoadOptions, LoadedModel},
        };
        if selection.num_gpu_blocks_override.is_some() || selection.upload_staging_mib.is_some() {
            return Err(Error::unsupported(
                "CUDA physical KV blocks/upload staging overrides are not implemented",
            ));
        }
        CudaDevice::enable_kernel_cache()?;
        let device = CudaDevice::new(0)?;
        let (free, total) = device.memory_info()?;
        // vLLM semantics: the utilization fraction bounds the whole engine, weights included.
        let share =
            crate::support::memory::utilization_bytes(total, selection.gpu_memory_utilization)?;
        let requested = share.min(if memory_bytes == 0 {
            total
        } else {
            memory_bytes
        });
        let limit = requested.min(free.saturating_sub(crate::constants::GIB_U64));
        preflight(path, device.target(), limit)?;
        let loaded = LoadedModel::open(
            device,
            path,
            infer_core::ModelId::ONE,
            LoadOptions {
                prefill_width: selection.max_num_batched_tokens.unwrap_or(0),
                mtp_depth: selection.num_speculative_tokens,
                autotune: selection.autotune,
                ..LoadOptions::default()
            },
        )?;
        report_tuning(loaded.tuning());
        let used = free.saturating_sub(loaded.device().memory_info()?.0);
        let state_budget = limit
            .checked_sub(used)
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| {
                Error::new(
                    infer_core::ErrorCode::Capacity,
                    "CUDA weights exhausted --host-memory-mib budget",
                )
            })?;
        let maximum_states = loaded.profile().resident_states();
        Ok(SelectedBackend::Cuda(Box::new(CudaBackend::new(
            loaded,
            state_budget,
            maximum_states,
        )?)))
    }
    #[cfg(not(all(target_os = "linux", feature = "cuda")))]
    {
        let _ = (path, memory_bytes, selection);
        Err(Error::unsupported(
            "CUDA execution requires Linux and a build with --features cuda",
        ))
    }
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn report_tuning(report: &infer_backend_cuda::tuning::TuningReport) {
    for decision in &report.measured {
        eprintln!(
            "cuda autotune: {} -> {}x{}  ({:.2}x the conservative tile)",
            decision.key,
            decision.tiling.rows(),
            decision.tiling.columns(),
            decision.speedup
        );
    }
    for failure in &report.fallback {
        eprintln!("cuda autotune: kept the conservative tile, measurement failed: {failure}");
    }
    if !report.measured.is_empty() {
        let count = report.measured.len();
        if report.cached {
            eprintln!(
                "cuda autotune: measured {count} projection tile(s); cached for the next load"
            );
        } else {
            eprintln!(
                "cuda autotune: measured {count} projection tile(s); no writable cache path, \
                 so the next load measures again"
            );
        }
    }
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn preflight(
    path: &Path,
    target: &infer_backend_cuda::target::CudaTarget,
    budget: u64,
) -> Result<()> {
    use infer_models::{QuantizedPackage, WeightEncoding};
    let package = QuantizedPackage::open(path, infer_core::ModelId::ONE)?;
    // The checkpoint's provider declares the precision preference; the device answers what it can
    // compute. A decoded fallback costs two bytes per element instead of the packed storage.
    let native_fp4 = package
        .imported
        .precision
        .resolve(|dtype| target.supports_compute(dtype))
        .storage
        == infer_ir::DType::Fp4E2M1;
    let mut minimum = 0_u64;
    for weight in package.weights.values() {
        let bytes = if weight.encoding == WeightEncoding::Nvfp4 && !native_fp4 {
            weight
                .shape
                .iter()
                .try_fold(2_u64, |n, d| n.checked_mul(*d as u64))
                .ok_or_else(|| Error::invalid("CUDA weight budget overflow"))?
        } else {
            weight.data.bytes
        };
        minimum = minimum
            .checked_add(bytes)
            .ok_or_else(|| Error::invalid("CUDA weight budget overflow"))?;
    }
    if minimum >= budget {
        return Err(Error::new(
            infer_core::ErrorCode::Capacity,
            format!(
                "CUDA model payload requires at least {minimum} bytes, exceeding available --host-memory-mib budget {budget}; allow additional request state space"
            ),
        ));
    }
    Ok(())
}
