//! Verification commands.
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
use super::host_quality;
use super::{VerifyOptions, backend, print, read_json};
#[cfg(feature = "test-backends")]
use super::{examples, outputs, results, run_requests, selected_engine};
use infer_core::{Error, Result};
#[cfg(feature = "test-backends")]
use infer_runtime::RuntimeConfig;

/// Batch width of the candidate engine in the fixture invariance check.
#[cfg(feature = "test-backends")]
const INVARIANCE_CANDIDATE_BATCH: usize = 4;

pub fn verify(options: VerifyOptions, backend_choice: &backend::Selection) -> Result<()> {
    let VerifyOptions {
        reference,
        candidate,
        atol,
        rtol,
        package,
        golden,
        host_memory_mib,
    } = options;
    if package.is_none()
        && let (Some(a), Some(b)) = (&reference, &candidate)
    {
        let report = infer_quality::compare(
            &read_json::<Vec<f32>>(a)?,
            &read_json::<Vec<f32>>(b)?,
            atol,
            rtol,
        )?;
        print(&report)?;
        return if report.passed {
            Ok(())
        } else {
            Err(Error::invariant("verification failed"))
        };
    }
    #[cfg(any(
        target_os = "macos",
        feature = "test-backends",
        all(target_os = "linux", feature = "cuda")
    ))]
    {
        if let Some(package) = package {
            let report = host_quality::verify_package(
                &package,
                golden
                    .as_deref()
                    .ok_or_else(|| Error::invariant("validated argument"))?,
                host_memory_mib,
                atol,
                rtol,
                backend_choice,
            )?;
            print(&report)?;
            return if report["passed"] == true {
                Ok(())
            } else {
                Err(Error::invariant("package golden verification failed"))
            };
        }
        #[cfg(feature = "test-backends")]
        {
            verify_fixture(host_memory_mib, backend_choice.clone())
        }
        #[cfg(not(feature = "test-backends"))]
        {
            Err(Error::invalid(
                "device backend verification requires --package and --golden",
            ))
        }
    }
    #[cfg(not(any(
        target_os = "macos",
        feature = "test-backends",
        all(target_os = "linux", feature = "cuda")
    )))]
    {
        if package.is_none() {
            return Err(Error::invalid(
                "device backend verification requires --package and --golden",
            ));
        }
        let _ = (golden, host_memory_mib, backend_choice);
        Err(Error::unsupported(
            "GPU verification requires a supported device backend",
        ))
    }
}
#[cfg(feature = "test-backends")]
pub(super) fn verify_fixture(
    host_memory_mib: u64,
    backend_choice: backend::Selection,
) -> Result<()> {
    let input = examples()?;
    let ids = input.iter().map(|r| r.id).collect::<Vec<_>>();
    let mut baseline = selected_engine(
        RuntimeConfig {
            max_num_seqs: 1,
            ..Default::default()
        },
        None,
        None,
        host_memory_mib,
        backend_choice.clone(),
    )?;
    run_requests(&mut baseline, input.clone())?;
    let mut candidate = selected_engine(
        RuntimeConfig {
            max_num_seqs: INVARIANCE_CANDIDATE_BATCH,
            max_num_batched_tokens: 2,
            ..Default::default()
        },
        None,
        None,
        host_memory_mib,
        backend_choice,
    )?;
    run_requests(&mut candidate, input)?;
    let passed = outputs(&baseline, &ids)? == outputs(&candidate, &ids)?
        && results(&baseline, &ids)?
            .iter()
            .chain(results(&candidate, &ids)?.iter())
            .all(|r| r.measurement.successful);
    print(
        &serde_json::json!({"scope":"CPU fixture runtime invariance","batch_invariance":passed,"chunk_invariance":passed,"state_leaks":candidate.inspect().state.allocated_pages,"passed":passed}),
    )?;
    if !passed {
        return Err(Error::invariant("runtime invariance failed"));
    }
    Ok(())
}
