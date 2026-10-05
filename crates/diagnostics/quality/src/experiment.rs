//! Experiment quality contract.
use super::BenchmarkReport;
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentVerdict {
    pub accepted: bool,
    pub goodput_change: f64,
    pub reasons: Vec<String>,
}
///
/// # Errors
/// Returns an invalid-input error for an invalid regression threshold or incompatible benchmark fingerprints.
#[expect(
    clippy::cast_precision_loss,
    reason = "Floating-point statistics, normalization, and deterministic sampling intentionally convert bounded counts to floating-point values"
)]
pub fn experiment(
    baseline: &BenchmarkReport,
    candidate: &BenchmarkReport,
    correctness: bool,
    max_p99_regression: f64,
) -> Result<ExperimentVerdict> {
    for report in [baseline, candidate] {
        if report.backend.is_empty()
            || report.workload_fingerprint.is_empty()
            || report.elapsed_us == 0
            || report.requests == 0
            || report.successful > report.requests
            || report.p50_e2e_us > report.p95_e2e_us
            || report.p95_e2e_us > report.p99_e2e_us
            || [
                report.requests_per_second,
                report.tokens_per_second,
                report.slo_goodput,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(Error::invalid("invalid experiment measurements"));
        }
    }
    if baseline.slo_goodput <= 0.0 || !max_p99_regression.is_finite() || max_p99_regression < 0.0 {
        return Err(Error::invalid("invalid experiment baseline/constraints"));
    }
    let mut reasons = Vec::new();
    if baseline.workload_fingerprint != candidate.workload_fingerprint
        || baseline.ttft_slo_us != candidate.ttft_slo_us
        || baseline.tpot_slo_us != candidate.tpot_slo_us
    {
        reasons.push("workload or SLO configuration differs".into());
    }
    if !correctness {
        reasons.push("correctness failed".into());
    }
    if candidate.slo_goodput < baseline.slo_goodput {
        reasons.push("SLO goodput regressed".into());
    }
    if candidate.p99_e2e_us as f64 > baseline.p99_e2e_us as f64 * (1.0 + max_p99_regression) {
        reasons.push("P99 latency constraint failed".into());
    }
    if candidate.requests != baseline.requests || candidate.successful != candidate.requests {
        reasons.push("workload/sample count or request success differs".into());
    }
    Ok(ExperimentVerdict {
        accepted: reasons.is_empty(),
        goodput_change: candidate.slo_goodput / baseline.slo_goodput - 1.0,
        reasons,
    })
}
