//! Calibration quality contract.
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};

/// Maximum accepted calibration bin count, bounding the bucket index space.
const MAX_BINS: usize = 10000;
/// Tolerance for validating that a probability distribution sums to one.
const PROBABILITY_SUM_TOLERANCE: f64 = 1e-5;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub brier: f64,
    pub nll: f64,
    pub ece: f64,
    pub accuracy: f64,
}
///
/// # Errors
/// Returns an invalid-input error for empty or mismatched datasets, invalid labels, probabilities, or bin counts.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Calibration validates finite confidence in [0, 1] and at most 10000 bins before truncating to a bounded bin index"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "Floating-point statistics, normalization, and deterministic sampling intentionally convert bounded counts to floating-point values"
)]
#[expect(
    clippy::cast_sign_loss,
    reason = "Calibration validates finite confidence in [0, 1] and at most 10000 bins before truncating to a bounded bin index"
)]
pub fn calibration(
    probabilities: &[Vec<f32>],
    labels: &[usize],
    bins: usize,
) -> Result<CalibrationReport> {
    if probabilities.is_empty()
        || probabilities.len() != labels.len()
        || bins == 0
        || bins > MAX_BINS
    {
        return Err(Error::invalid("invalid calibration dataset"));
    }
    let mut brier = 0.0;
    let mut nll = 0.0;
    let mut correct = 0usize;
    let mut buckets = vec![(0usize, 0.0, 0usize); bins];
    for (p, label) in probabilities.iter().zip(labels) {
        if p.is_empty()
            || *label >= p.len()
            || p.iter().any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
            || (p.iter().map(|x| f64::from(*x)).sum::<f64>() - 1.0).abs()
                > PROBABILITY_SUM_TOLERANCE
        {
            return Err(Error::invalid("invalid probability distribution/label"));
        }
        let predicted = (0..p.len())
            .max_by(|a, b| p[*a].total_cmp(&p[*b]).then(b.cmp(a)))
            .ok_or_else(|| Error::invariant("nonempty probabilities"))?;
        let hit = usize::from(predicted == *label);
        correct += hit;
        brier += p
            .iter()
            .enumerate()
            .map(|(i, p)| (f64::from(*p) - f64::from(i == *label)).powi(2))
            .sum::<f64>();
        nll -= f64::from(p[*label]).max(f64::MIN_POSITIVE).ln();
        let confidence = f64::from(p[predicted]);
        let bucket = ((confidence * bins as f64) as usize).min(bins - 1);
        buckets[bucket].0 += 1;
        buckets[bucket].1 += confidence;
        buckets[bucket].2 += hit;
    }
    let n = labels.len() as f64;
    let ece = buckets
        .iter()
        .filter(|(count, _, _)| *count > 0)
        .map(|(_, sum, hits)| (sum - *hits as f64).abs() / n)
        .sum();
    Ok(CalibrationReport {
        brier: brier / n,
        nll: nll / n,
        ece,
        accuracy: correct as f64 / n,
    })
}
