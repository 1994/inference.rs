//! Verification quality contract.
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationReport {
    pub max_error: f64,
    pub rmse: f64,
    pub cosine: f64,
    pub passed: bool,
}
///
/// # Errors
/// Returns an invalid-input error for incompatible shapes, invalid tolerances, or non-finite values.
#[expect(
    clippy::cast_precision_loss,
    reason = "Floating-point statistics, normalization, and deterministic sampling intentionally convert bounded counts to floating-point values"
)]
#[expect(
    clippy::suboptimal_flops,
    reason = "Separate multiply/add rounding preserves the numerical contract of the independent Torch golden and the scalar reference"
)]
pub fn compare(
    reference: &[f32],
    candidate: &[f32],
    atol: f64,
    rtol: f64,
) -> Result<VerificationReport> {
    if reference.is_empty()
        || reference.len() != candidate.len()
        || reference.iter().chain(candidate).any(|x| !x.is_finite())
        || !atol.is_finite()
        || !rtol.is_finite()
        || atol < 0.0
        || rtol < 0.0
    {
        return Err(Error::invalid("invalid verification arrays/tolerances"));
    }
    let mut max_error = 0.0f64;
    let mut square = 0.0;
    let mut dot = 0.0;
    let mut a2 = 0.0;
    let mut b2 = 0.0;
    let mut passed = true;
    for (a, b) in reference.iter().zip(candidate) {
        let (a, b) = (f64::from(*a), f64::from(*b));
        let e = (a - b).abs();
        max_error = max_error.max(e);
        square += e * e;
        dot += a * b;
        a2 += a * a;
        b2 += b * b;
        passed &= e <= atol + rtol * a.abs();
    }
    let cosine = if a2 == 0.0 && b2 == 0.0 {
        1.0
    } else if a2 == 0.0 || b2 == 0.0 {
        0.0
    } else {
        dot / (a2 * b2).sqrt()
    };
    Ok(VerificationReport {
        max_error,
        rmse: (square / reference.len() as f64).sqrt(),
        cosine,
        passed,
    })
}
