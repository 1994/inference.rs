//! Exact speculative rejection sampling over the effective, filtered distributions.
use infer_core::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verification {
    Accepted(u32),
    Replaced(u32),
}

/// Draw from nonnegative unnormalized weights.
/// # Errors
/// Rejects invalid weights, empty mass or a uniform value outside [0, 1).
pub fn draw_distribution(weights: &[f64], uniform: f64) -> Result<u32> {
    if !uniform.is_finite()
        || !(0.0..1.0).contains(&uniform)
        || weights.is_empty()
        || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
    {
        return Err(Error::invalid("invalid speculative distribution"));
    }
    let sum: f64 = weights.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(Error::invalid("empty speculative probability mass"));
    }
    let threshold = uniform * sum;
    let mut cumulative = 0.0;
    let mut last = 0;
    for (index, weight) in weights.iter().enumerate() {
        if *weight > 0.0 {
            last = index;
        }
        cumulative += weight;
        if threshold < cumulative {
            return u32::try_from(index).map_err(|_| Error::invalid("vocabulary overflow"));
        }
    }
    u32::try_from(last).map_err(|_| Error::invalid("vocabulary overflow"))
}

/// Accept q's proposal with min(1,p/q), otherwise sample normalized max(p-q,0).
/// # Errors
/// Rejects non-normalized distributions, invalid proposals or invalid random inputs.
pub fn verify_draft(
    p: &[f64],
    q: &[f64],
    proposal: u32,
    acceptance: f64,
    correction: f64,
) -> Result<Verification> {
    let index = proposal as usize;
    if p.len() != q.len()
        || index >= p.len()
        || !acceptance.is_finite()
        || !(0.0..1.0).contains(&acceptance)
        || !correction.is_finite()
        || !(0.0..1.0).contains(&correction)
    {
        return Err(Error::invalid("invalid speculative verification input"));
    }
    for distribution in [p, q] {
        if distribution.iter().any(|w| !w.is_finite() || *w < 0.0)
            || (distribution.iter().sum::<f64>() - 1.0).abs() > 1e-8
        {
            return Err(Error::invalid(
                "speculative distributions must be normalized",
            ));
        }
    }
    if q[index] == 0.0 {
        return Err(Error::invalid("draft proposed a zero-probability token"));
    }
    if acceptance < (p[index] / q[index]).min(1.0) {
        return Ok(Verification::Accepted(proposal));
    }
    let residual: Vec<_> = p.iter().zip(q).map(|(p, q)| (p - q).max(0.0)).collect();
    draw_distribution(&residual, correction).map(Verification::Replaced)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejection_corrects_draft_bias_and_greedy_mismatch() -> Result<()> {
        let p = [0.2, 0.8];
        let q = [0.8, 0.2];
        assert_eq!(
            verify_draft(&p, &q, 0, 0.1, 0.3)?,
            Verification::Accepted(0)
        );
        assert_eq!(
            verify_draft(&p, &q, 0, 0.9, 0.3)?,
            Verification::Replaced(1)
        );
        assert_eq!(
            verify_draft(&[0.0, 1.0], &[1.0, 0.0], 0, 0.0, 0.5)?,
            Verification::Replaced(1)
        );
        let mut counts = [0u32; 2];
        for i in 0..40_000 {
            let proposal = draw_distribution(&q, crate::sampling_uniform(13, 2, i))?;
            let result = verify_draft(
                &p,
                &q,
                proposal,
                crate::sampling_uniform(13, 3, i),
                crate::sampling_uniform(13, 4, i),
            )?;
            let token = match result {
                Verification::Accepted(t) | Verification::Replaced(t) => t,
            };
            counts[token as usize] += 1;
        }
        assert!((f64::from(counts[0]) / 40_000.0 - p[0]).abs() < 0.01);
        Ok(())
    }
}
