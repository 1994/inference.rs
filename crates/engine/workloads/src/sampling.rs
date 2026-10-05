//! Deterministic sampling reuses worker-local storage and selects only the requested top K.
use infer_core::{Error, Result};
use infer_ir::Sampling;

#[derive(Default)]
pub struct SamplingWorkspace {
    candidates: Vec<(usize, f32)>,
    weights: Vec<f64>,
    seen: Vec<u8>,
}
impl SamplingWorkspace {
    /// Allocate worker-local storage before accepting requests.
    /// # Errors
    /// Reports capacity errors instead of growing on the first non-greedy completion.
    pub fn with_capacity(vocabulary: usize) -> Result<Self> {
        let mut workspace = Self::default();
        workspace
            .candidates
            .try_reserve_exact(vocabulary)
            .map_err(|error| Error::invalid(error.to_string()))?;
        workspace
            .weights
            .try_reserve_exact(vocabulary)
            .map_err(|error| Error::invalid(error.to_string()))?;
        Ok(workspace)
    }
}
/// Stateless seeded sampling makes trajectory replay independent of batching order.
/// # Errors
/// Rejects invalid logits, sampling parameters or vocabulary outside the token ABI.
pub fn sample(logits: &[f32], sampling: &Sampling, request: u64, position: usize) -> Result<u32> {
    sample_reusing(
        logits,
        sampling,
        request,
        position,
        &mut SamplingWorkspace::default(),
    )
}
/// Reuse scratch storage without retaining request logits or changing the sampling distribution.
/// # Errors
/// Rejects invalid logits, sampling parameters or vocabulary outside the token ABI.
pub fn sample_reusing(
    logits: &[f32],
    sampling: &Sampling,
    request: u64,
    position: usize,
    scratch: &mut SamplingWorkspace,
) -> Result<u32> {
    sample_with_history(
        logits,
        sampling,
        request,
        position,
        SamplingHistory::default(),
        scratch,
    )
}

#[derive(Default, Clone, Copy)]
pub struct SamplingHistory<'a> {
    pub prompt: &'a [u32],
    pub generated: &'a [u32],
}

/// Sample with repetition penalties over prompt/output and presence over output.
/// # Errors
/// Rejects invalid logits, history token IDs and sampling parameters.
pub fn sample_with_history(
    logits: &[f32],
    sampling: &Sampling,
    request: u64,
    position: usize,
    history: SamplingHistory<'_>,
    scratch: &mut SamplingWorkspace,
) -> Result<u32> {
    sampling.validate()?;
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(Error::invalid("invalid sampling logits"));
    }
    scratch.candidates.clear();
    scratch
        .candidates
        .extend(logits.iter().copied().enumerate());
    penalties(sampling, history, scratch)?;
    let order = |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    if sampling.temperature == 0.0 {
        return scratch
            .candidates
            .iter()
            .min_by(|a, b| order(a, b))
            .ok_or_else(|| Error::invalid("empty logits"))
            .and_then(|(index, _)| token(*index));
    }
    if let Some(k) = sampling.top_k.filter(|k| *k < scratch.candidates.len()) {
        scratch.candidates.select_nth_unstable_by(k, order);
        scratch.candidates.truncate(k);
    }
    scratch.candidates.sort_unstable_by(order);
    let maximum = scratch.candidates[0].1;
    scratch.weights.clear();
    scratch
        .weights
        .extend(scratch.candidates.iter().map(|(_, logit)| {
            ((f64::from(*logit) - f64::from(maximum)) / f64::from(sampling.temperature)).exp()
        }));
    filter(sampling, scratch);
    draw(scratch, random(sampling.seed, request, position))
}
fn token(index: usize) -> Result<u32> {
    u32::try_from(index).map_err(|_| Error::invalid("vocabulary exceeds token ABI"))
}
#[expect(
    clippy::cast_precision_loss,
    reason = "The seeded random generator maps a bounded 53-bit mantissa to the F64 uniform interval"
)]
fn random(seed: u64, request: u64, position: usize) -> f64 {
    let mut x = seed
        .wrapping_add(request.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add(position as u64);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    (x >> 11) as f64 / (1_u64 << 53) as f64
}
#[expect(
    clippy::cast_possible_truncation,
    reason = "F64 normalized weights are rounded to F32 exactly as the public F32 softmax sampling contract requires"
)]
fn draw(scratch: &SamplingWorkspace, random: f64) -> Result<u32> {
    let sum: f64 = scratch.weights.iter().sum();
    let mut cumulative = 0.0;
    for ((index, _), weight) in scratch.candidates.iter().zip(&scratch.weights) {
        cumulative += f64::from((weight / sum) as f32);
        if random < cumulative {
            return token(*index);
        }
    }
    token(
        scratch
            .candidates
            .last()
            .ok_or_else(|| Error::invariant("nonempty candidates"))?
            .0,
    )
}
#[cfg(test)]
mod tests;

fn penalties(
    sampling: &Sampling,
    history: SamplingHistory<'_>,
    scratch: &mut SamplingWorkspace,
) -> Result<()> {
    if sampling.presence_penalty == 0.0
        && sampling.repetition_penalty.to_bits() == 1.0_f32.to_bits()
    {
        return Ok(());
    }
    scratch.seen.resize(scratch.candidates.len(), 0);
    scratch.seen.fill(0);
    for (tokens, flag) in [(history.prompt, 1), (history.generated, 3)] {
        for token in tokens {
            let index =
                usize::try_from(*token).map_err(|_| Error::invalid("history token overflow"))?;
            let mark = scratch
                .seen
                .get_mut(index)
                .ok_or_else(|| Error::invalid("history token outside vocabulary"))?;
            *mark |= flag;
        }
    }
    for ((_, logit), seen) in scratch.candidates.iter_mut().zip(&scratch.seen) {
        if seen & 1 != 0 {
            *logit = if *logit < 0.0 {
                *logit * sampling.repetition_penalty
            } else {
                *logit / sampling.repetition_penalty
            };
        }
        if seen & 2 != 0 {
            *logit -= sampling.presence_penalty;
        }
    }
    Ok(())
}

fn filter(sampling: &Sampling, scratch: &mut SamplingWorkspace) {
    let count = scratch
        .weights
        .iter()
        .take_while(|w| **w >= f64::from(sampling.min_p))
        .count()
        .max(1);
    scratch.weights.truncate(count);
    scratch.candidates.truncate(count);
    if sampling.top_p < 1.0 {
        let threshold = scratch.weights.iter().sum::<f64>() * f64::from(sampling.top_p);
        let mut cumulative = 0.0;
        let mut retained = 0;
        for weight in &scratch.weights {
            cumulative += weight;
            retained += 1;
            if cumulative >= threshold {
                break;
            }
        }
        scratch.weights.truncate(retained);
        scratch.candidates.truncate(retained);
    }
}

/// Return the filtered distribution used by sampling, for speculative verification.
/// # Errors
/// Rejects the same invalid inputs as ordinary sampling.
pub fn probabilities(
    logits: &[f32],
    sampling: &Sampling,
    history: SamplingHistory<'_>,
) -> Result<Vec<f64>> {
    let mut scratch = SamplingWorkspace::default();
    let selected = sample_with_history(logits, sampling, 0, 0, history, &mut scratch)?;
    let mut probabilities = vec![0.0; logits.len()];
    if sampling.temperature == 0.0 {
        probabilities[selected as usize] = 1.0;
    } else {
        let sum: f64 = scratch.weights.iter().sum();
        for ((index, _), weight) in scratch.candidates.iter().zip(&scratch.weights) {
            probabilities[*index] = weight / sum;
        }
    }
    Ok(probabilities)
}

/// Deterministic independent random streams for draft, acceptance and correction.
#[must_use]
pub fn sampling_uniform(seed: u64, stream: u64, position: usize) -> f64 {
    random(seed, stream, position)
}
