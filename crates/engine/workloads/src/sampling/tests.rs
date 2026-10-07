use super::*;

#[test]
fn greedy_preserves_ties_signed_zero_and_rejects_nonfinite_tail() -> Result<()> {
    let sampling = Sampling::default();
    for (logits, expected) in [
        (vec![2.0, 2.0, 1.0], 0),
        (vec![-0.0, 0.0, 0.0], 1),
        (vec![f32::MIN, -5.0, -6.0], 1),
        (vec![f32::MAX, f32::MAX], 0),
    ] {
        assert_eq!(sample(&logits, &sampling, 1, 0)?, expected);
    }
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(sample(&[f32::MAX, invalid], &sampling, 1, 0).is_err());
    }
    assert!(sample(&[], &sampling, 1, 0).is_err());
    Ok(())
}
fn reference(logits: &[f32], sampling: &Sampling, request: u64, position: usize) -> Result<u32> {
    let mut candidates: Vec<_> = logits.iter().copied().enumerate().collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if let Some(k) = sampling.top_k {
        candidates.truncate(k);
    }
    let probabilities = crate::softmax(
        &candidates
            .iter()
            .map(|(_, value)| *value)
            .collect::<Vec<_>>(),
        sampling.temperature,
    )?;
    let mut cumulative = 0.0;
    let uniform = random(sampling.seed, request, position);
    for ((index, _), probability) in candidates.iter().zip(probabilities) {
        cumulative += f64::from(probability);
        if uniform < cumulative {
            return token(*index);
        }
    }
    token(
        candidates
            .last()
            .ok_or_else(|| Error::invariant("empty reference"))?
            .0,
    )
}
#[test]
fn partial_top_k_matches_full_sort_distribution_and_seeded_trajectory() -> Result<()> {
    let mut scratch = SamplingWorkspace::default();
    for seed in 0..128_u64 {
        let logits: Vec<f32> = (0..257_u16)
            .map(|index| {
                f32::from(
                    (u64::from(index) * 17 + seed)
                        .rem_euclid(53)
                        .try_into()
                        .unwrap_or(0_u16),
                )
            })
            .collect();
        for top_k in [None, Some(1), Some(7), Some(256), Some(1024)] {
            let sampling = Sampling {
                seed,
                top_k,
                temperature: 0.7,
                ..Sampling::default()
            };
            for position in 0..8 {
                assert_eq!(
                    sample_reusing(&logits, &sampling, seed + 1, position, &mut scratch)?,
                    reference(&logits, &sampling, seed + 1, position)?
                );
            }
        }
    }
    Ok(())
}
#[test]
fn repeated_sampling_preserves_allocations_and_greedy_ties_use_lowest_token() -> Result<()> {
    let logits = vec![1.0; 1024];
    let mut scratch = SamplingWorkspace::default();
    let mut sampling = Sampling {
        temperature: 1.0,
        ..Sampling::default()
    };
    sample_reusing(&logits, &sampling, 1, 0, &mut scratch)?;
    let pointers = (scratch.candidates.as_ptr(), scratch.weights.as_ptr());
    for position in 1..128 {
        sampling.top_k = Some(7);
        sample_reusing(&logits, &sampling, 1, position, &mut scratch)?;
        assert_eq!(
            (scratch.candidates.as_ptr(), scratch.weights.as_ptr()),
            pointers
        );
    }
    assert_eq!(sample(&logits, &Sampling::default(), 1, 0)?, 0);
    assert!(sample(&[f32::NAN], &sampling, 1, 0).is_err());
    Ok(())
}

#[test]
fn nucleus_minimum_probability_and_penalties_change_candidates() -> Result<()> {
    let mut scratch = SamplingWorkspace::default();
    let mut parameters = Sampling {
        temperature: 1.0,
        top_p: 0.5,
        ..Sampling::default()
    };
    for seed in 0..20 {
        parameters.seed = seed;
        assert_eq!(sample(&[3.0, 1.0, 0.0], &parameters, 1, 0)?, 0);
    }
    parameters.top_p = 1.0;
    parameters.min_p = 0.9;
    assert_eq!(sample(&[3.0, 1.0, 0.0], &parameters, 1, 0)?, 0);
    parameters.temperature = 0.0;
    parameters.presence_penalty = 2.0;
    let history = SamplingHistory {
        prompt: &[1],
        generated: &[0, 0],
    };
    assert_eq!(
        sample_with_history(&[3.0, 2.0], &parameters, 1, 0, history, &mut scratch)?,
        1
    );
    parameters.presence_penalty = 0.0;
    parameters.repetition_penalty = 2.0;
    let history = SamplingHistory {
        prompt: &[0],
        generated: &[],
    };
    assert_eq!(
        sample_with_history(&[3.0, 2.0], &parameters, 1, 0, history, &mut scratch)?,
        1
    );
    Ok(())
}
