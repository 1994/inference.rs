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
