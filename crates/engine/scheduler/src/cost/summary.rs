//! Exact native aggregate updates. No assumption is made about arbitrary provider costs.
use super::{Shape, aggregate, bucket};
use infer_core::{Error, Result};
use infer_ir::{CostEstimate, CostQuery, ExecutionRole};
use infer_spi::BatchCostSummary;

/// # Errors
/// Rejects invalid queries, nonmonotone feature replacement or overflowing native resources.
pub fn update_summary(
    mut summary: BatchCostSummary,
    previous: Option<CostQuery>,
    next: CostQuery,
) -> Result<BatchCostSummary> {
    if previous.is_some_and(|old| {
        old.role != next.role
            || old.workspace_bytes > next.workspace_bytes
            || old.context_tokens > next.context_tokens
    }) {
        return Err(Error::invariant("nonmonotone native summary replacement"));
    }
    let old = previous.map_or_else(|| Ok(CostEstimate::default()), |query| aggregate(&[query]))?;
    let new = aggregate(&[next])?;
    let error = || Error::invalid("batch summary cost overflow");
    let total = &mut summary.fallback;
    for (value, old, new) in [
        (&mut total.gpu_us, old.gpu_us, new.gpu_us),
        (&mut total.state_bytes, old.state_bytes, new.state_bytes),
        (&mut total.transfer_us, old.transfer_us, new.transfer_us),
        (&mut total.encoder_us, old.encoder_us, new.encoder_us),
    ] {
        *value = value
            .checked_sub(old)
            .and_then(|value| value.checked_add(new))
            .ok_or_else(error)?;
    }
    for (value, old, new) in [
        (&mut total.state_pages, old.state_pages, new.state_pages),
        (
            &mut total.logical_pages,
            old.logical_pages,
            new.logical_pages,
        ),
    ] {
        *value = value
            .checked_sub(old)
            .and_then(|value| value.checked_add(new))
            .ok_or_else(error)?;
    }
    total.workspace_bytes = total.workspace_bytes.max(new.workspace_bytes);
    summary.tokens = summary
        .tokens
        .checked_sub(previous.map_or(0, |query| query.tokens))
        .and_then(|tokens| tokens.checked_add(next.tokens))
        .ok_or_else(error)?;
    summary.batch = summary
        .batch
        .checked_add(usize::from(previous.is_none()))
        .ok_or_else(error)?;
    summary.context_tokens = summary.context_tokens.max(next.context_tokens);
    summary.roles |= match next.role {
        ExecutionRole::Prefill => 1,
        ExecutionRole::Decode => 2,
        ExecutionRole::Forward => 4,
        ExecutionRole::Mixed => return Err(Error::invalid("mixed summary query")),
    };
    Ok(summary)
}
pub(super) fn shape(summary: &BatchCostSummary) -> Shape {
    Shape {
        role: match summary.roles {
            1 => ExecutionRole::Prefill,
            2 => ExecutionRole::Decode,
            4 => ExecutionRole::Forward,
            _ => ExecutionRole::Mixed,
        },
        tokens: bucket(summary.tokens),
        context: bucket(summary.context_tokens),
        batch: bucket(summary.batch),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CalibratedCosts, FallbackCosts};
    use infer_core::{ProgramId, StepId};
    use infer_ir::{
        BackendKind, CostModelConfig, CostObservation, ExecutionTiming, PageGrowth, TimingSource,
    };
    use infer_spi::CostModelProvider;
    #[test]
    fn native_summaries_match_complete_queries_after_mixed_phase_growth_and_calibration()
    -> Result<()> {
        let mut costs = CalibratedCosts::new("summary-parity".into(), CostModelConfig::default())?;
        for round in 0..64 {
            let mut summary = BatchCostSummary::default();
            let mut queries = Vec::new();
            for index in 0..8 {
                let mut query = CostQuery::from_unit(
                    ProgramId::ONE,
                    BackendKind::Metal,
                    [
                        ExecutionRole::Prefill,
                        ExecutionRole::Decode,
                        ExecutionRole::Forward,
                    ][index % 3],
                    index + 1,
                    32 + round,
                    CostEstimate {
                        gpu_us: 2,
                        workspace_bytes: 16 + index as u64,
                        state_pages: 1,
                        state_bytes: 32,
                        transfer_us: 3,
                        encoder_us: 4,
                        ..CostEstimate::default()
                    },
                );
                query.page_growth = Some(PageGrowth {
                    page_tokens: 16,
                    allocated_pages: 1,
                    bytes_per_page: 64,
                    cow_tail: true,
                });
                query.logical_growth = Some(PageGrowth {
                    page_tokens: 8,
                    allocated_pages: 1,
                    bytes_per_page: 0,
                    cow_tail: false,
                });
                summary = update_summary(summary, None, query)?;
                queries.push(query);
                assert_eq!(
                    FallbackCosts.estimate_summary(&summary)?,
                    Some(aggregate(&queries)?)
                );
                assert_eq!(
                    costs.estimate_summary(&summary)?,
                    Some(costs.estimate(&queries)?)
                );
            }
            for index in 0..queries.len() {
                let previous = queries[index];
                queries[index].tokens += 1;
                queries[index].context_tokens += 1;
                summary = update_summary(summary, Some(previous), queries[index])?;
                assert_eq!(
                    FallbackCosts.estimate_summary(&summary)?,
                    Some(aggregate(&queries)?)
                );
                assert_eq!(
                    costs.estimate_summary(&summary)?,
                    Some(costs.estimate(&queries)?)
                );
            }
            costs.observe(&CostObservation {
                step: StepId::ONE,
                work: queries.into(),
                timing: ExecutionTiming {
                    elapsed_us: 100 + round as u64,
                    source: TimingSource::MetalGpu,
                },
            })?;
        }
        Ok(())
    }
}
