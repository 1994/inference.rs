//! Progress quality contract.
use infer_core::{Error, ErrorCode, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressGuard {
    limit: usize,
    empty_plans: usize,
}
impl ProgressGuard {
    #[must_use]
    pub const fn valid_for(&self, limit: usize) -> bool {
        self.limit == limit && limit > 0
    }
    ///
    /// # Errors
    /// Returns an invalid-input error for invalid configuration, or a capacity error if the requested resources cannot be reserved.
    pub fn new(limit: usize) -> Result<Self> {
        if limit == 0 {
            return Err(Error::invalid("progress limit must be positive"));
        }
        Ok(Self {
            limit,
            empty_plans: 0,
        })
    }
    ///
    /// # Errors
    /// Returns an invariant error if feasible work is repeatedly left unscheduled past the configured starvation limit.
    pub fn observe(&mut self, runnable: usize, feasible: bool, scheduled: bool) -> Result<()> {
        if scheduled || runnable == 0 || !feasible {
            self.empty_plans = 0;
            return Ok(());
        }
        self.empty_plans = self.empty_plans.saturating_add(1);
        if self.empty_plans >= self.limit {
            return Err(Error::new(
                ErrorCode::Invariant,
                "ProgressInvariantViolation: feasible runnable work repeatedly received an empty StepPlan",
            ));
        }
        Ok(())
    }
}
