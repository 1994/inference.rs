//! Structured, non-panicking results for backend contract checks.
use infer_core::{Error, Result};

/// Outcome of one named contract check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// The backend satisfied the check.
    Passed,
    /// The backend violated the check; the string is the observed mismatch.
    Failed(String),
    /// The check does not apply to this backend or configuration.
    Skipped(&'static str),
}

/// One named check and its outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    /// Stable identifier used by adapters to allow-list optional gaps.
    pub name: &'static str,
    /// Whether a failure blocks Tier-0 conformance.
    pub required: bool,
    /// Observed outcome.
    pub status: Status,
}

/// Aggregated conformance result.
///
/// A required failure is reported through [`Report::require`] rather than a panic so an adapter can
/// print the complete matrix before failing the test.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    checks: Vec<Check>,
}

impl Report {
    /// Empty report.
    #[must_use]
    pub const fn new() -> Self {
        Self { checks: Vec::new() }
    }

    /// Record a required pass.
    pub fn pass(&mut self, name: &'static str) {
        self.record(name, true, Status::Passed);
    }

    /// Record a required failure.
    pub fn fail(&mut self, name: &'static str, detail: impl Into<String>) {
        self.record(name, true, Status::Failed(detail.into()));
    }

    /// Record a required check that does not apply to this backend.
    pub fn skip(&mut self, name: &'static str, reason: &'static str) {
        self.record(name, true, Status::Skipped(reason));
    }

    /// Record a check whose failure is informative but does not block Tier-0.
    pub fn optional(&mut self, name: &'static str, status: Status) {
        self.record(name, false, status);
    }

    /// Merge another report into this one, preserving its order.
    pub fn extend(&mut self, other: Self) {
        self.checks.extend(other.checks);
    }

    /// All checks in execution order.
    #[must_use]
    pub fn checks(&self) -> &[Check] {
        &self.checks
    }

    /// True when no required check failed.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.failures().next().is_none()
    }

    /// Required failures only.
    pub fn failures(&self) -> impl Iterator<Item = &Check> {
        self.checks
            .iter()
            .filter(|check| check.required && matches!(check.status, Status::Failed(_)))
    }

    /// Convert required failures into an error; optional and skipped checks never fail.
    ///
    /// # Errors
    /// Returns an invariant error naming every required check that failed.
    pub fn require(&self) -> Result<()> {
        let failed: Vec<_> = self.failures().map(|check| check.name).collect();
        if failed.is_empty() {
            return Ok(());
        }
        Err(Error::invariant(format!(
            "backend conformance failed: {}",
            failed.join(", ")
        )))
    }

    /// Human-readable counts and failed names for test output.
    #[must_use]
    pub fn summary(&self) -> String {
        let passed = self
            .checks
            .iter()
            .filter(|check| check.status == Status::Passed)
            .count();
        let skipped = self
            .checks
            .iter()
            .filter(|check| matches!(check.status, Status::Skipped(_)))
            .count();
        let failed = self.failures().count();
        let mut text = format!("{passed} passed, {failed} failed, {skipped} skipped");
        for check in self.failures() {
            if let Status::Failed(detail) = &check.status {
                text.push_str("\n- ");
                text.push_str(check.name);
                text.push_str(": ");
                text.push_str(detail);
            }
        }
        text
    }

    fn record(&mut self, name: &'static str, required: bool, status: Status) {
        self.checks.push(Check {
            name,
            required,
            status,
        });
    }
}

#[cfg(test)]
#[path = "../tests/unit/report.rs"]
mod tests;
