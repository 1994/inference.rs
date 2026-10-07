//! Load-phase logging and error attribution.
//!
//! A checkpoint load can take minutes, so every phase reports when it starts and how long it
//! took, and names itself in any error it returns. That is what makes a bare driver failure
//! actionable: `out of memory` alone does not say whether the weights, the KV scales or the
//! vision tower ran out. Phase progress below the phase name (individual weights) is emitted at
//! `debug`, so a caller chooses how much detail it wants.
use infer_core::Error;
use std::time::Instant;

/// One named load phase in flight.
pub(super) struct Phase {
    name: &'static str,
    description: String,
    started: Instant,
}

impl Phase {
    /// Start a phase and report it. `name` is the machine-readable phase for filtering; the
    /// description is what an operator reads.
    pub(super) fn start(name: &'static str, description: impl Into<String>) -> Self {
        let description = description.into();
        tracing::info!(target: "infer::load", phase = name, "{description}");
        Self {
            name,
            description,
            started: Instant::now(),
        }
    }

    /// Report the phase finished, with its duration.
    pub(super) fn finish(self) {
        tracing::info!(
            target: "infer::load",
            phase = self.name,
            elapsed_ms = elapsed_ms(self.started),
            "{} finished",
            self.description
        );
    }

    /// Name this phase in a failure, so the error says which step ran out of room or failed.
    pub(super) fn fail(&self, error: &Error) -> Error {
        Error::new(
            error.code,
            format!(
                "{} failed after {:.1}s: {}",
                self.description,
                self.started.elapsed().as_secs_f64(),
                error.message
            ),
        )
    }
}

/// Whole milliseconds since `started`, saturating at the `u64` bound.
fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}
