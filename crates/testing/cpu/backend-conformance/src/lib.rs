//! Backend-independent contract checks for device executors.
//!
//! The crate turns the Tier-0 expectations in
//! `docs/research/backend-standardization-2026-10-25.md` and the checklist in
//! `docs/guides/adding-a-backend.md` into executable assertions. A backend adds a device-gated
//! test that calls these functions with its own capabilities, compiled program and fixture; the
//! checks never open a device themselves.
//!
//! The checks cover the parts a new backend can get wrong without a numerical failure:
//! a manifest that registrations cannot select, a state quote below its own recipe, a missing
//! identity, or `validate_program` accepting a program compiled for another backend/model/precision.
//! Numerical parity and state paging remain backend-owned until the shared fixture runner lands.

mod manifest;
mod program;
mod report;
mod state;

#[cfg(test)]
#[path = "../tests/support/mod.rs"]
mod support;

pub use manifest::kernel_manifest;
pub use program::program_contract;
pub use report::{Check, Report, Status};
pub use state::backend_contract;
