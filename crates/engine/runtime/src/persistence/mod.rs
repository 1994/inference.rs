//! Checkpoint coordination, snapshot schema and deterministic replay.
pub mod checkpoint;
pub mod snapshot;
pub use snapshot::{ReplayAction, RuntimeSnapshot};
