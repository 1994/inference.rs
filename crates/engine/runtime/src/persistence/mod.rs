//! Checkpoint coordination, snapshot capture/restore and deterministic replay.
pub mod checkpoint;
pub mod snapshot;
pub use snapshot::{ReplayAction, RuntimeSnapshot};
