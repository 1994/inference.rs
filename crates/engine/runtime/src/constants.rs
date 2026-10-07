//! Named bounds shared by the runtime's bounded collectors, owners and snapshots.

/// Largest capacity accepted for one bounded collector (events, history records or device states).
pub const MAX_BOUNDED_CAPACITY: usize = 1_048_576;
/// Largest poll interval a CPU owner may configure, in microseconds.
pub const MAX_POLL_INTERVAL_US: u64 = 1_000_000;
/// Snapshot schema version written by capture and required by restore validation.
pub const SNAPSHOT_SCHEMA_VERSION: u32 = 6;
