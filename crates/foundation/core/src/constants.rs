//! Named limits shared by CPU topology discovery and OS affinity bitmaps.

/// Upper bound on CPUs or NUMA nodes representable in an OS topology bitmap.
pub const MAX_TOPOLOGY_CPUS: usize = 1_048_576;
