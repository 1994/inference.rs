//! Shared CLI facilities, grouped by responsibility.
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
mod engine;
#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "cuda")))]
pub mod memory;
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
mod serving;
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
pub use engine::*;
mod json;
pub use json::*;
mod logging;
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub use logging::human_bytes;
pub use logging::{flush as flush_logs, init as init_logging};
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
mod fixtures;
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
pub use fixtures::*;
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
mod execution;
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
pub use execution::*;
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
mod server;
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
pub use server::*;
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
mod agent;

#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
pub use agent::agent;
