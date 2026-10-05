//! Shared CLI facilities, grouped by responsibility.
#[cfg(any(target_os = "macos", feature = "test-backends"))]
mod engine;
#[cfg(any(target_os = "macos", feature = "test-backends"))]
pub use engine::*;
mod json;
pub use json::*;
#[cfg(any(target_os = "macos", feature = "test-backends"))]
mod fixtures;
#[cfg(any(target_os = "macos", feature = "test-backends"))]
pub use fixtures::*;
#[cfg(any(target_os = "macos", feature = "test-backends"))]
mod execution;
#[cfg(any(target_os = "macos", feature = "test-backends"))]
pub use execution::*;
#[cfg(any(target_os = "macos", feature = "test-backends"))]
mod server;
#[cfg(any(target_os = "macos", feature = "test-backends"))]
pub use server::*;
#[cfg(any(target_os = "macos", feature = "test-backends"))]
mod agent;

#[cfg(any(target_os = "macos", feature = "test-backends"))]
pub use agent::agent;
