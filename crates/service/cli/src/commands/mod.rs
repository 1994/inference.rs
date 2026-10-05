//! Typed command options and their execution paths.
use crate::{backend, support::print, support::read_json};
#[cfg(any(target_os = "macos", feature = "test-backends"))]
use crate::{
    host_quality, support::axum_serve, support::config, support::example, support::results,
    support::run_requests, support::run_to_idle, support::selected_engine, support::write_json,
};
#[cfg(feature = "test-backends")]
use crate::{support::examples, support::outputs};

mod options;
pub use options::*;

#[cfg(any(target_os = "macos", feature = "test-backends"))]
mod server;

#[cfg(any(target_os = "macos", feature = "test-backends"))]
mod inference;

mod verification;

#[cfg(any(target_os = "macos", feature = "test-backends"))]
mod replay;

mod measurement;

mod models;

#[cfg(feature = "test-backends")]
pub use inference::demo;
#[cfg(any(target_os = "macos", feature = "test-backends"))]
pub use inference::run;
pub use measurement::compare;
#[cfg(any(target_os = "macos", feature = "test-backends"))]
pub use measurement::{benchmark, profile};
pub use models::{inspect_model, inspect_package, tokenize};
#[cfg(any(target_os = "macos", feature = "test-backends"))]
pub use replay::replay;
#[cfg(any(target_os = "macos", feature = "test-backends"))]
pub use server::serve;
pub use verification::verify;
