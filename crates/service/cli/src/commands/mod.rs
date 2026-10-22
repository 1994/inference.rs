//! Typed command options and their execution paths.
use crate::{backend, support::print, support::read_json};
#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "cuda")))]
use crate::{
    host_quality, support::axum_serve, support::config, support::example, support::results,
    support::run_requests, support::run_to_idle, support::selected_engine, support::write_json,
};

mod options;
pub use options::*;

#[cfg(any(all(test, unix), all(target_os = "linux", feature = "cuda")))]
mod tuning_worker;

#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "cuda")))]
mod server;

#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "cuda")))]
mod inference;

mod verification;

#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "cuda")))]
mod replay;

mod measurement;

mod models;

#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "cuda")))]
pub use inference::run;
pub use measurement::compare;
#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "cuda")))]
pub use measurement::{benchmark, profile};
pub use models::{inspect_model, inspect_package, tokenize};
#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "cuda")))]
pub use replay::replay;
#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "cuda")))]
pub use server::serve;
pub use verification::verify;
