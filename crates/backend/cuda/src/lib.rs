//! Native CUDA execution with GPU kernels authored in Rust through cuTile.
mod attention_profile;
pub use attention_profile::AttentionProfile;
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub mod attention;
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub mod benchmark;
pub(crate) mod constants;
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub mod device;
#[cfg(all(target_os = "linux", feature = "cuda"))]
mod kernels;
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub mod loading;
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub mod mlp;
pub mod nvfp4;
#[cfg(all(target_os = "linux", feature = "cuda"))]
mod quantized;
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub mod resident;
pub mod strategy;
pub mod target;
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub mod tuning;
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub mod vision;

#[cfg(all(target_os = "linux", feature = "cuda"))]
pub mod executor;
pub mod registry;

// Fixtures that two different module trees both need. Gated exactly like the modules that use it,
// so it is neither dead code nor built where CUDA does not compile.
#[cfg(all(test, target_os = "linux", feature = "cuda"))]
#[path = "../tests/unit/test_support.rs"]
pub(crate) mod test_support;
