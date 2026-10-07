//! Native CUDA execution with GPU kernels authored in Rust through cuTile.
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
