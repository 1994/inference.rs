//! Native Metal backend and kernel registration.
//! Apple Metal device executor. CUDA remains a separate native execution API.
#[cfg(target_os = "macos")]
mod device;
#[cfg(target_os = "macos")]
mod executor;
#[cfg(target_os = "macos")]
pub use executor::{CommandTrace, MetalBackend, MetalConfig, MetalTicket};
mod registry;
pub use registry::MetalKernels;
