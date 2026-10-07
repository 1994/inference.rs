//! Internal CPU dataflow executor used only by opt-in tests.
mod constants;
mod executor;
pub use executor::{
    HostBackend, HostConfig, HostInspection, HostKernels, HostTicket, LayerProbe, OpTrace,
    reference_operation,
};
