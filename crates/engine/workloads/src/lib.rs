//! Native planning and output semantics for four orthogonal workloads.
mod sampling;
pub use sampling::{SamplingWorkspace, sample, sample_reusing};
mod registry;
pub use registry::WorkloadRegistry;
mod readouts;
pub use readouts::ProjectionWorkloads;
mod math;
mod native;
mod outputs;
#[cfg(test)]
mod tests;
pub use math::{pool, softmax};
pub use native::NativeWorkloads;
use outputs::{decisions, embedding, ranking};
