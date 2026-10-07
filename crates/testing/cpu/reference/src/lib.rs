//! Internal scalar reference model for correctness tests.
mod constants;
mod model;
pub use model::{
    LayerWeights, ReferenceBackend, ReferenceKernels, ReferenceModel, ReferenceTicket,
};
