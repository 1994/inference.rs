//! Scalar reference implementations of the physical operations.
//!
//! A device kernel is compared against a computation that shares nothing with it but the operation
//! and the state layout. That comparison used to live inside the CPU host executor, which E1
//! deletes; it is a reference rather than an executor, so it moves here and survives the deletion,
//! with the CUDA checks comparing against this crate instead of a backend.

mod kernels;

pub use kernels::{execute as reference_operation, sigmoid};
