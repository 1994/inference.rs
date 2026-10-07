//! Backend-independent kernel selection and registration.
pub mod attention;
mod registry;
pub use registry::KernelRegistry;
