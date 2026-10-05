//! Model lowering and validated kernel/lifetime selection.
mod compilation;
pub mod dataflow;
pub use compilation::{compile, lower};
