//! Package metadata, weight bindings and bounded loading.
pub mod index;
pub mod loader;
pub mod memory;
pub mod package;
pub mod quantized;
pub mod safetensors;
#[cfg(test)]
pub mod shards_tests;
