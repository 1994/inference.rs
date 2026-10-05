//! HF packages, metadata preflight, bounded weight loading, and text assets.
//! Device allocation and execution remain owned by independent backends.
mod loader;
mod package;
mod quantized;
#[cfg(test)]
mod shards_tests;
mod workload;
pub use workload::{ProjectionCatalog, ProjectionWorkload};
mod qwen;
mod safetensors;
mod text;
pub use loader::{
    LoadOptions, LoadedWeights, TensorLoadPlan, WeightLoadPlan, WeightStorage, WeightTarget,
    load_weights,
};
pub use package::{PackageManifest, QwenPackage, WeightBinding, package_path};
pub use quantized::{DeviceWeight, QuantizedPackage, WeightEncoding, WeightSource};
pub use qwen::{ImportedQwen, QwenProvider, VisionConfig};
pub use safetensors::{
    HostTensor, SafetensorsFile, TensorDtype, TensorHeader, convert_float_bytes, write_safetensors,
};
pub use text::{ChatMessage, ChatOptions, TextAssets};
mod index;
mod memory;
pub use index::SafetensorsIndex;
pub use memory::{MemoryEstimate, memory_estimate};

mod generation;
pub use generation::{GenerationDefaults, ResolvedGeneration, SamplingOverrides};
