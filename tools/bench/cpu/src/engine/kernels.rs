//! Declared kernel support for the protocol double.
//!
//! The engine validates its plan against the kernel registry, so a benchmark that runs no model
//! still has to say which operations its declared model lowers to. These are declarations only:
//! there is no execution logic here, and the backend reports a planning sample rather than a real
//! device. They replace the reference executor's kernels, which the benchmark used to borrow.
use infer_core::KernelId;
use infer_ir::{BackendKind, CapabilityRequirements, DType, Operation, PrecisionPlan};
use infer_spi::{KernelProvider, KernelRegistration, SourceLocation};
use std::num::NonZeroU64;

/// The operations the benchmark's explicit decoder model lowers to.
const OPERATIONS: [Operation; 15] = [
    Operation::TokenEmbedding,
    Operation::RmsNorm,
    Operation::Attention,
    Operation::MatMul,
    Operation::Silu,
    Operation::Residual,
    Operation::LmHead,
    Operation::Pool,
    Operation::Rank,
    Operation::Decision,
    Operation::Split,
    Operation::Rope,
    Operation::Sigmoid,
    Operation::Multiply,
    Operation::GatedNorm,
];

pub struct DeclaredKernels;

impl KernelProvider for DeclaredKernels {
    fn kernels(&self) -> Vec<KernelRegistration> {
        OPERATIONS
            .into_iter()
            .enumerate()
            .map(|(index, operation)| KernelRegistration {
                backend: BackendKind::Cuda,
                id: KernelId::from_nonzero(
                    NonZeroU64::MIN.saturating_add(u64::try_from(index).unwrap_or(0)),
                ),
                operation,
                precision: PrecisionPlan::f32(),
                requirements: CapabilityRequirements {
                    compute_dtypes: vec![DType::F32],
                    ..CapabilityRequirements::default()
                },
                max_shape_elements: usize::MAX,
                workspace_bytes: 0,
                priority: 0,
                estimated_ns: 1,
                source: SourceLocation {
                    crate_name: "infer-cpu-bench".into(),
                    file: "tools/bench/cpu/src/engine/kernels.rs".into(),
                    function: "DeclaredKernels::kernels".into(),
                },
            })
            .collect()
    }
}
