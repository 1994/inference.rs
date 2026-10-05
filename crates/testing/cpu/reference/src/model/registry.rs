//! Registry reference implementation.
use infer_core::KernelId;
use infer_ir::{BackendKind, CapabilityRequirements, DType, Operation, PrecisionPlan};
use infer_spi::{KernelProvider, KernelRegistration, SourceLocation};

pub struct ReferenceKernels;
impl KernelProvider for ReferenceKernels {
    fn kernels(&self) -> Vec<KernelRegistration> {
        [
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
        ]
        .into_iter()
        .enumerate()
        .map(|(index, operation)| KernelRegistration {
            backend: BackendKind::TestCpu,
            id: KernelId::from_nonzero(
                std::num::NonZeroU64::MIN.saturating_add((index as u64 + 1) - 1),
            ),
            operation,
            precision: PrecisionPlan::f32(),
            requirements: CapabilityRequirements {
                compute_dtypes: vec![DType::F32],
                ..Default::default()
            },
            max_shape_elements: usize::MAX,
            workspace_bytes: 0,
            priority: 0,
            estimated_ns: 1,
            source: SourceLocation {
                crate_name: "infer-backend-reference".into(),
                file: "crates/testing/cpu/reference/src/model/forward.rs".into(),
                function: "ReferenceModel::forward".into(),
            },
        })
        .collect()
    }
}
