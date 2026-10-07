//! Host test kernel registration.
use infer_core::KernelId;
use infer_ir::{BackendKind, CapabilityRequirements, DType, Operation, PrecisionPlan};
use infer_spi::{KernelProvider, KernelRegistration, SourceLocation};

/// First non-zero kernel id handed to the host reference kernels.
const HOST_KERNEL_ID_BASE: u64 = 1000;

pub struct HostKernels;
impl KernelProvider for HostKernels {
    fn kernels(&self) -> Vec<KernelRegistration> {
        [
            Operation::TokenEmbedding,
            Operation::MatMul,
            Operation::RmsNorm,
            Operation::Split,
            Operation::Rope,
            Operation::Attention,
            Operation::Convolution,
            Operation::LinearAttention,
            Operation::GatedNorm,
            Operation::Silu,
            Operation::Sigmoid,
            Operation::Multiply,
            Operation::Residual,
            Operation::LmHead,
        ]
        .into_iter()
        .enumerate()
        .map(|(i, operation)| KernelRegistration {
            backend: BackendKind::TestCpu,
            id: KernelId::from_nonzero(
                std::num::NonZeroU64::MIN.saturating_add((HOST_KERNEL_ID_BASE + i as u64) - 1),
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
                crate_name: "infer-backend-host".into(),
                file: "crates/testing/cpu/host/src/executor/kernels.rs".into(),
                function: "execute".into(),
            },
        })
        .collect()
    }
}
