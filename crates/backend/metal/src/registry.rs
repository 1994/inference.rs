use infer_core::KernelId;
use infer_ir::{BackendKind, CapabilityRequirements, DType, Operation, PrecisionPlan};
use infer_spi::{KernelProvider, KernelRegistration, SourceLocation};

/// First kernel id reserved for the Metal backend's registration table.
const KERNEL_ID_BASE: u64 = 2000;

pub struct MetalKernels;
impl KernelProvider for MetalKernels {
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
            backend: BackendKind::Metal,
            id: KernelId::from_nonzero(
                std::num::NonZeroU64::MIN.saturating_add((KERNEL_ID_BASE + i as u64) - 1),
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
                crate_name: "infer-backend-metal".into(),
                file: "crates/backend/metal/src/kernels.metal".into(),
                function: shader(operation).into(),
            },
        })
        .collect()
    }
}
pub const fn shader(op: Operation) -> &'static str {
    match op {
        Operation::TokenEmbedding => "embedding",
        Operation::MatMul | Operation::LmHead => "linear",
        Operation::RmsNorm => "norm",
        Operation::Split => "split",
        Operation::Rope => "rope",
        Operation::Attention => "attention",
        Operation::Convolution => "conv",
        Operation::LinearAttention => "delta",
        Operation::GatedNorm => "gated_norm",
        Operation::Silu | Operation::Sigmoid => "unary",
        Operation::Multiply | Operation::Residual => "binary",
        _ => "unsupported",
    }
}
