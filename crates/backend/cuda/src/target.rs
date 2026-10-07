//! Explicit hardware profiles. Architecture identifiers are not feature levels.
use infer_core::{Error, Result};
use infer_ir::DType;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum CudaTarget {
    HopperSm90,
    BlackwellSm120,
    Other(String),
}

impl CudaTarget {
    /// Compute dtypes this architecture executes natively.
    ///
    /// This is a capability, not a policy: which precision a checkpoint uses is declared by its
    /// model provider ([`infer_spi::PrecisionPolicy`]) and resolved against this answer. Declared
    /// requirements are validated against [`DeviceCapabilities`](infer_ir::DeviceCapabilities).
    #[must_use]
    pub const fn compute_dtypes(&self) -> &'static [DType] {
        match self {
            Self::BlackwellSm120 => &[
                DType::F32,
                DType::Bf16,
                DType::Fp8E4M3,
                DType::Fp8E5M2,
                DType::Fp4E2M1,
            ],
            Self::HopperSm90 => &[DType::F32, DType::Bf16, DType::Fp8E4M3, DType::Fp8E5M2],
            Self::Other(_) => &[DType::F32, DType::Bf16],
        }
    }

    /// Whether this architecture executes `dtype` natively.
    #[must_use]
    pub fn supports_compute(&self, dtype: DType) -> bool {
        self.compute_dtypes().contains(&dtype)
    }

    #[must_use]
    pub fn from_sm_name(name: &str) -> Self {
        match name {
            "sm_90" => Self::HopperSm90,
            "sm_120" => Self::BlackwellSm120,
            other => Self::Other(other.to_owned()),
        }
    }

    /// Gate this implementation's FP4 tile type before allocating launch output.
    /// # Errors
    /// Hopper cannot execute native FP4; other profiles require validation.
    pub fn require_native_nvfp4(&self) -> Result<()> {
        match self {
            Self::BlackwellSm120 => Ok(()),
            Self::HopperSm90 => Err(Error::unsupported(
                "Hopper/H200 has no native FP4 type; use the DecodedBf16 model-loading policy",
            )),
            Self::Other(name) => Err(Error::unsupported(format!(
                "native NVFP4 target {name} has not been validated by this backend"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_fp4_is_a_capability_not_a_storage_policy() {
        assert!(CudaTarget::BlackwellSm120.supports_compute(DType::Fp4E2M1));
        assert!(!CudaTarget::HopperSm90.supports_compute(DType::Fp4E2M1));
        assert!(!CudaTarget::Other("sm_999".into()).supports_compute(DType::Fp4E2M1));
        assert!(CudaTarget::HopperSm90.supports_compute(DType::Fp8E4M3));
        assert!(CudaTarget::Other("sm_999".into()).supports_compute(DType::Bf16));
    }

    #[test]
    fn production_and_test_devices_do_not_share_fp4_capability() {
        assert!(
            CudaTarget::from_sm_name("sm_90")
                .require_native_nvfp4()
                .is_err()
        );
        assert!(
            CudaTarget::from_sm_name("sm_120")
                .require_native_nvfp4()
                .is_ok()
        );
        assert!(
            CudaTarget::from_sm_name("sm_999")
                .require_native_nvfp4()
                .is_err()
        );
    }
}
