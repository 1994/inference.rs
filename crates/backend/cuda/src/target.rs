//! Explicit hardware profiles. Architecture identifiers are not feature levels.
use infer_core::{Error, Result};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum CudaTarget {
    HopperSm90,
    BlackwellSm120,
    Other(String),
}

impl CudaTarget {
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
                "Hopper/H200 has no native NVFP4 execution; use a separately validated BF16/FP8 conversion or software unpack path",
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
