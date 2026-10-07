use serde::{Deserialize, Serialize};

/// FP4 values sharing one E4M3 block scale in an NVFP4 weight.
pub const NVFP4_BLOCK_ELEMENTS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    Bf16,
    Fp8E4M3,
    Fp8E5M2,
    Fp4E2M1,
    U32,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleGranularity {
    Tensor,
    Channel,
    Block { elements: usize },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorLayout {
    RowMajor,
    ColumnMajor,
    Blocked { rows: usize, columns: usize },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScaleSpec {
    pub dtype: DType,
    pub granularity: ScaleGranularity,
    pub layout: TensorLayout,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrecisionPlan {
    pub storage: DType,
    pub compute: DType,
    pub accumulator: DType,
    pub scale: Option<ScaleSpec>,
    pub layout: TensorLayout,
}
impl PrecisionPlan {
    #[must_use]
    pub const fn f32() -> Self {
        Self {
            storage: DType::F32,
            compute: DType::F32,
            accumulator: DType::F32,
            scale: None,
            layout: TensorLayout::RowMajor,
        }
    }
    /// Dense BF16 storage with F32 accumulation; the universal fallback.
    #[must_use]
    pub const fn bf16() -> Self {
        Self {
            storage: DType::Bf16,
            compute: DType::Bf16,
            accumulator: DType::F32,
            scale: None,
            layout: TensorLayout::RowMajor,
        }
    }
    /// Per-channel FP8 storage with F32 accumulation.
    #[must_use]
    pub const fn fp8_channel() -> Self {
        Self {
            storage: DType::Fp8E4M3,
            compute: DType::Fp8E4M3,
            accumulator: DType::F32,
            scale: Some(ScaleSpec {
                dtype: DType::F32,
                granularity: ScaleGranularity::Channel,
                layout: TensorLayout::RowMajor,
            }),
            layout: TensorLayout::RowMajor,
        }
    }
    /// NVFP4 storage with one E4M3 scale per block of sixteen values.
    #[must_use]
    pub const fn nvfp4_block() -> Self {
        Self {
            storage: DType::Fp4E2M1,
            compute: DType::Fp4E2M1,
            accumulator: DType::F32,
            scale: Some(ScaleSpec {
                dtype: DType::Fp8E4M3,
                granularity: ScaleGranularity::Block {
                    elements: NVFP4_BLOCK_ELEMENTS,
                },
                layout: TensorLayout::RowMajor,
            }),
            layout: TensorLayout::RowMajor,
        }
    }
}
