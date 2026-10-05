use serde::{Deserialize, Serialize};

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
}
