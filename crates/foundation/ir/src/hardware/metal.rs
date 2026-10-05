use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetalCapabilities {
    pub simd_width: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetalRequirements {
    pub min_simd_width: Option<u32>,
}
impl MetalCapabilities {
    #[must_use]
    pub fn satisfies(&self, r: &MetalRequirements) -> bool {
        r.min_simd_width.is_none_or(|n| self.simd_width >= n)
    }
}
