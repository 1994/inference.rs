//! Model state and workspace memory preflight.
use infer_core::{Error, Result};
use infer_ir::{DType, ModelIr};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEstimate {
    pub weights_bytes: u64,
    pub state_bytes: u64,
    pub workspace_bytes: u64,
    pub total_bytes: u64,
    pub device_bytes: u64,
    pub fits: bool,
}
///
/// # Errors
/// Returns an invalid-input or unsupported error for invalid model dimensions, unsupported state layouts, or size overflow.
pub fn memory_estimate(
    model: &ModelIr,
    weights_bytes: u64,
    context_tokens: usize,
    sequences: usize,
    workspace_bytes: u64,
    device_bytes: u64,
) -> Result<MemoryEstimate> {
    if context_tokens == 0 || context_tokens > model.max_sequence || sequences == 0 {
        return Err(Error::invalid("invalid memory estimate workload"));
    }
    let state = model.state.iter().try_fold(0u128, |sum, s| {
        let bits = match s.dtype {
            DType::F32 | DType::U32 => 32,
            DType::Bf16 | DType::F16 => 16,
            DType::Fp8E4M3 | DType::Fp8E5M2 => 8,
            DType::Fp4E2M1 => 4,
        };
        let positions = if s.per_token { context_tokens } else { 1 };
        let state_bits = (s.elements as u128)
            .checked_mul(positions as u128)
            .and_then(|v| v.checked_mul(bits))
            .ok_or_else(|| Error::invalid("state byte overflow"))?;
        sum.checked_add(state_bits.div_ceil(8))
            .ok_or_else(|| Error::invalid("state byte overflow"))
    })?;
    let state_bytes = u64::try_from(
        state
            .checked_mul(sequences as u128)
            .ok_or_else(|| Error::invalid("state memory overflow"))?,
    )
    .map_err(|_| Error::invalid("state memory overflow"))?;
    let total_bytes = weights_bytes
        .checked_add(state_bytes)
        .and_then(|v| v.checked_add(workspace_bytes))
        .ok_or_else(|| Error::invalid("total memory overflow"))?;
    Ok(MemoryEstimate {
        weights_bytes,
        state_bytes,
        workspace_bytes,
        total_bytes,
        device_bytes,
        fits: total_bytes <= device_bytes,
    })
}
