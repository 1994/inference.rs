//! Long-history fixtures compare every F32 output bit without a quadratic CPU oracle.
use super::{
    APPEND_KEY_PHASE, APPEND_VALUE_PHASE, DIVISOR, Fixture, HISTORY_KEY_PHASE, HISTORY_VALUE_PHASE,
    KEY_SCALE, PERIOD, QUERY_CENTER, QUERY_STRIDE, Shape, VALUE_SCALE, code, decode,
};
use crate::constants::KV_BLOCK_TOKENS;
use crate::device::device_error;
use infer_core::{Error, Result};
const PHYSICAL_SHIFT: usize = 3;
impl Fixture {
    pub fn history(shape: Shape, history: usize, capacity: usize) -> Result<Self> {
        let block = KV_BLOCK_TOKENS;
        if !capacity.is_multiple_of(block)
            || history
                .checked_add(shape.lanes)
                .is_none_or(|end| end > capacity)
            || shape.lanes == 0
        {
            return Err(Error::invalid("attention history fixture bounds"));
        }
        let query = (0..shape.lanes * shape.heads * shape.dim)
            .map(|i| {
                Ok(
                    (f32::from(u16::try_from((i * QUERY_STRIDE) % PERIOD).map_err(device_error)?)
                        - QUERY_CENTER)
                        / DIVISOR,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let raw_len = shape.lanes * shape.kv_heads * shape.dim;
        let append_keys = (0..raw_len)
            .map(|i| code(i, APPEND_KEY_PHASE).map(|c| decode(c) * KEY_SCALE))
            .collect::<Result<Vec<_>>>()?;
        let append_values = (0..raw_len)
            .map(|i| code(i, APPEND_VALUE_PHASE).map(|c| decode(c) * VALUE_SCALE))
            .collect::<Result<Vec<_>>>()?;
        let size = shape.kv_heads * capacity * shape.dim;
        let keys = (0..size)
            .map(|i| code(i, HISTORY_KEY_PHASE))
            .collect::<Result<Vec<_>>>()?;
        let values = (0..size)
            .map(|i| code(i, HISTORY_VALUE_PHASE))
            .collect::<Result<Vec<_>>>()?;
        let blocks = capacity / block;
        let mut fixture = Self {
            shape,
            capacity,
            metadata: vec![
                i32::try_from(history).map_err(device_error)?,
                i32::try_from(shape.lanes - 1).map_err(device_error)?,
                0,
                0,
            ],
            table: (0..blocks)
                .map(|i| i32::try_from((i + PHYSICAL_SHIFT) % blocks).map_err(device_error))
                .collect::<Result<Vec<_>>>()?,
            query,
            append_keys,
            append_values,
            expected_keys: keys.clone(),
            expected_values: values.clone(),
            keys,
            values,
            // This fixture is only used for exact comparison and paired timing.
            // The independent F64 gate remains the separate complete four-fixture suite.
            expected: Vec::new(),
            window: shape.window,
        };
        fixture.append_reference()?;
        Ok(fixture)
    }
}
