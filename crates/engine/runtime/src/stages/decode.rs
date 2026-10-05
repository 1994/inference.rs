use crate::RequestRecord;
use infer_core::{Error, Result};
use infer_ir::ExecutionInput;

pub fn prepare(record: &RequestRecord, count: usize) -> Result<ExecutionInput> {
    if count != 1 || record.prefill_done < record.prefill_target {
        return Err(Error::invariant(
            "decode requires completed prefill and one new token",
        ));
    }
    let token = record
        .generated
        .last()
        .or_else(|| record.context.prompt.last())
        .copied()
        .ok_or_else(|| Error::invariant("decode input is empty"))?;
    Ok(ExecutionInput::Decode {
        position: record.context.len() - 1,
        token,
    })
}
pub fn commit(record: &RequestRecord, count: usize) -> Result<usize> {
    prepare(record, count)?;
    Ok(record.context.len())
}
