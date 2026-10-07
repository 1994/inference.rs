//! Incremental prompt ingestion with optional three-token GEMV or 32-token BF16 MMA batches.
use super::{Model, mtp::Mtp};
use infer_core::{Error, Result};

pub fn run(
    model: &mut Model,
    draft: &mut Option<Mtp>,
    input: &[u32],
    requested_width: usize,
) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Err(Error::invalid("empty prefill"));
    }
    let width = if requested_width > 1 && model.resident.is_some() {
        requested_width
    } else {
        1
    };
    let full = if width == 32 {
        input.len()
    } else if width > 1 {
        input.len() / width * width
    } else {
        0
    };
    let mut logits = Vec::new();
    for (batch, tokens) in input[..full].chunks(width).enumerate() {
        let start = batch * width;
        let outputs = model
            .resident
            .as_mut()
            .ok_or_else(|| Error::invariant("prefill program"))?
            .prefill_batch(tokens, start, start + tokens.len() == input.len())?;
        for (lane, (hidden, output)) in outputs.into_iter().enumerate() {
            let position = start + lane;
            if position > 0
                && let Some(draft) = draft
            {
                draft.step(tokens[lane], &model.hidden, position, false)?;
            }
            model.hidden = hidden;
            logits = output;
        }
    }
    for (position, token) in input.iter().enumerate().skip(full) {
        if position > 0
            && let Some(draft) = draft
        {
            draft.step(*token, &model.hidden, position, false)?;
        }
        logits = model.step(*token, position, position + 1 == input.len())?;
    }
    Ok(logits)
}
