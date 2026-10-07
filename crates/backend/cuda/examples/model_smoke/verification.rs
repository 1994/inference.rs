use super::{Model, decode::DecodeResult};
use infer_core::{Error, Result};

pub enum TargetSteps {
    Sequential,
    Batch {
        outputs: std::vec::IntoIter<(Vec<f32>, Vec<f32>)>,
        committed: usize,
    },
}

impl TargetSteps {
    pub fn new(
        model: &mut Model,
        tokens: impl Iterator<Item = u32>,
        position: usize,
        stats: &mut DecodeResult,
    ) -> Result<Self> {
        if !model.batch_verify || model.resident.as_ref().is_none_or(|p| p.batch_width() == 1) {
            return Ok(Self::Sequential);
        }
        let program = model
            .resident
            .as_mut()
            .ok_or_else(|| Error::invariant("verification program"))?;
        let tokens: Vec<_> = tokens.collect();
        let outputs = program.step_batch(&tokens, position)?;
        stats.target_steps += program.batch_width();
        stats.target_batches += 1;
        Ok(Self::Batch {
            outputs: outputs.into_iter(),
            committed: 0,
        })
    }

    pub fn next(
        &mut self,
        model: &mut Model,
        token: u32,
        position: usize,
        stats: &mut DecodeResult,
    ) -> Result<Vec<f32>> {
        match self {
            Self::Sequential => {
                stats.target_steps += 1;
                model.step(token, position, true)
            }
            Self::Batch { outputs, committed } => {
                let (hidden, logits) = outputs
                    .next()
                    .ok_or_else(|| Error::invariant("verification output count"))?;
                model.hidden = hidden;
                *committed += 1;
                Ok(logits)
            }
        }
    }

    pub fn finish(self, model: &mut Model) -> Result<()> {
        if let Self::Batch { committed, .. } = self {
            model
                .resident
                .as_mut()
                .ok_or_else(|| Error::invariant("verification program"))?
                .commit_batch(committed)?;
        }
        Ok(())
    }
}
