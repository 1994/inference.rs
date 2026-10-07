//! Forward responsibilities.
use super::{ReferenceModel, add_in_place, matvec, rms_norm, rope};
use infer_core::{Error, ErrorCode, Result};
use infer_ir::ModelOutput;

impl ReferenceModel {
    ///
    /// # Errors
    /// Returns an invalid-input error for an invalid model, empty prompt, or token outside the vocabulary.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
    )]
    #[expect(
        clippy::cast_precision_loss,
        reason = "Floating-point statistics, normalization, and deterministic sampling intentionally convert bounded counts to floating-point values"
    )]
    pub fn forward(&self, tokens: &[u32]) -> Result<ModelOutput> {
        self.validate()?;
        if tokens.is_empty()
            || tokens.len() > self.ir.max_sequence
            || tokens.iter().any(|t| *t as usize >= self.ir.vocab_size)
        {
            return Err(Error::invalid("invalid forward token sequence"));
        }
        let h = self.ir.hidden_size;
        let mut hidden: Vec<Vec<f32>> = tokens
            .iter()
            .map(|t| self.embeddings[*t as usize * h..(*t as usize + 1) * h].to_vec())
            .collect();
        for layer in &self.layers {
            let norm: Vec<_> = hidden
                .iter()
                .map(|x| rms_norm(x, self.ir.norm_epsilon))
                .collect();
            let mut queries = Vec::with_capacity(tokens.len());
            let mut keys = Vec::with_capacity(tokens.len());
            let mut values = Vec::with_capacity(tokens.len());
            for (position, x) in norm.iter().enumerate() {
                let mut q = matvec(&layer.query, x);
                let mut k = matvec(&layer.key, x);
                rope(&mut q, position, self.ir.position.rope_theta);
                rope(&mut k, position, self.ir.position.rope_theta);
                queries.push(q);
                keys.push(k);
                values.push(matvec(&layer.value, x));
            }
            for pos in 0..tokens.len() {
                let scores: Vec<f64> = keys[..=pos]
                    .iter()
                    .map(|k| {
                        queries[pos]
                            .iter()
                            .zip(k)
                            .map(|(q, k)| f64::from(*q) * f64::from(*k))
                            .sum::<f64>()
                            / (h as f64).sqrt()
                    })
                    .collect();
                let max = scores
                    .iter()
                    .copied()
                    .reduce(f64::max)
                    .ok_or_else(|| Error::invariant("nonempty attention"))?;
                let exp: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = exp.iter().sum();
                let attention: Vec<f32> = (0..h)
                    .map(|i| {
                        (exp.iter()
                            .zip(&values[..=pos])
                            .map(|(p, v)| p * f64::from(v[i]) / sum)
                            .sum::<f64>()) as f32
                    })
                    .collect();
                add_in_place(&mut hidden[pos], &matvec(&layer.attention_out, &attention));
                let x = rms_norm(&hidden[pos], self.ir.norm_epsilon);
                let gate = matvec(&layer.gate, &x);
                let up = matvec(&layer.up, &x);
                let activated: Vec<_> = gate
                    .iter()
                    .zip(up)
                    .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                    .collect();
                add_in_place(&mut hidden[pos], &matvec(&layer.down, &activated));
            }
        }
        let hidden: Vec<_> = hidden
            .iter()
            .map(|x| rms_norm(x, self.ir.norm_epsilon))
            .collect();
        let logits = matvec(
            &self.lm_head,
            hidden
                .last()
                .ok_or_else(|| Error::invariant("nonempty hidden"))?,
        );
        if logits
            .iter()
            .chain(hidden.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Err(Error::new(
                ErrorCode::Backend,
                "reference forward produced non-finite output",
            ));
        }
        Ok(ModelOutput {
            logits,
            hidden,
            tokens: Vec::new(),
        })
    }
}
