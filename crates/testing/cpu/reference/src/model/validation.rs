//! Validation responsibilities.
use super::ReferenceModel;
use infer_core::{Error, Result};
use infer_ir::{BackboneKind, FeedForward, Mixer, Modality};

impl ReferenceModel {
    ///
    /// # Errors
    /// Returns an invalid-input or invariant error if dimensions, identities, ranges, or ownership are inconsistent.
    pub fn validate(&self) -> Result<()> {
        self.ir.validate()?;
        let h = self.ir.hidden_size;
        let v = self.ir.vocab_size;
        let FeedForward::Dense { intermediate } = self.ir.feed_forward else {
            return Err(Error::unsupported(
                "reference fixture only supports dense FFN",
            ));
        };
        if h > crate::constants::REFERENCE_MAX_HIDDEN_SIZE
            || v > crate::constants::REFERENCE_MAX_VOCAB_SIZE
            || intermediate > crate::constants::REFERENCE_MAX_INTERMEDIATE_SIZE
            || self.ir.max_sequence > crate::constants::REFERENCE_MAX_SEQUENCE_LENGTH
            || self.ir.mixers.len() > crate::constants::REFERENCE_MAX_MIXER_COUNT
        {
            return Err(Error::unsupported(
                "model exceeds bounded reference executor dimensions",
            ));
        }
        if self.ir.norm_weight_offset != 0.0
            || self.ir.backbone != BackboneKind::Decoder
            || self.ir.modalities != vec![Modality::Text]
            || self.ir.tied_embeddings
            || self.ir.position.rotary_fraction.to_bits() != 1.0_f64.to_bits()
            || !self.ir.position.multimodal_sections.is_empty()
            || self.ir.position.interleaved
            || !h.is_multiple_of(2)
        {
            return Err(Error::unsupported(
                "reference executor requires a text decoder with full RoPE and untied weights",
            ));
        }
        for mixer in &self.ir.mixers {
            if *mixer
                != (Mixer::Attention {
                    query_heads: 1,
                    kv_heads: 1,
                    head_dim: h,
                    sliding_window: None,
                    output_gate: false,
                    qk_norm: false,
                })
            {
                return Err(Error::unsupported(
                    "reference executor supports single-head full attention",
                ));
            }
        }
        let check = |weights: &[f32], expected: usize| -> Result<()> {
            if weights.len() != expected || weights.iter().any(|x| !x.is_finite()) {
                Err(Error::invalid("weight shape or numeric value mismatch"))
            } else {
                Ok(())
            }
        };
        check(&self.embeddings, v * h)?;
        check(&self.lm_head, v * h)?;
        if self.layers.len() != self.ir.mixers.len() {
            return Err(Error::invalid("layer weight count mismatch"));
        }
        for layer in &self.layers {
            for w in [&layer.query, &layer.key, &layer.value, &layer.attention_out] {
                check(w, h * h)?;
            }
            check(&layer.gate, intermediate * h)?;
            check(&layer.up, intermediate * h)?;
            check(&layer.down, h * intermediate)?;
        }
        Ok(())
    }
}
