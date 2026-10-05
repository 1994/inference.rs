//! Fixture responsibilities.
use super::{LayerWeights, ReferenceModel};
use infer_core::ModelId;
use infer_ir::{
    BackboneKind, DType, FeedForward, Head, Mixer, Modality, ModelIr, PositionSpec, StateKind,
    StateRequirement,
};

impl ReferenceModel {
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
    )]
    #[expect(
        clippy::cast_precision_loss,
        reason = "Floating-point statistics, normalization, and deterministic sampling intentionally convert bounded counts to floating-point values"
    )]
    #[expect(
        clippy::suboptimal_flops,
        reason = "Separate multiply/add rounding preserves the numerical contract of the independent Torch golden and the scalar reference"
    )]
    pub fn fixture(id: ModelId, seed: u64) -> Self {
        let (hidden, intermediate, vocab, layers) = (8, 16, 32, 2);
        let ir = ModelIr {
            id,
            backbone: BackboneKind::Decoder,
            vocab_size: vocab,
            hidden_size: hidden,
            max_sequence: 128,
            mixers: vec![
                Mixer::Attention {
                    query_heads: 1,
                    kv_heads: 1,
                    head_dim: hidden,
                    sliding_window: None,
                    output_gate: false,
                    qk_norm: false
                };
                layers
            ],
            feed_forward: FeedForward::Dense { intermediate },
            position: PositionSpec {
                rope_theta: 10000.0,
                rotary_fraction: 1.0,
                multimodal_sections: vec![],
                interleaved: false,
            },
            norm_epsilon: 1e-5,
            norm_weight_offset: 0.0,
            heads: vec![
                Head::LanguageModel,
                Head::Embedding,
                Head::Rank,
                Head::Decision,
            ],
            modalities: vec![Modality::Text],
            state: (0..layers)
                .map(|layer| StateRequirement {
                    layer,
                    kind: StateKind::AttentionKv,
                    dtype: DType::F32,
                    elements: hidden * 2,
                    per_token: true,
                })
                .collect(),
            tied_embeddings: false,
        };
        let mut rng = seed;
        let mut weights = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    rng = rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
                    let mut z = rng;
                    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                    z ^= z >> 31;
                    (((z >> 40) as f64 / (1u64 << 24) as f64) * 2.0 - 1.0) as f32 * 0.15
                })
                .collect()
        };
        let embeddings = weights(vocab * hidden);
        let layers = (0..layers)
            .map(|_| LayerWeights {
                query: weights(hidden * hidden),
                key: weights(hidden * hidden),
                value: weights(hidden * hidden),
                attention_out: weights(hidden * hidden),
                gate: weights(intermediate * hidden),
                up: weights(intermediate * hidden),
                down: weights(hidden * intermediate),
            })
            .collect();
        let lm_head = weights(vocab * hidden);
        Self {
            ir,
            embeddings,
            layers,
            lm_head,
        }
    }
}
