//! Fixture responsibilities.
use super::{LayerWeights, ReferenceModel};
use crate::constants::{
    FIXTURE_HIDDEN_SIZE, FIXTURE_INTERMEDIATE_SIZE, FIXTURE_KV_ELEMENTS_PER_HEAD,
    FIXTURE_LAYER_COUNT, FIXTURE_MAX_SEQUENCE, FIXTURE_NORM_EPSILON, FIXTURE_ROPE_THETA,
    FIXTURE_UNIFORM_CENTER, FIXTURE_VOCAB_SIZE, FIXTURE_WEIGHT_SCALE, SPLITMIX_FINAL_MIX_SHIFT,
    SPLITMIX_GOLDEN_GAMMA, SPLITMIX_MIX_MULTIPLIER_A, SPLITMIX_MIX_MULTIPLIER_B,
    SPLITMIX_MIX_SHIFT_A, SPLITMIX_MIX_SHIFT_B, SPLITMIX_OUTPUT_SHIFT, SPLITMIX_UNIFORM_BASE,
};
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
        let (hidden, intermediate, vocab, layers) = (
            FIXTURE_HIDDEN_SIZE,
            FIXTURE_INTERMEDIATE_SIZE,
            FIXTURE_VOCAB_SIZE,
            FIXTURE_LAYER_COUNT,
        );
        let ir = ModelIr {
            id,
            backbone: BackboneKind::Decoder,
            vocab_size: vocab,
            hidden_size: hidden,
            max_sequence: FIXTURE_MAX_SEQUENCE,
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
                rope_theta: FIXTURE_ROPE_THETA,
                rotary_fraction: 1.0,
                multimodal_sections: vec![],
                interleaved: false,
            },
            norm_epsilon: FIXTURE_NORM_EPSILON,
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
                    elements: hidden * FIXTURE_KV_ELEMENTS_PER_HEAD,
                    per_token: true,
                })
                .collect(),
            tied_embeddings: false,
        };
        let mut rng = seed;
        let mut weights = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    rng = rng.wrapping_add(SPLITMIX_GOLDEN_GAMMA);
                    let mut z = rng;
                    z = (z ^ (z >> SPLITMIX_MIX_SHIFT_A)).wrapping_mul(SPLITMIX_MIX_MULTIPLIER_A);
                    z = (z ^ (z >> SPLITMIX_MIX_SHIFT_B)).wrapping_mul(SPLITMIX_MIX_MULTIPLIER_B);
                    z ^= z >> SPLITMIX_FINAL_MIX_SHIFT;
                    let uniform =
                        (z >> SPLITMIX_OUTPUT_SHIFT) as f64 / SPLITMIX_UNIFORM_BASE as f64;
                    (uniform * 2.0 - FIXTURE_UNIFORM_CENTER) as f32 * FIXTURE_WEIGHT_SCALE
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
