use infer_core::{Error, ModelId, ProviderId, Result};
use infer_ir::{
    BackboneKind, DType, FeedForward, Head, Mixer, Modality, ModelIr, PositionSpec, StateKind,
    StateRequirement,
};
use infer_spi::{ModelProvider, ProviderMetadata, SPI_VERSION};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub out_hidden_size: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportedQwen {
    pub model: ModelIr,
    pub vision: Option<VisionConfig>,
    pub mtp_layers: usize,
    pub eos_token: Option<u32>,
}
#[derive(Deserialize)]
struct HfConfig {
    model_type: String,
    text_config: Option<TextConfig>,
    vision_config: Option<VisionConfig>,
    #[serde(default)]
    language_model_only: bool,
    #[serde(flatten)]
    text: serde_json::Map<String, serde_json::Value>,
}
#[derive(Deserialize)]
struct RopeConfig {
    #[serde(default = "theta")]
    rope_theta: f64,
    #[serde(default = "one")]
    partial_rotary_factor: f64,
    #[serde(default)]
    mrope_section: Vec<usize>,
    #[serde(default)]
    mrope_interleaved: bool,
    #[serde(default)]
    rope_type: Option<String>,
}
const fn theta() -> f64 {
    10000.0
}
const fn one() -> f64 {
    1.0
}
const fn epsilon() -> f32 {
    1e-6
}
const fn head_dim() -> usize {
    128
}
#[derive(Deserialize)]
struct TextConfig {
    hidden_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    #[serde(default = "head_dim")]
    head_dim: usize,
    #[serde(default = "epsilon")]
    rms_norm_eps: f32,
    #[serde(default)]
    layer_types: Vec<String>,
    #[serde(default)]
    attn_output_gate: bool,
    #[serde(default)]
    output_gate_type: Option<String>,
    #[serde(default)]
    linear_num_key_heads: usize,
    #[serde(default)]
    linear_num_value_heads: usize,
    #[serde(default)]
    linear_key_head_dim: usize,
    #[serde(default)]
    linear_value_head_dim: usize,
    #[serde(default)]
    linear_conv_kernel_dim: usize,
    #[serde(default)]
    rope_parameters: Option<RopeConfig>,
    #[serde(default = "theta")]
    rope_theta: f64,
    #[serde(default = "one")]
    partial_rotary_factor: f64,
    #[serde(default)]
    mtp_num_hidden_layers: usize,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    eos_token_id: Option<u32>,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    attention_bias: bool,
}
pub struct QwenProvider;
impl QwenProvider {
    ///
    /// # Errors
    /// Returns an invalid-input or unsupported error for malformed configuration or an unsupported Qwen architecture.
    pub fn import_manifest(&self, id: ModelId, bytes: &[u8]) -> Result<ImportedQwen> {
        let config: HfConfig =
            serde_json::from_slice(bytes).map_err(|e| Error::invalid(e.to_string()))?;
        let hybrid = match config.model_type.as_str() {
            "qwen3_5" | "qwen3_5_text" => true,
            "qwen3" => false,
            other => return Err(Error::unsupported(format!("model type {other}"))),
        };
        let text = if let Some(text) = config.text_config {
            text
        } else {
            serde_json::from_value(serde_json::Value::Object(config.text))
                .map_err(|e| Error::invalid(e.to_string()))?
        };
        let (mixers, state) = text.import_layers(hybrid)?;
        let rope = text.rope_parameters.unwrap_or(RopeConfig {
            rope_theta: text.rope_theta,
            partial_rotary_factor: text.partial_rotary_factor,
            mrope_section: vec![],
            mrope_interleaved: false,
            rope_type: None,
        });
        if rope.rope_type.as_deref().is_some_and(|s| s != "default") {
            return Err(Error::unsupported("RoPE scaling provider required"));
        }
        let vision = if config.language_model_only {
            None
        } else {
            config.vision_config
        };
        validate_vision(vision.as_ref(), text.hidden_size)?;
        let model = ModelIr {
            id,
            backbone: if hybrid {
                BackboneKind::Hybrid
            } else {
                BackboneKind::Decoder
            },
            vocab_size: text.vocab_size,
            hidden_size: text.hidden_size,
            max_sequence: text.max_position_embeddings,
            mixers,
            feed_forward: FeedForward::Dense {
                intermediate: text.intermediate_size,
            },
            position: PositionSpec {
                rope_theta: rope.rope_theta,
                rotary_fraction: rope.partial_rotary_factor,
                multimodal_sections: rope.mrope_section,
                interleaved: rope.mrope_interleaved,
            },
            norm_epsilon: text.rms_norm_eps,
            norm_weight_offset: if hybrid { 1.0 } else { 0.0 },
            heads: vec![Head::LanguageModel],
            modalities: if vision.is_some() {
                vec![Modality::Text, Modality::Image, Modality::Video]
            } else {
                vec![Modality::Text]
            },
            state,
            tied_embeddings: text.tie_word_embeddings,
        };
        model.validate()?;
        Ok(ImportedQwen {
            model,
            vision,
            mtp_layers: text.mtp_num_hidden_layers,
            eos_token: text.eos_token_id,
        })
    }
}
impl ModelProvider for QwenProvider {
    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            id: ProviderId::ONE,
            name: "qwen-hf-config".into(),
            spi_version: SPI_VERSION,
        }
    }
    fn import(&self, id: ModelId, config: &[u8]) -> Result<ModelIr> {
        Ok(self.import_manifest(id, config)?.model)
    }
}

impl TextConfig {
    fn import_layers(&self, hybrid: bool) -> Result<(Vec<Mixer>, Vec<StateRequirement>)> {
        if self.hidden_act.as_deref().is_some_and(|s| s != "silu")
            || self.attention_bias
            || self
                .output_gate_type
                .as_deref()
                .is_some_and(|s| s != "swish")
        {
            return Err(Error::unsupported(
                "unmodeled attention/FFN semantics in Qwen configuration",
            ));
        }
        if self.num_hidden_layers == 0 || self.num_hidden_layers > 10000 {
            return Err(Error::invalid("invalid layer count"));
        }
        let layer_types = if self.layer_types.is_empty() && !hybrid {
            vec!["full_attention".into(); self.num_hidden_layers]
        } else {
            self.layer_types.clone()
        };
        if layer_types.len() != self.num_hidden_layers {
            return Err(Error::invalid("layer_types must cover every layer"));
        }
        let mut mixers = Vec::new();
        let mut state = Vec::new();
        for (layer, kind) in layer_types.iter().enumerate() {
            match kind.as_str() {
                "full_attention" => {
                    mixers.push(Mixer::Attention {
                        query_heads: self.num_attention_heads,
                        kv_heads: self.num_key_value_heads,
                        head_dim: self.head_dim,
                        sliding_window: None,
                        output_gate: self.attn_output_gate,
                        qk_norm: true,
                    });
                    let elements = self
                        .num_key_value_heads
                        .checked_mul(self.head_dim)
                        .and_then(|v| v.checked_mul(2))
                        .ok_or_else(|| Error::invalid("KV shape overflow"))?;
                    state.push(StateRequirement {
                        layer,
                        kind: StateKind::AttentionKv,
                        dtype: DType::Bf16,
                        elements,
                        per_token: true,
                    });
                }
                "linear_attention" if hybrid => {
                    mixers.push(Mixer::LinearAttention {
                        key_heads: self.linear_num_key_heads,
                        value_heads: self.linear_num_value_heads,
                        key_dim: self.linear_key_head_dim,
                        value_dim: self.linear_value_head_dim,
                        conv_kernel: self.linear_conv_kernel_dim,
                    });
                    self.import_linear_state(layer, &mut state)?;
                }
                other => return Err(Error::unsupported(format!("mixer {other}"))),
            }
        }
        Ok((mixers, state))
    }
    fn import_linear_state(&self, layer: usize, state: &mut Vec<StateRequirement>) -> Result<()> {
        let elements = self
            .linear_num_value_heads
            .checked_mul(self.linear_key_head_dim)
            .and_then(|v| v.checked_mul(self.linear_value_head_dim))
            .ok_or_else(|| Error::invalid("linear state shape overflow"))?;
        state.push(StateRequirement {
            layer,
            kind: StateKind::LinearAttention,
            dtype: DType::F32,
            elements,
            per_token: false,
        });
        let channels = self
            .linear_num_key_heads
            .checked_mul(self.linear_key_head_dim)
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| {
                self.linear_num_value_heads
                    .checked_mul(self.linear_value_head_dim)
                    .and_then(|w| v.checked_add(w))
            })
            .ok_or_else(|| Error::invalid("convolution shape overflow"))?;
        let elements = channels
            .checked_mul(
                self.linear_conv_kernel_dim
                    .checked_sub(1)
                    .ok_or_else(|| Error::invalid("empty convolution kernel"))?,
            )
            .ok_or_else(|| Error::invalid("convolution state overflow"))?;
        state.push(StateRequirement {
            layer,
            kind: StateKind::Conv,
            dtype: DType::Bf16,
            elements,
            per_token: false,
        });
        Ok(())
    }
}
fn validate_vision(vision: Option<&VisionConfig>, hidden_size: usize) -> Result<()> {
    if let Some(vision) = vision
        && ([
            vision.depth,
            vision.hidden_size,
            vision.out_hidden_size,
            vision.num_heads,
            vision.patch_size,
            vision.temporal_patch_size,
            vision.spatial_merge_size,
        ]
        .contains(&0)
            || vision.out_hidden_size != hidden_size)
    {
        return Err(Error::invalid("invalid vision encoder configuration"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn imports_pinned_qwen38_hybrid_architecture() {
        let imported = QwenProvider
            .import_manifest(
                ModelId::new(1).unwrap(),
                include_bytes!("../../../../examples/qwen3.8-27b/config.json"),
            )
            .unwrap();
        assert_eq!(imported.model.mixers.len(), 64);
        assert_eq!(
            imported
                .model
                .mixers
                .iter()
                .filter(|m| matches!(m, Mixer::Attention { .. }))
                .count(),
            16
        );
        assert_eq!(
            imported
                .model
                .state
                .iter()
                .filter(|s| s.kind == StateKind::LinearAttention)
                .count(),
            48
        );
        assert_eq!(imported.model.hidden_size, 5120);
        assert_eq!(imported.model.vocab_size, 248_320);
        assert!(imported.vision.is_some());
        assert_eq!(imported.mtp_layers, 1);
    }
    #[test]
    fn official_bf16_package_does_not_fit_32_gib() {
        let model = QwenProvider
            .import(
                ModelId::new(1).unwrap(),
                include_bytes!("../../../../examples/qwen3.8-27b/config.json"),
            )
            .unwrap();
        let index = crate::SafetensorsIndex::parse(include_bytes!(
            "../../../../examples/qwen3.8-27b/model.safetensors.index.json"
        ))
        .unwrap();
        let budget = crate::memory_estimate(
            &model,
            index.weight_bytes().unwrap(),
            4096,
            1,
            1 << 30,
            32 << 30,
        )
        .unwrap();
        assert_eq!(index.weight_bytes().unwrap(), 55_562_855_904);
        assert!(!budget.fits);
        assert!(budget.state_bytes > 0);
    }
    #[test]
    fn rejects_shard_path_traversal() {
        let bytes =
            br#"{"metadata":{"total_size":4},"weight_map":{"weight":"../weight.safetensors"}}"#;
        assert!(crate::SafetensorsIndex::parse(bytes).is_err());
    }
}
