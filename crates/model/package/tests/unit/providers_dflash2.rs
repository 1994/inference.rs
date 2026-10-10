use super::*;

fn geometry(taps: Vec<usize>) -> DraftGeometry {
    DraftGeometry {
        target_taps: taps,
        ..DraftGeometry::official()
    }
}

fn decoder(hidden_size: usize, layers: usize) -> ModelIr {
    ModelIr {
        id: infer_core::ModelId::ONE,
        backbone: BackboneKind::Decoder,
        vocab_size: 32,
        hidden_size,
        max_sequence: 128,
        mixers: vec![
            Mixer::Attention {
                query_heads: 1,
                kv_heads: 1,
                head_dim: hidden_size,
                sliding_window: None,
                output_gate: false,
                qk_norm: false,
            };
            layers
        ],
        feed_forward: infer_ir::FeedForward::Dense { intermediate: 16 },
        position: infer_ir::PositionSpec {
            rope_theta: 10_000.0,
            rotary_fraction: 1.0,
            multimodal_sections: vec![],
            interleaved: false,
        },
        norm_epsilon: 1e-5,
        norm_weight_offset: 0.0,
        heads: vec![infer_ir::Head::LanguageModel],
        modalities: vec![infer_ir::Modality::Text],
        state: vec![],
        tied_embeddings: false,
    }
}

#[test]
fn the_official_geometry_validates() {
    DraftGeometry::official().validate().unwrap();
}

#[test]
fn key_value_heads_must_divide_the_attention_heads() {
    let broken = DraftGeometry {
        attention_heads: 32,
        key_value_heads: 5,
        ..DraftGeometry::official()
    };
    assert!(broken.validate().is_err());
}

#[test]
fn taps_must_be_a_non_empty_ascending_list() {
    assert!(geometry(vec![]).validate().is_err());
    assert!(geometry(vec![5, 5]).validate().is_err());
    assert!(geometry(vec![19, 5]).validate().is_err());
}

#[test]
fn the_selector_may_not_take_more_candidates_than_it_scores() {
    let broken = DraftGeometry {
        selector_top_k: 0,
        ..DraftGeometry::official()
    };
    assert!(broken.validate().is_err());
    let broken = DraftGeometry {
        selector_top_k: 512,
        ..DraftGeometry::official()
    };
    assert!(broken.validate().is_err());
}

#[test]
fn a_block_wider_than_the_window_is_rejected() {
    let broken = DraftGeometry {
        block_size: 4096,
        ..DraftGeometry::official()
    };
    assert!(broken.validate().is_err());
}

#[test]
fn the_target_hidden_size_must_match_the_draft() {
    let draft = DraftGeometry::official();
    draft.check_target(&decoder(5120, 64)).unwrap();
    let error = draft.check_target(&decoder(4096, 64)).unwrap_err();
    assert!(error.to_string().contains("hidden"), "{error}");
}

#[test]
fn every_tap_must_name_a_target_attention_layer() {
    let draft = DraftGeometry::official();
    // The largest official tap is 61, so a 61-layer target has no layer for it.
    let error = draft.check_target(&decoder(5120, 61)).unwrap_err();
    assert!(error.to_string().contains("tap 61"), "{error}");
}

#[test]
fn a_non_decoder_target_is_rejected() {
    let draft = DraftGeometry::official();
    let mut target = decoder(5120, 64);
    target.backbone = BackboneKind::Encoder;
    assert!(draft.check_target(&target).is_err());
}

#[test]
fn an_unsupported_operation_is_named() {
    let draft = DraftGeometry::official();
    let mut missing: BTreeSet<Operation> = REQUIRED_OPERATIONS.into_iter().collect();
    draft.check_operations(&missing).unwrap();
    missing.remove(&Operation::Convolution);
    let error = draft.check_operations(&missing).unwrap_err();
    assert!(error.to_string().contains("Convolution"), "{error}");
}

#[test]
fn the_quote_scales_with_geometry_and_precision() {
    let draft = DraftGeometry::official();
    let quote = draft.quote(2);
    // Five layers, eight KV heads of 128 values, K and V, two bytes each.
    assert_eq!(quote.kv_bytes_per_token, 5 * 8 * 128 * 2 * 2);
    assert_eq!(quote.window_kv_bytes, quote.kv_bytes_per_token * 2048);
    // Five taps of hidden 5120 at two bytes.
    assert_eq!(quote.feature_bytes_per_token, 5 * 5120 * 2);
    let wide = draft.quote(4);
    assert_eq!(wide.kv_bytes_per_token, quote.kv_bytes_per_token * 2);
    assert_eq!(
        wide.feature_bytes_per_token,
        quote.feature_bytes_per_token * 2
    );
}

#[test]
fn another_architecture_does_not_answer_an_explicit_dflash2_request() {
    check_declared_architecture(&[ARCHITECTURE.to_owned()]).unwrap();
    let error = check_declared_architecture(&["Qwen3ForCausalLM".to_owned()]).unwrap_err();
    assert!(error.to_string().contains(ARCHITECTURE), "{error}");
}
