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
fn the_target_layer_count_must_match_the_draft() {
    let draft = DraftGeometry::official();
    // The largest official tap is 61, but the layer count is checked first: a 61-layer target is
    // not the 64-layer target the draft was trained against.
    let error = draft.check_target(&decoder(5120, 61)).unwrap_err();
    assert!(error.to_string().contains("64 target layers"), "{error}");
}

#[test]
fn every_tap_must_name_a_target_attention_layer() {
    let draft = DraftGeometry::official();
    let mut target = decoder(5120, 64);
    // A recurrent layer at a tapped index feeds no attention hidden state.
    target.mixers[61] = Mixer::Recurrent { state_width: 128 };
    let error = draft.check_target(&target).unwrap_err();
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

fn fixture() -> DFlash2Config {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/qwen3.8-27b-dflash2/config.json");
    DFlash2Config::parse(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn the_registered_fixture_parses_to_the_official_geometry() {
    let config = fixture();
    assert_eq!(config.geometry(), DraftGeometry::official());
    config.geometry().validate().unwrap();
    config.check_block_semantics().unwrap();
}

#[test]
fn the_inventory_covers_the_released_draft() {
    let config = fixture();
    let expected = config.expected_weight_shapes();
    // The released checkpoint has 81 tensors: six shared and fifteen per draft layer.
    assert_eq!(expected.len(), 6 + 15 * config.num_hidden_layers);
    assert_eq!(expected["fc.weight"], Some(vec![5120, 25600]));
    assert_eq!(
        expected["layers.0.self_attn.q_proj.weight"],
        Some(vec![4096, 5120])
    );
    assert_eq!(
        expected["layers.0.self_attn.k_proj.weight"],
        Some(vec![1024, 5120])
    );
    assert_eq!(
        expected["layers.4.mlp.down_proj.weight"],
        Some(vec![5120, 17408])
    );
    // The convolution shapes come from the loader, not from geometry.
    assert_eq!(expected["layers.0.attention_conv.base_kernel"], None);
}

#[test]
fn a_complete_inventory_is_accepted_and_each_shortfall_is_named() {
    let config = fixture();
    let mut actual: BTreeMap<String, Vec<usize>> = config
        .expected_weight_shapes()
        .into_iter()
        .map(|(name, shape)| (name, shape.unwrap_or_else(|| vec![1280, 5120])))
        .collect();
    config.check_weight_inventory(&actual).unwrap();

    let complete = actual.clone();
    let missing = actual.keys().next().unwrap().clone();
    actual.remove(&missing);
    let error = config.check_weight_inventory(&actual).unwrap_err();
    assert!(error.to_string().contains(&missing), "{error}");

    actual = complete;
    actual.insert("layers.0.mystery.weight".to_owned(), vec![1]);
    let error = config.check_weight_inventory(&actual).unwrap_err();
    assert!(error.to_string().contains("unknown tensor"), "{error}");

    actual.remove("layers.0.mystery.weight");
    actual.insert(
        "layers.0.self_attn.q_proj.weight".to_owned(),
        vec![4096, 4096],
    );
    let error = config.check_weight_inventory(&actual).unwrap_err();
    assert!(error.to_string().contains("q_proj"), "{error}");
}

#[test]
fn a_causal_or_tied_draft_configuration_is_rejected() {
    let mut causal = fixture();
    causal.is_causal = true;
    assert!(causal.check_block_semantics().is_err());
    let mut tied = fixture();
    tied.tie_word_embeddings = true;
    assert!(tied.check_block_semantics().is_err());
}

#[test]
fn a_configuration_for_another_architecture_does_not_parse() {
    let bytes = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../examples/qwen3.8-27b/config.json"),
    )
    .unwrap();
    assert!(DFlash2Config::parse(&bytes).is_err());
}

fn golden() -> serde_json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/qwen3.8-27b-dflash2/fusion-golden.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn the_fusion_golden_matches_the_registered_contract() {
    let config = fixture();
    let geometry = config.geometry();
    let golden = golden();

    // The golden is only usable if it came from the same configuration the fixture registers.
    let digest = {
        use sha2::{Digest, Sha256};
        let bytes = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../examples/qwen3.8-27b-dflash2/config.json"),
        )
        .unwrap();
        format!("{:x}", Sha256::digest(&bytes))
    };
    assert_eq!(golden["model"]["config_sha256"], digest);
    let taps: Vec<usize> = golden["target_layer_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| usize::try_from(value.as_u64().unwrap()).unwrap())
        .collect();
    assert_eq!(taps, geometry.target_taps);
    assert_eq!(
        golden["fused"].as_array().unwrap().len(),
        geometry.hidden_size
    );
    let expected = config.expected_weight_shapes();
    let shape: Vec<usize> = golden["model"]["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| usize::try_from(value.as_u64().unwrap()).unwrap())
        .collect();
    assert_eq!(Some(shape), expected["fc.weight"].clone());
}

#[test]
fn the_fusion_projection_is_recorded_in_float32() {
    let golden = golden();
    assert_eq!(golden["model"]["tensor"], "fc.weight");
    assert_eq!(golden["model"]["dtype"], "BF16");
    assert!(
        golden["formula"]["projection"]
            .as_str()
            .unwrap()
            .contains("no bias"),
        "the golden records the projection's semantics"
    );
    assert!(golden["fused"].as_array().unwrap().iter().all(|value| {
        let value = value.as_f64().unwrap();
        value.is_finite()
    }));
}

fn selector_golden() -> serde_json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/qwen3.8-27b-dflash2/selector-golden.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn the_selector_golden_matches_the_registered_contract() {
    let config = fixture();
    let geometry = config.geometry();
    let golden = selector_golden();
    let expected = config.expected_weight_shapes();

    assert_eq!(
        golden["rank"].as_u64().unwrap(),
        geometry.selector_rank as u64
    );
    assert_eq!(
        golden["top_k"].as_u64().unwrap(),
        geometry.selector_top_k as u64
    );
    for tensor in [
        "candidate_selector.hidden_projection.weight",
        "candidate_selector.predecessor_codebook",
        "candidate_selector.successor_codebook",
    ] {
        let shape: Vec<usize> = golden["model"]["tensors"][tensor]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| usize::try_from(value.as_u64().unwrap()).unwrap())
            .collect();
        assert_eq!(Some(shape), expected[tensor].clone(), "{tensor}");
    }
    let scores = golden["scores"].as_array().unwrap();
    assert_eq!(scores.len(), geometry.selector_top_k);
    for row in scores {
        assert_eq!(row.as_array().unwrap().len(), geometry.selector_top_k);
    }
    let logits = golden["inputs"]["unary_logits"].as_array().unwrap();
    assert_eq!(logits[0], logits[1], "the recorded tie stays visible");
    assert_eq!(
        golden["inputs"]["candidate_ids"].as_array().unwrap().len(),
        geometry.selector_top_k
    );
    assert!(golden["scores"].as_array().unwrap().iter().all(|row| {
        row.as_array()
            .unwrap()
            .iter()
            .all(|value| value.as_f64().unwrap().is_finite())
    }));
}
