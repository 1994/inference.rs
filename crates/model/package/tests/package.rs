use infer_core::*;
use infer_ir::*;
use infer_models::*;
use std::{collections::BTreeMap, path::PathBuf, sync::atomic::AtomicU64, sync::atomic::Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn temp() -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "infer-tensor-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}
#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn raw(header: &str, data: &[u8]) -> PathBuf {
    let root = temp();
    let p = root.join("model.safetensors");
    let mut b = (header.len() as u64).to_le_bytes().to_vec();
    b.extend(header.as_bytes());
    b.extend(data);
    std::fs::write(&p, b).unwrap();
    p
}
#[test]
fn safetensors_float_conversion_and_row_read_are_exact() {
    let bf16 = raw(
        r#"{"w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}"#,
        &[0x80, 0x3f, 0, 0xc0, 0, 0x3f, 0, 0],
    );
    let mut file = SafetensorsFile::open(&bf16).unwrap();
    assert_eq!(
        file.read_f32("w", 1024).unwrap().data,
        [1.0, -2.0, 0.5, 0.0]
    );
    assert_eq!(file.read_row_f32("w", 1, 1024).unwrap(), [0.5, 0.0]);
    assert_eq!(file.read_f32("w", 1).unwrap_err().code, ErrorCode::Capacity);
    let f16 = raw(
        r#"{"w":{"dtype":"F16","shape":[4],"data_offsets":[0,8]}}"#,
        &[0, 0x3c, 0, 0xc0, 1, 0, 0, 0x80],
    );
    let out = SafetensorsFile::open(&f16)
        .unwrap()
        .read_f32("w", 1024)
        .unwrap();
    assert_eq!(out.data, [1.0, -2.0, 2.0f32.powi(-24), -0.0]);
    std::fs::remove_dir_all(bf16.parent().unwrap()).unwrap();
    std::fs::remove_dir_all(f16.parent().unwrap()).unwrap();
}
#[test]
fn malformed_header_bounds_duplicates_and_nonfinite_payload_are_rejected() {
    for header in [
        r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
        r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
        r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}}"#,
    ] {
        let p = raw(header, &[0; 8]);
        assert!(SafetensorsFile::open(&p).is_err());
        std::fs::remove_dir_all(p.parent().unwrap()).unwrap();
    }
    let p = raw(
        r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        &f32::NAN.to_le_bytes(),
    );
    assert!(
        SafetensorsFile::open(&p)
            .unwrap()
            .read_f32("w", 1024)
            .is_err()
    );
    std::fs::remove_dir_all(p.parent().unwrap()).unwrap();
}
#[test]
fn host_package_binds_payload_and_respects_budget() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let mut package = QwenPackage::open(&root, ModelId::new(1).unwrap()).unwrap();
    assert!(
        package
            .manifest
            .bindings
            .iter()
            .any(|b| b.source.contains("A_log"))
    );
    assert!(
        package
            .manifest
            .bindings
            .iter()
            .any(|b| b.shape == vec![16, 8])
    );
    assert_eq!(
        package.load_host_weights(1).unwrap_err().code,
        ErrorCode::Capacity
    );
    let loaded = package.load_host_weights(1024 * 1024).unwrap();
    assert_eq!(loaded.len(), package.manifest.bindings.len());
    assert!(loaded.values().all(|t| t.validate().is_ok()));
}
#[test]
fn qwen38_required_bindings_exist_in_pinned_official_index() {
    let config = include_bytes!("../../../../examples/qwen3.8-27b/config.json");
    let model = QwenProvider
        .import_manifest(ModelId::new(1).unwrap(), config)
        .unwrap()
        .model;
    let index = SafetensorsIndex::parse(include_bytes!(
        "../../../../examples/qwen3.8-27b/model.safetensors.index.json"
    ))
    .unwrap();
    let graph = infer_compiler::dataflow::lower(&model).unwrap();
    let expected: BTreeMap<_, _> = graph
        .tensors
        .iter()
        .filter_map(|s| match &s.storage {
            TensorStorage::Weight { slot } => Some((
                if slot == "lm_head.weight" {
                    slot.clone()
                } else {
                    format!("model.language_model.{slot}")
                },
                s.shape.clone(),
            )),
            _ => None,
        })
        .collect();
    for name in expected.keys() {
        assert!(
            index.weight_map.contains_key(name),
            "missing official binding {name}"
        );
    }

    assert_eq!(
        expected.len(),
        index
            .weight_map
            .keys()
            .filter(|s| s.starts_with("model.language_model.") || s.as_str() == "lm_head.weight")
            .count(),
        "every backbone weight must be consumed"
    );
}
#[test]
fn official_qwen_text_parity_when_assets_are_supplied() {
    let Ok(root) = std::env::var("INFER_QWEN_TEXT_PACKAGE") else {
        return;
    };
    let assets = TextAssets::open(root, 262_144).unwrap();
    let golden: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../../examples/qwen3.8-27b/text-golden.json"
    ))
    .unwrap();
    for case in golden["cases"].as_array().unwrap() {
        let messages: Vec<ChatMessage> = serde_json::from_value(case["messages"].clone()).unwrap();
        let options = ChatOptions {
            enable_thinking: case["enable_thinking"].as_bool().unwrap(),
            ..Default::default()
        };
        let rendered = assets.render_chat(&messages, &options).unwrap();
        assert_eq!(rendered, case["rendered"].as_str().unwrap());
        let tokens: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        assert_eq!(assets.encode_chat(&messages, &options).unwrap(), tokens);
    }
}
