use super::*;
use std::fs::remove_dir_all;

fn strict_fixture(text: &str) -> (RuntimeHandle, Arc<infer_models::TextAssets>) {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let package = temp_dir().join(format!(
        "infer-strict-{}-{}",
        id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    create_dir_all(&package).unwrap();
    let mut vocab = serde_json::Map::new();
    for byte in 32u8..127 {
        let key = if byte == b' ' {
            "Ġ".into()
        } else {
            char::from(byte).to_string()
        };
        vocab.insert(key, json!(byte - 32));
    }
    vocab.insert("<eos>".into(), json!(95));
    let byte_level =
        json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true});
    write(package.join("tokenizer.json"), serde_json::to_vec(&json!({
        "version":"1.0", "truncation":null, "padding":null,
        "added_tokens":[{"id":95,"content":"<eos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        "normalizer":null,"pre_tokenizer":byte_level,"post_processor":null,"decoder":byte_level,
        "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"vocab":vocab,"merges":[]}
    })).unwrap()).unwrap();
    write(
        package.join("chat_template.jinja"),
        "{# <tool_call> name arguments #}x",
    )
    .unwrap();
    write(
        package.join("generation_config.json"),
        r#"{"eos_token_id":95,"temperature":0.0,"enable_thinking":false}"#,
    )
    .unwrap();
    let mut model = support::model(ModelId::ONE);
    model.vocab_size = 96;
    model.max_sequence = 512;
    let assets = Arc::new(infer_models::TextAssets::open(&package, model.max_sequence).unwrap());
    remove_dir_all(package).unwrap();
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels).unwrap();
    let tokens = text
        .bytes()
        .map(|byte| usize::from(byte - 32))
        .chain([95])
        .collect();
    let engine = Engine::new(
        ProtocolBackend::scripted(tokens, &model).unwrap(),
        model,
        PrecisionPlan::f32(),
        &kernels,
        RuntimeConfig::default(),
    )
    .unwrap();
    (RuntimeHandle::start(engine).unwrap(), assets)
}
fn request(stream: bool) -> Value {
    json!({"model":"1","messages":[{"role":"user","content":"x"}],"max_tokens":128,"temperature":0.0,
        "stream":stream,"tool_choice":"required","parallel_tool_calls":false,
        "tools":[{"type":"function","function":{"name":"ping","strict":true,"parameters":{"type":"object","properties":{},"required":[],"additionalProperties":false}}}]})
}
#[tokio::test]
async fn strict_tools_round_trip_through_runtime_and_stream() {
    let text = r#"<tool_call>{"name":"ping","arguments":{}}</tool_call>"#;
    let (handle, assets) = strict_fixture(text);
    let app = router_with_text(handle.clone(), assets);
    // Repeated requests also exercise cached grammars with independent matcher positions.
    for _ in 0..2 {
        let (status, body) = call(app.clone(), "/v1/chat/completions", Some(request(false))).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls", "{body}");
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{}"
        );
    }
    let (status, body) = call_stream(app, "/v1/chat/completions", request(true)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let events = stream_data(&body);
    assert!(
        events
            .iter()
            .any(|event| event["choices"][0]["finish_reason"] == "tool_calls"),
        "{body}"
    );
    assert_eq!(events.last().unwrap(), "[DONE]");
    let state = handle.inspect().await.unwrap();
    assert_eq!(state.active_requests, 0);
    assert_eq!(state.state.allocated_pages, 0);
    handle.shutdown().await.unwrap();
}
#[tokio::test]
async fn strict_tool_truncation_does_not_claim_a_complete_call() {
    let (handle, assets) = strict_fixture("<tool_call>");
    let mut payload = request(false);
    payload["max_tokens"] = json!(5);
    let (status, body) = call(
        router_with_text(handle.clone(), assets),
        "/v1/chat/completions",
        Some(payload),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert!(body["choices"][0]["message"]["tool_calls"].is_null());
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn unsupported_strict_schema_is_rejected_before_admission() {
    let (handle, assets) = strict_fixture("x");
    let mut payload = request(false);
    payload["tools"][0]["function"]["parameters"]["patternProperties"] = json!({});
    let (status, body) = call(
        router_with_text(handle.clone(), assets),
        "/v1/chat/completions",
        Some(payload),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    assert_eq!(handle.inspect().await.unwrap().active_requests, 0);
    handle.shutdown().await.unwrap();
}
