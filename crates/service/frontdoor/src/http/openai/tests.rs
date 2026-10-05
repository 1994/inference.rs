use axum::{Router, body::Body, http::Request, http::StatusCode};
use infer_backend_host::{HostBackend, HostConfig, HostKernels};
use infer_core::ModelId;
use infer_ir::PrecisionPlan;
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, RuntimeConfig};
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};
use tower::ServiceExt;

use crate::{RuntimeHandle, router_with_text};

fn fixture() -> (RuntimeHandle, Arc<infer_models::TextAssets>) {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let mut package = infer_models::QwenPackage::open(&root, ModelId::ONE).unwrap();
    let backend = HostBackend::from_package(&mut package, HostConfig::default()).unwrap();
    let model = backend.model().clone();
    let assets = Arc::new(infer_models::TextAssets::open(root, model.max_sequence).unwrap());
    let mut kernels = KernelRegistry::default();
    kernels.register(&HostKernels).unwrap();
    let engine = Engine::new(
        backend,
        model,
        PrecisionPlan::f32(),
        &kernels,
        RuntimeConfig::default(),
    )
    .unwrap();
    (RuntimeHandle::start(engine).unwrap(), assets)
}

async fn call(app: Router, path: &str, payload: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder().uri(path);
    let request = if let Some(payload) = payload {
        request
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap()
    } else {
        request.body(Body::empty()).unwrap()
    };
    let response = tokio::time::timeout(Duration::from_secs(10), app.oneshot(request))
        .await
        .unwrap()
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn completions_match_native_golden_and_report_usage() {
    let (handle, assets) = fixture();
    let app = router_with_text(handle.clone(), assets);
    let (status, models) = call(app.clone(), "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(models["data"][0]["id"], "1");
    let (status, result) = call(
        app,
        "/v1/completions",
        Some(json!({
            "model":"1", "prompt":"hello world system assistant token8 token13", "max_tokens":5
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{result}");
    assert_eq!(result["object"], "text_completion");
    assert_eq!(
        result["choices"][0]["text"],
        "token25 system system system system"
    );
    assert_eq!(result["choices"][0]["finish_reason"], "length");
    assert_eq!(
        result["usage"],
        json!({"prompt_tokens":6,"completion_tokens":5,"total_tokens":11})
    );
    let inspection = handle.inspect().await.unwrap();
    assert_eq!(inspection.active_requests, 0);
    assert_eq!(inspection.state.allocated_pages, 0);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn chat_matches_native_template_and_ids_survive_router_rebuilds() {
    let (handle, assets) = fixture();
    let messages = json!([{"role":"user","content":"hello world"}]);
    let native = json!({"id":100,"model":1,"messages":messages,
        "workload":{"Generate":{"max_new_tokens":3}}});
    let app = router_with_text(handle.clone(), assets.clone());
    let (status, expected) = call(app.clone(), "/native/v1/text", Some(native)).await;
    assert_eq!(status, StatusCode::OK);
    let payload = json!({"model":"1","messages":messages,"max_completion_tokens":3});
    let rebuilt = router_with_text(handle.clone(), assets);
    let (a, b) = tokio::join!(
        call(app, "/v1/chat/completions", Some(payload.clone())),
        call(rebuilt, "/v1/chat/completions", Some(payload))
    );
    assert_eq!(a.0, StatusCode::OK, "{}", a.1);
    assert_eq!(b.0, StatusCode::OK, "{}", b.1);
    assert_ne!(a.1["id"], b.1["id"]);
    for result in [a.1, b.1] {
        assert_eq!(result["choices"][0]["message"]["content"], expected["text"]);
        assert_eq!(result["choices"][0]["message"]["role"], "assistant");
        assert_eq!(result["object"], "chat.completion");
        assert_eq!(result["usage"]["prompt_tokens"], 4);
        assert!(result["created"].as_u64().unwrap() > 0);
    }
    assert_eq!(handle.allocate_request_id().unwrap().get(), 103);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn invalid_and_unsupported_requests_return_errors_without_admission() {
    let (handle, assets) = fixture();
    let app = router_with_text(handle.clone(), assets);
    let payload =
        json!({"model":"1","messages":[{"role":"user","content":"hello"}],"max_tokens":2});
    let cases = [
        ("model", json!("missing"), StatusCode::NOT_FOUND),
        ("stream", json!(true), StatusCode::NOT_IMPLEMENTED),
        ("n", json!(2), StatusCode::NOT_IMPLEMENTED),
        ("top_p", json!(0.0), StatusCode::BAD_REQUEST),
        ("n", json!(0), StatusCode::BAD_REQUEST),
        ("max_tokens", json!(0), StatusCode::BAD_REQUEST),
        ("temperature", json!(-1), StatusCode::BAD_REQUEST),
        ("temperature", json!(3), StatusCode::BAD_REQUEST),
        ("messages", json!([]), StatusCode::BAD_REQUEST),
        (
            "messages",
            json!([{"role":"tool","content":"hello"}]),
            StatusCode::BAD_REQUEST,
        ),
        ("prompt", json!("hello"), StatusCode::BAD_REQUEST),
        ("tools", json!([]), StatusCode::BAD_REQUEST),
        ("max_completion_tokens", json!(3), StatusCode::BAD_REQUEST),
    ];
    for (field, value, expected) in cases {
        let mut invalid = payload.clone();
        invalid[field] = value;
        let (status, body) = call(app.clone(), "/v1/chat/completions", Some(invalid)).await;
        assert_eq!(status, expected, "{field}: {body}");
        assert!(body["error"]["message"].is_string());
        assert!(body["error"]["type"].is_string());
        assert!(body["error"]["code"].is_string());
    }
    assert_eq!(handle.inspect().await.unwrap().active_requests, 0);
    assert_eq!(handle.allocate_request_id().unwrap().get(), 1);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn malformed_and_oversized_bodies_use_the_compatibility_error_envelope() {
    let (handle, assets) = fixture();
    let app = router_with_text(handle.clone(), assets);
    for (body, expected) in [
        ("{".into(), StatusCode::BAD_REQUEST),
        (
            " ".repeat(2 * 1024 * 1024 + 1),
            StatusCode::PAYLOAD_TOO_LARGE,
        ),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), expected);
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        let error: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(error["error"]["type"], "invalid_request_error");
    }
    handle.shutdown().await.unwrap();
}
