//! Text HTTP adapter.
use super::{HttpError, HttpResult, RuntimeHandle, cpu, router, trace_parent};
use axum::{
    Json, Router, extract::DefaultBodyLimit, extract::State, http::HeaderMap, routing::post,
};
use infer_core::{Error, ErrorCode, RequestId, Result};
use infer_ir::{CanonicalRequest, WorkloadOutput};
use infer_runtime::EngineOutput;
use std::collections::BTreeMap;

#[derive(Clone)]
pub(super) struct TextState {
    handle: RuntimeHandle,
    assets: std::sync::Arc<infer_models::TextAssets>,
    delivery: cpu::CpuPool,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTextRequest {
    pub id: RequestId,
    pub model: infer_core::ModelId,
    pub prompt: Option<String>,
    pub messages: Option<Vec<infer_models::ChatMessage>>,
    pub workload: infer_ir::Workload,
    #[serde(default)]
    pub qos: infer_ir::Qos,
    #[serde(default)]
    pub sampling: infer_ir::Sampling,
    #[serde(default)]
    pub options: infer_models::ChatOptions,
}
pub fn router_with_text(
    handle: RuntimeHandle,
    assets: std::sync::Arc<infer_models::TextAssets>,
) -> Router {
    let state = TextState {
        handle: handle.clone(),
        assets,
        delivery: handle.delivery_pool(),
    };
    router(handle).merge(
        Router::new()
            .route("/native/v1/text", post(text_infer))
            .layer(DefaultBodyLimit::max(2 * 1024 * 1024))
            .with_state(state),
    )
}
pub(super) fn text_preparation_bytes(request: &NativeTextRequest) -> Result<usize> {
    let bytes = request
        .prompt
        .as_ref()
        .map_or(0, String::len)
        .checked_add(
            request
                .messages
                .as_ref()
                .map_or(0, |messages| messages.iter().map(|m| m.content.len()).sum()),
        )
        .ok_or_else(|| Error::invalid("text preparation size overflow"))?;
    bytes
        .checked_mul(16)
        .and_then(|n| n.checked_add(4096))
        .ok_or_else(|| Error::invalid("text staging size overflow"))
}

pub(super) async fn text_infer(
    State(state): State<TextState>,
    headers: HeaderMap,
    request: std::result::Result<Json<NativeTextRequest>, axum::extract::rejection::JsonRejection>,
) -> HttpResult<Json<serde_json::Value>> {
    let Json(request) = request.map_err(HttpError::from)?;
    let assets = state.assets.clone();
    let bytes = text_preparation_bytes(&request)?;
    let mut receiver = state
        .handle
        .submit_preparing_until(
            request.id,
            bytes,
            trace_parent(&headers),
            request.qos.deadline_us,
            move |context| {
                context.check()?;
                let tokens = match (request.prompt, request.messages) {
                    (Some(prompt), None) => assets.encode(&prompt, true)?,
                    (None, Some(messages)) => assets.encode_chat(&messages, &request.options)?,
                    _ => return Err(Error::invalid("provide exactly one of prompt/messages")),
                };
                Ok(CanonicalRequest {
                    id: request.id,
                    model: request.model,
                    session: None,
                    input: infer_ir::RequestInput::Sequence {
                        tokens: tokens.into(),
                        media: vec![],
                    },
                    workload: request.workload,
                    qos: request.qos,
                    sampling: request.sampling,
                    extensions: BTreeMap::new(),
                })
            },
        )
        .await?;
    while let Some(event) = receiver.recv().await {
        if let EngineOutput::Finished(result) = event? {
            let text = match &result.output {
                Some(WorkloadOutput::Tokens(tokens)) => Some({
                    let assets = state.assets.clone();
                    let tokens = tokens.clone();
                    state
                        .delivery
                        .run(tokens.len().saturating_mul(8), move |_| {
                            assets.decode(&tokens, true)
                        })
                        .await?
                }),
                _ => None,
            };
            return Ok(Json(
                serde_json::json!({"result":result,"text":text,"tokenizer_fingerprint":state.assets.fingerprint}),
            ));
        }
    }
    Err(Error::new(ErrorCode::Backend, "text stream closed before completion").into())
}
