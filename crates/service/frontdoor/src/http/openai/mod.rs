//! A deliberately bounded subset of the `OpenAI` text generation protocol.
mod error;
mod execution;
mod request;
#[cfg(test)]
mod tests;

use super::text::TextState;
use axum::{Json, Router, extract::State, http::HeaderMap, routing::get, routing::post};
use error::ApiError;
use request::{GenerationRequest, ToolPolicy};
use serde_json::{Value, json};

type ApiResult<T> = Result<T, ApiError>;
type Payload = Result<Json<GenerationRequest>, axum::extract::rejection::JsonRejection>;

pub(super) fn routes() -> Router<TextState> {
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completion))
}

async fn models(State(state): State<TextState>) -> ApiResult<Json<Value>> {
    let model = state.handle.inspect().await?.model.to_string();
    Ok(Json(json!({"object":"list", "data":[{
        "id":model, "object":"model", "created":0, "owned_by":"inference.rs"
    }]})))
}

async fn chat(
    State(state): State<TextState>,
    headers: HeaderMap,
    request: Payload,
) -> ApiResult<Json<Value>> {
    let Json(request) = request.map_err(ApiError::from)?;
    execution::generate(state, headers, request, true)
        .await
        .map(Json)
}

async fn completion(
    State(state): State<TextState>,
    headers: HeaderMap,
    request: Payload,
) -> ApiResult<Json<Value>> {
    let Json(request) = request.map_err(ApiError::from)?;
    execution::generate(state, headers, request, false)
        .await
        .map(Json)
}
