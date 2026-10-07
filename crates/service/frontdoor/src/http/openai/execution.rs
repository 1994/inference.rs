use super::{ApiResult, GenerationRequest, TextState};
use axum::http::HeaderMap;
use infer_core::{Error, ErrorCode, FinishReason};
use infer_ir::{CanonicalRequest, RequestInput, Workload, WorkloadOutput};
use infer_runtime::{CompletedRequest, EngineOutput};
use serde_json::{Value, json};
use std::{collections::BTreeMap, sync::Arc, sync::atomic::AtomicUsize, sync::atomic::Ordering};

pub(super) async fn generate(
    state: TextState,
    headers: HeaderMap,
    request: GenerationRequest,
    chat: bool,
) -> ApiResult<Value> {
    let max_new_tokens = request.validate(chat)?;
    let bytes = request.preparation_bytes(max_new_tokens)?;
    let model = state.handle.inspect().await?.model;
    if request.model != model.to_string() {
        return Err(Error::new(ErrorCode::NotFound, "model is not served; see /v1/models").into());
    }
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| Error::new(ErrorCode::Backend, e.to_string()))?
        .as_secs();
    let prompt_tokens = Arc::new(AtomicUsize::new(0));
    let count = prompt_tokens.clone();
    let assets = state.assets.clone();
    let resolved = assets
        .generation
        .resolve(&infer_models::SamplingOverrides {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            min_p: request.min_p,
            presence_penalty: request.presence_penalty,
            repetition_penalty: request.repetition_penalty,
            seed: request.seed,
            enable_thinking: request.enable_thinking,
            ..Default::default()
        })?;
    let id = state.handle.allocate_request_id()?;
    let mut receiver = state
        .handle
        .submit_preparing(
            id,
            bytes,
            super::super::trace_parent(&headers),
            move |context| {
                context.check()?;
                let tokens = if let Some(messages) = request.messages {
                    assets.encode_chat(
                        &messages,
                        &infer_models::ChatOptions {
                            enable_thinking: resolved.enable_thinking,
                            ..Default::default()
                        },
                    )?
                } else {
                    assets.encode(request.prompt.as_deref().unwrap_or_default(), true)?
                };
                count.store(tokens.len(), Ordering::Relaxed);
                Ok(CanonicalRequest {
                    id,
                    model,
                    session: None,
                    input: RequestInput::Sequence {
                        tokens: tokens.into(),
                        media: vec![],
                    },
                    workload: Workload::Generate { max_new_tokens },
                    qos: infer_ir::Qos::default(),
                    sampling: resolved.sampling,
                    extensions: BTreeMap::new(),
                })
            },
        )
        .await?;
    while let Some(event) = receiver.recv().await {
        if let EngineOutput::Finished(result) = event? {
            return response(
                &state,
                result,
                prompt_tokens.load(Ordering::Relaxed),
                &model.to_string(),
                created,
                chat,
            )
            .await;
        }
    }
    Err(Error::new(
        ErrorCode::Backend,
        "request stream closed before completion",
    )
    .into())
}

async fn response(
    state: &TextState,
    result: CompletedRequest,
    prompt_tokens: usize,
    model: &str,
    created: u64,
    chat: bool,
) -> ApiResult<Value> {
    let reason = finish_reason(&result.reason)?;
    let Some(WorkloadOutput::Tokens(tokens)) = result.output else {
        return Err(Error::invariant("generation completed without tokens").into());
    };
    let completion_tokens = tokens.len();
    let assets = state.assets.clone();
    let text = state
        .delivery
        .run(
            tokens
                .len()
                .saturating_mul(crate::constants::TOKEN_STAGING_BYTES),
            move |_| assets.decode(&tokens, true),
        )
        .await?;
    let choice = if chat {
        json!({"index":0, "message":{"role":"assistant", "content":text, "refusal":null},
            "finish_reason":reason, "logprobs":null})
    } else {
        json!({"index":0, "text":text, "finish_reason":reason, "logprobs":null})
    };
    let prefix = if chat { "chatcmpl" } else { "cmpl" };
    Ok(json!({
        "id":format!("{prefix}-{}", result.request),
        "object":if chat { "chat.completion" } else { "text_completion" },
        "created":created, "model":model, "choices":[choice],
        "usage":{"prompt_tokens":prompt_tokens, "completion_tokens":completion_tokens,
            "total_tokens":prompt_tokens + completion_tokens}
    }))
}

fn finish_reason(reason: &FinishReason) -> infer_core::Result<&'static str> {
    match reason {
        FinishReason::Length => Ok("length"),
        FinishReason::Eos | FinishReason::Completed => Ok("stop"),
        FinishReason::Failed(message) => Err(Error::new(ErrorCode::Backend, message.clone())),
        FinishReason::Cancelled | FinishReason::Deadline => Err(Error::new(
            ErrorCode::Backend,
            "generation cancelled or deadline exceeded",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_successful_terminal_states_are_completion_choices() {
        assert_eq!(finish_reason(&FinishReason::Length).unwrap(), "length");
        assert_eq!(finish_reason(&FinishReason::Eos).unwrap(), "stop");
        for reason in [
            FinishReason::Cancelled,
            FinishReason::Deadline,
            FinishReason::Failed("device fault".into()),
        ] {
            assert_eq!(finish_reason(&reason).unwrap_err().code, ErrorCode::Backend);
        }
    }
}
