use super::{ApiResult, GenerationRequest, TextState, ToolPolicy};
use axum::http::HeaderMap;
use infer_core::{Error, ErrorCode, FinishReason};
use infer_ir::{CanonicalRequest, RequestInput, Workload, WorkloadOutput};
use infer_runtime::{CompletedRequest, EngineOutput};
use serde_json::{Value, json};
use std::{collections::BTreeMap, sync::Arc, sync::atomic::AtomicUsize, sync::atomic::Ordering};

#[expect(
    clippy::too_many_lines,
    reason = "The adapter sequences request validation, limit resolution, preparation and delivery in one place so the admitted request has a single owner"
)]
pub(super) async fn generate(
    state: TextState,
    headers: HeaderMap,
    mut request: GenerationRequest,
    chat: bool,
) -> ApiResult<Value> {
    let requested = request.validate(chat)?;
    let policy = request.tool_policy(chat)?;
    let inspection = state.handle.inspect().await?;
    let limits = inspection.lengths;
    // An explicit budget over the service cap is a parameter error; an omitted budget uses the
    // bounded default and is only lowered by the cap. Neither silently truncates the prompt.
    if requested.explicit && requested.tokens > limits.output_cap {
        return Err(Error::invalid(format!(
            "requested output of {} tokens exceeds the service output cap {}",
            requested.tokens, limits.output_cap
        ))
        .into());
    }
    let max_new_tokens = requested.tokens.min(limits.output_cap);
    let bytes = request.preparation_bytes(max_new_tokens)?;
    if request.model != inspection.model_name {
        return Err(Error::new(ErrorCode::NotFound, "model is not served; see /v1/models").into());
    }
    let model = inspection.model;
    // Convert the wire history before consuming a request identity, so a malformed tool round
    // is a parameter error rather than a failed request.
    let messages = if chat {
        Some(request.chat_messages()?)
    } else {
        None
    };
    let prompt = request.prompt.take();
    let tools = policy.template();
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
    let enable_thinking = resolved.enable_thinking;
    let id = state.handle.allocate_request_id()?;
    let mut receiver = state
        .handle
        .submit_preparing(
            id,
            bytes,
            super::super::trace_parent(&headers),
            move |context| {
                context.check()?;
                let tokens = if let Some(messages) = messages {
                    assets.encode_chat(
                        &messages,
                        &infer_models::ChatOptions {
                            enable_thinking,
                            tools: tools.clone(),
                            ..Default::default()
                        },
                    )?
                } else {
                    assets.encode(prompt.as_deref().unwrap_or_default(), true)?
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
                &policy,
                enable_thinking,
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

#[expect(
    clippy::too_many_arguments,
    reason = "The response assembles one protocol object from the request context it was generated under"
)]
async fn response(
    state: &TextState,
    result: CompletedRequest,
    prompt_tokens: usize,
    model: &str,
    created: u64,
    chat: bool,
    policy: &ToolPolicy,
    enable_thinking: bool,
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
    // Text is only split when the request could have produced a structured region: a declared
    // tool set for calls, or thinking mode for reasoning. Otherwise the decoded text is the
    // content, exactly as before.
    let choice = if chat {
        chat_choice(
            result.request,
            &text,
            state.assets.tool_dialect(),
            policy,
            enable_thinking,
            reason,
        )?
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

/// Assemble one chat choice from a decoded completion.
///
/// # Errors
/// Returns an invalid-input error when the response must be parsed but its generated region does
/// not decode in the package dialect.
fn chat_choice(
    request: infer_core::RequestId,
    text: &str,
    dialect: infer_models::ToolDialect,
    policy: &ToolPolicy,
    enable_thinking: bool,
    reason: &'static str,
) -> infer_core::Result<Value> {
    let parsed = if policy.parses() || enable_thinking {
        infer_models::parse_model_output_with(text, dialect, policy.declared())?
    } else {
        infer_models::ParsedOutput {
            content: text.to_string(),
            ..Default::default()
        }
    };
    let infer_models::ParsedOutput {
        content,
        reasoning,
        calls,
        ..
    } = parsed;
    let tool_calls: Vec<Value> = if policy.parses() {
        calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                json!({
                    // Stable and unique per response, so a later `tool` result can name it.
                    "id": format!("call_{}_{}", request.get(), index),
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                })
            })
            .collect()
    } else {
        Vec::new()
    };
    let content = if content.is_empty() && !tool_calls.is_empty() {
        Value::Null
    } else {
        Value::String(content)
    };
    let mut message = json!({"role":"assistant", "content":content, "refusal":null});
    if !tool_calls.is_empty() {
        message["tool_calls"] = json!(tool_calls);
    }
    if let Some(reasoning) = reasoning {
        message["reasoning_content"] = json!(reasoning);
    }
    // A length stop stays `length` even when a complete call preceded the cap; the client
    // decides from the terminal reason whether to execute it.
    let finish = if tool_calls.is_empty() || reason == "length" {
        reason
    } else {
        "tool_calls"
    };
    Ok(json!({"index":0, "message":message, "finish_reason":finish, "logprobs":null}))
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
    use infer_models::ToolDialect;
    use std::collections::BTreeSet;

    fn policy(names: &[&str]) -> ToolPolicy {
        ToolPolicy::Auto {
            declared: names
                .iter()
                .map(|name| (*name).to_string())
                .collect::<BTreeSet<_>>(),
            template: json!([]),
        }
    }

    fn choice(text: &str, policy: &ToolPolicy, reason: &'static str) -> Value {
        chat_choice(
            infer_core::RequestId::new(7).unwrap(),
            text,
            ToolDialect::FunctionParameters,
            policy,
            false,
            reason,
        )
        .unwrap()
    }

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

    #[test]
    fn a_declared_call_becomes_a_structured_choice_with_a_stable_id() {
        let choice = choice(
            "Sure.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
            &policy(&["get_weather"]),
            "stop",
        );
        assert_eq!(choice["finish_reason"], "tool_calls");
        assert_eq!(choice["message"]["content"], "Sure.");
        let call = &choice["message"]["tool_calls"][0];
        assert_eq!(call["id"], "call_7_0");
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "get_weather");
        let arguments: Value =
            serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(arguments["city"], "Paris");
    }

    #[test]
    fn a_call_only_response_carries_null_content() {
        let choice = choice(
            "<tool_call><function=ping>\n</function></tool_call>",
            &policy(&["ping"]),
            "stop",
        );
        assert!(choice["message"]["content"].is_null());
        assert_eq!(
            choice["message"]["tool_calls"][0]["function"]["arguments"],
            "{}"
        );
    }

    #[test]
    fn a_length_stop_keeps_length_even_with_a_published_call() {
        let choice = choice(
            "<tool_call><function=ping>\n</function></tool_call>",
            &policy(&["ping"]),
            "length",
        );
        assert_eq!(choice["finish_reason"], "length");
        assert_eq!(choice["message"]["tool_calls"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn an_undeclared_call_is_not_published() {
        let choice = choice(
            "<tool_call><function=invented>\n</function></tool_call>",
            &policy(&["declared"]),
            "stop",
        );
        assert_eq!(choice["finish_reason"], "stop");
        assert!(choice["message"]["tool_calls"].is_null());
        assert!(
            choice["message"]["content"]
                .as_str()
                .unwrap()
                .contains("invented")
        );
    }

    #[test]
    fn disabled_tool_publishing_leaves_the_markers_in_the_text() {
        let choice = choice(
            "<tool_call><function=ping>\n</function></tool_call>",
            &ToolPolicy::Disabled,
            "stop",
        );
        assert!(choice["message"]["tool_calls"].is_null());
        assert!(
            choice["message"]["content"]
                .as_str()
                .unwrap()
                .contains("<tool_call>")
        );
    }

    #[test]
    fn thinking_is_split_out_and_reasoning_is_published_separately() {
        let text = "\u{3c}think\u{3e}weigh the options\u{3c}/think\u{3e}\n\nThe answer is 17.";
        let parsed =
            infer_models::parse_model_output_with(text, ToolDialect::FunctionParameters, None)
                .unwrap();
        let choice = chat_choice(
            infer_core::RequestId::new(7).unwrap(),
            text,
            ToolDialect::FunctionParameters,
            &ToolPolicy::Disabled,
            true,
            "stop",
        )
        .unwrap();
        assert_eq!(parsed.reasoning.as_deref(), Some("weigh the options"));
        assert_eq!(choice["message"]["content"], "The answer is 17.");
        assert_eq!(choice["message"]["reasoning_content"], "weigh the options");
    }
}
