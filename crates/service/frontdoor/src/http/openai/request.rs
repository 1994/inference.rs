use infer_core::{Error, Result};
use infer_models::ChatMessage;

/// Maximum chat messages accepted in one generation request.
const MAX_CHAT_MESSAGES: usize = 256;
/// Default completion-token budget when a request omits an explicit limit.
const DEFAULT_MAX_COMPLETION_TOKENS: usize = 16;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GenerationRequest {
    pub model: String,
    pub messages: Option<Vec<ChatMessage>>,
    pub prompt: Option<String>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub seed: Option<u64>,
    pub stream: Option<bool>,
    pub n: Option<usize>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub enable_thinking: Option<bool>,
}

impl GenerationRequest {
    pub(super) fn validate(&self, chat: bool) -> Result<usize> {
        if self.model.is_empty() {
            return Err(Error::invalid("model must not be empty"));
        }
        if chat {
            if self.prompt.is_some() || self.messages.as_ref().is_none_or(Vec::is_empty) {
                return Err(Error::invalid(
                    "chat requires nonempty messages and no prompt",
                ));
            }
            if self.messages.as_ref().is_some_and(|messages| {
                messages.len() > MAX_CHAT_MESSAGES
                    || messages
                        .iter()
                        .any(|m| !["system", "user", "assistant"].contains(&m.role.as_str()))
            }) {
                return Err(Error::invalid(
                    "at most 256 text messages with system/user/assistant roles are supported",
                ));
            }
        } else if self.messages.is_some()
            || self.prompt.as_ref().is_none_or(String::is_empty)
            || self.max_completion_tokens.is_some()
        {
            return Err(Error::invalid(
                "completions require a nonempty string prompt; use max_tokens",
            ));
        }
        if self.max_tokens.is_some() && self.max_completion_tokens.is_some() {
            return Err(Error::invalid("provide only one token limit"));
        }
        let tokens = self
            .max_completion_tokens
            .or(self.max_tokens)
            .unwrap_or(DEFAULT_MAX_COMPLETION_TOKENS);
        if tokens == 0
            || self.n == Some(0)
            || self
                .temperature
                .is_some_and(|t| !t.is_finite() || !(0.0..=2.0).contains(&t))
            || self
                .top_p
                .is_some_and(|p| !p.is_finite() || !(0.0..=1.0).contains(&p))
        {
            return Err(Error::invalid("invalid generation parameters"));
        }
        if self.stream == Some(true) || self.n.is_some_and(|n| n != 1) {
            return Err(Error::unsupported(
                "only stream=false and n=1 are supported",
            ));
        }
        Ok(tokens)
    }

    pub(super) fn preparation_bytes(&self, generated: usize) -> Result<usize> {
        let text = self
            .messages
            .as_ref()
            .map_or(Some(0), |messages| {
                messages.iter().try_fold(0usize, |sum, message| {
                    sum.checked_add(message.content.len())
                })
            })
            .and_then(|bytes| bytes.checked_add(self.prompt.as_ref().map_or(0, String::len)));
        text.and_then(|n| n.checked_mul(crate::constants::TEXT_STAGING_EXPANSION))
            .and_then(|n| n.checked_add(crate::constants::TEXT_STAGING_OVERHEAD_BYTES))
            .and_then(|n| {
                generated
                    .checked_mul(crate::constants::GENERATED_TOKEN_STAGING_BYTES)
                    .and_then(|tail| n.checked_add(tail))
            })
            .ok_or_else(|| Error::invalid("text preparation budget overflow"))
    }
}
