use infer_core::{Error, Result};
use infer_models::{ChatMessage, ToolCall};
use serde_json::Value;
use std::collections::BTreeSet;

/// Maximum chat messages accepted in one generation request.
const MAX_CHAT_MESSAGES: usize = 256;
/// Default completion-token budget when a request omits an explicit limit.
const DEFAULT_MAX_COMPLETION_TOKENS: usize = 16;
/// Largest number of function tools accepted in one request.
const MAX_TOOLS: usize = 128;

/// The generated-token budget a request asked for, before the service cap and remaining
/// context apply.
pub(super) struct RequestedOutput {
    pub(super) tokens: usize,
    /// True when the client set `max_tokens` or `max_completion_tokens` explicitly.
    pub(super) explicit: bool,
}

/// Tool calling enabled by one request.
///
/// The first delivery supports `none` and `auto` by parsing a completed response. `required`,
/// a named function and `strict` schemas need generation-time constraints, so they are refused
/// explicitly rather than approximated after the fact.
#[derive(Debug)]
pub(super) enum ToolPolicy {
    /// No `tools`, or `tool_choice: "none"`. Structured calls are not published.
    Disabled,
    /// Calls are parsed from the completed response and published when they name a declared tool.
    Auto {
        declared: BTreeSet<String>,
        /// The declared tools as the chat template expects them.
        template: Value,
    },
}

impl ToolPolicy {
    /// Declared names for the parser, if any call may be published.
    pub(super) const fn declared(&self) -> Option<&BTreeSet<String>> {
        match self {
            Self::Disabled => None,
            Self::Auto { declared, .. } => Some(declared),
        }
    }
    /// Declared tools for the chat template.
    pub(super) fn template(&self) -> Option<Value> {
        match self {
            Self::Disabled => None,
            Self::Auto { template, .. } => Some(template.clone()),
        }
    }
    /// Whether a completed response should be parsed for calls.
    pub(super) const fn parses(&self) -> bool {
        matches!(self, Self::Auto { .. })
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GenerationRequest {
    pub model: String,
    pub messages: Option<Vec<WireMessage>>,
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
    pub tools: Option<Vec<WireTool>>,
    pub tool_choice: Option<WireToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub stream_options: Option<StreamOptions>,
}

/// Streaming modifiers accepted alongside `stream=true`.
#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

impl GenerationRequest {
    pub(super) fn validate(&self, chat: bool) -> Result<RequestedOutput> {
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
                    || messages.iter().any(|m| {
                        !["system", "user", "assistant", "tool"].contains(&m.role.as_str())
                    })
            }) {
                return Err(Error::invalid(
                    "at most 256 text messages with system/user/assistant/tool roles are supported",
                ));
            }
        } else if self.messages.is_some()
            || self.prompt.as_ref().is_none_or(String::is_empty)
            || self.max_completion_tokens.is_some()
            || self.tools.is_some()
            || self.tool_choice.is_some()
            || self.parallel_tool_calls.is_some()
            || self.stream_options.is_some()
        {
            return Err(Error::invalid(
                "completions require a nonempty string prompt; use max_tokens",
            ));
        }
        if self.max_tokens.is_some() && self.max_completion_tokens.is_some() {
            return Err(Error::invalid("provide only one token limit"));
        }
        let explicit = self.max_tokens.is_some() || self.max_completion_tokens.is_some();
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
        if self.n.is_some_and(|n| n != 1) {
            return Err(Error::unsupported("only n=1 is supported"));
        }
        if self.stream_options.is_some() && self.stream != Some(true) {
            return Err(Error::invalid("stream_options requires stream=true"));
        }
        Ok(RequestedOutput { tokens, explicit })
    }

    /// Resolve the request's tool-calling mode.
    ///
    /// # Errors
    /// Rejects a mode that needs generation-time constraints, a malformed declaration, or a
    /// `tool_choice` that names a tool the request did not declare.
    pub(super) fn tool_policy(&self, chat: bool) -> Result<ToolPolicy> {
        if !chat {
            return Ok(ToolPolicy::Disabled);
        }
        let declared_tools = self.tools.as_deref().unwrap_or_default();
        if declared_tools.len() > MAX_TOOLS {
            return Err(Error::invalid(format!(
                "at most {MAX_TOOLS} tools may be declared"
            )));
        }
        let mut names = BTreeSet::new();
        let mut template = Vec::with_capacity(declared_tools.len());
        for tool in declared_tools {
            if tool.kind != "function" {
                return Err(Error::unsupported("only function tools are supported"));
            }
            let name = tool.function.name.trim();
            if name.is_empty() {
                return Err(Error::invalid("tool function name must not be empty"));
            }
            if tool.function.strict == Some(true) {
                return Err(Error::unsupported(
                    "strict tool schemas require generation constraints",
                ));
            }
            if !names.insert(name.to_string()) {
                return Err(Error::invalid("tool function names must be unique"));
            }
            template.push(serde_json::json!({
                "type": "function",
                "function": tool.function,
            }));
        }
        // `parallel_tool_calls: false` needs a constraint that limits the call count. Trimming a
        // second call after generation would not satisfy the contract, so it is refused.
        let parallel = self.parallel_tool_calls.unwrap_or(true);
        if !parallel && !declared_tools.is_empty() {
            return Err(Error::unsupported(
                "parallel_tool_calls=false requires generation constraints",
            ));
        }
        // `none` keeps the declared tools in the prompt but publishes no structured call.
        let enabled = match &self.tool_choice {
            None => !declared_tools.is_empty(),
            Some(WireToolChoice::Mode(mode)) => match mode.as_str() {
                "auto" => !declared_tools.is_empty(),
                "none" => false,
                "required" => {
                    return Err(Error::unsupported(
                        "tool_choice=required requires generation constraints",
                    ));
                }
                _ => return Err(Error::invalid("unknown tool_choice mode")),
            },
            Some(WireToolChoice::Named { function, .. }) => {
                return Err(Error::unsupported(format!(
                    "tool_choice naming {:?} requires generation constraints",
                    function.name
                )));
            }
        };
        if !enabled {
            return Ok(ToolPolicy::Disabled);
        }
        Ok(ToolPolicy::Auto {
            declared: names,
            template: Value::Array(template),
        })
    }

    /// Convert the wire messages into the internal history, checking result association.
    ///
    /// # Errors
    /// Rejects unsupported content parts, an assistant call without an identifier, a duplicated
    /// call identifier, and a `tool` result that does not answer an earlier assistant call.
    pub(super) fn chat_messages(&self) -> Result<Vec<ChatMessage>> {
        let wire = self.messages.as_deref().unwrap_or_default();
        let mut call_ids = BTreeSet::new();
        let mut out = Vec::with_capacity(wire.len());
        for message in wire {
            let mut history = ChatMessage::new(message.role.as_str(), message.text()?);
            if !message.tool_calls.is_empty() && message.role != "assistant" {
                return Err(Error::invalid(
                    "only an assistant message may carry tool_calls",
                ));
            }
            for call in &message.tool_calls {
                if call.id.trim().is_empty() {
                    return Err(Error::invalid("tool call id must not be empty"));
                }
                if call.function.name.trim().is_empty() {
                    return Err(Error::invalid("tool call name must not be empty"));
                }
                if call.kind.as_deref().is_some_and(|kind| kind != "function") {
                    return Err(Error::unsupported("only function tool calls are supported"));
                }
                if !call_ids.insert(call.id.clone()) {
                    return Err(Error::invalid("tool call ids must be unique"));
                }
                history.tool_calls.push(ToolCall {
                    id: call.id.clone(),
                    name: call.function.name.clone(),
                    arguments: call.function.arguments.clone(),
                });
            }
            if let Some(id) = &message.tool_call_id {
                if message.role != "tool" {
                    return Err(Error::invalid("only a tool message may carry tool_call_id"));
                }
                // A result must answer a call the same history contains. History may reference
                // tools that are no longer declared; it just cannot invent an association.
                if !call_ids.contains(id) {
                    return Err(Error::invalid(
                        "tool result must answer an earlier assistant tool call",
                    ));
                }
                history.tool_call_id = Some(id.clone());
            } else if message.role == "tool" {
                return Err(Error::invalid("a tool message requires tool_call_id"));
            }
            out.push(history);
        }
        if out.is_empty() {
            return Err(Error::invalid("chat requires nonempty messages"));
        }
        Ok(out)
    }

    pub(super) fn preparation_bytes(&self, generated: usize) -> Result<usize> {
        let text = self
            .messages
            .as_ref()
            .map_or(Some(0), |messages| {
                messages
                    .iter()
                    .try_fold(0usize, |sum, message| sum.checked_add(message.text_len()))
            })
            .and_then(|bytes| bytes.checked_add(self.prompt.as_ref().map_or(0, String::len)))
            .and_then(|bytes| {
                bytes.checked_add(self.tools.as_deref().map_or(0, |tools| {
                    tools.iter().fold(0usize, |sum, tool| {
                        sum.saturating_add(serde_json::to_string(tool).map_or(0, |s| s.len()))
                    })
                }))
            });
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

/// One assistant or tool message on the wire.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub tool_calls: Vec<WireToolCall>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

impl WireMessage {
    /// The message text parts in order, validating each one.
    fn parts(&self) -> Result<Vec<&str>> {
        match &self.content {
            None | Some(Value::Null) => Ok(Vec::new()),
            Some(Value::String(text)) => Ok(vec![text.as_str()]),
            Some(Value::Array(parts)) => parts
                .iter()
                .enumerate()
                .map(|(index, part)| text_part(part, index))
                .collect(),
            Some(_) => Err(Error::invalid(
                "message content must be a string, null, or a list of text parts",
            )),
        }
    }
    /// Total text and call bytes charged to the staging budget.
    fn text_len(&self) -> usize {
        let content = self
            .parts()
            .map_or(0, |parts| parts.iter().map(|text| text.len()).sum());
        self.tool_calls.iter().fold(content, |sum, call| {
            sum.saturating_add(call.id.len())
                .saturating_add(call.function.name.len())
                .saturating_add(call.function.arguments.len())
        })
    }
    /// The message text. `null` and absent both mean no text.
    fn text(&self) -> Result<String> {
        Ok(self.parts()?.concat())
    }
}

/// Validate one content part and return its text.
fn text_part(part: &Value, index: usize) -> Result<&str> {
    let object = part
        .as_object()
        .ok_or_else(|| Error::invalid(format!("message content part {index} must be an object")))?;
    // The part kind is checked before its fields so an image or audio part reports the missing
    // capability rather than an incidental unknown field.
    match object.get("type").and_then(Value::as_str) {
        Some("text") => {}
        Some(_) => {
            return Err(Error::unsupported(
                "only text message content parts are supported",
            ));
        }
        None => {
            return Err(Error::invalid(format!(
                "message content part {index} requires a type"
            )));
        }
    }
    if let Some(key) = object
        .keys()
        .find(|key| !["type", "text"].contains(&key.as_str()))
    {
        return Err(Error::invalid(format!(
            "unknown message content part field {key:?}"
        )));
    }
    object.get("text").and_then(Value::as_str).ok_or_else(|| {
        Error::invalid(format!(
            "text content part {index} requires a string text field"
        ))
    })
}

/// An assistant tool call in the history.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireToolCall {
    pub id: String,
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    pub function: WireToolCallFunction,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireToolCallFunction {
    pub name: String,
    /// Raw JSON text exactly as the caller received it.
    pub arguments: String,
}

/// One declared function tool.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: WireToolFunction,
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireToolFunction {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// `tool_choice` as either a mode string or a named function.
#[derive(serde::Deserialize)]
#[serde(untagged)]
pub(super) enum WireToolChoice {
    Mode(String),
    Named {
        #[serde(rename = "type")]
        _kind: String,
        function: WireToolName,
    },
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireToolName {
    pub name: String,
}

#[cfg(test)]
#[path = "request/tests.rs"]
mod tests;
