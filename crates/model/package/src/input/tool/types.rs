/// Opening marker of a tool-call block.
pub const TOOL_CALL_OPEN: &str = "<tool_call>";
/// Closing marker of a tool-call block.
pub const TOOL_CALL_CLOSE: &str = "</tool_call>";
/// Opening marker of a reasoning block; its token identity belongs to the package.
pub const REASONING_OPEN: &str = "\u{3c}think\u{3e}";
/// Closing marker of a reasoning block; its token identity belongs to the package.
pub const REASONING_CLOSE: &str = "\u{3c}/think\u{3e}";
/// Marker that identifies the function-parameter dialect inside a template.
pub const FUNCTION_MARKER: &str = "<function=";
/// Marker delimiting one named parameter of the function-parameter dialect.
pub const PARAMETER_MARKER: &str = "<parameter=";

/// Largest number of tool calls published from one response.
pub const MAX_TOOL_CALLS: usize = 128;
/// Largest accepted JSON argument payload for one tool call.
pub const MAX_TOOL_ARGUMENT_BYTES: usize = 64 * 1024;

/// Tool-call dialect of one model package, detected from its chat template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolDialect {
    /// One JSON object per `<tool_call>` block; emitted by the `Qwen3-VL` templates.
    JsonBlock,
    /// A `<function=...>` block with `<parameter=...>` children; emitted by the Qwen3.8 template.
    FunctionParameters,
}

impl ToolDialect {
    /// Recognize only templates that declare one of the supported tool envelopes.
    #[must_use]
    pub fn from_template(template: &str) -> Option<Self> {
        if !template.contains(TOOL_CALL_OPEN) {
            return None;
        }
        if template.contains(FUNCTION_MARKER) {
            Some(Self::FunctionParameters)
        } else if template.contains("arguments") && template.contains("name") {
            Some(Self::JsonBlock)
        } else {
            None
        }
    }

    /// Detect the dialect a chat template renders and instructs the model to emit.
    ///
    /// Detection reads the package's own template so the parser cannot drift from the prompt
    /// format: a template that asks for `<function=...>` gets the matching parser.
    #[must_use]
    pub fn detect(template: &str) -> Self {
        if template.contains(FUNCTION_MARKER) {
            Self::FunctionParameters
        } else {
            Self::JsonBlock
        }
    }
}

/// A tool call in the assistant history or in a published response.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCall {
    /// Stable identifier that a later `tool` message associates its result with.
    pub id: String,
    /// Declared function name exactly as the model produced it.
    pub name: String,
    /// Raw JSON argument text, preserved byte-for-byte and never repaired.
    pub arguments: String,
}

impl ToolCall {
    /// The argument object as the chat templates expect it.
    ///
    /// The wire protocol carries arguments as a string, while both shipped templates require a
    /// mapping. Decoding is lossless for valid JSON; an undecodable payload is reported instead
    /// of being handed to a template that would reject it anyway.
    ///
    /// # Errors
    /// Returns an invalid-input error when the arguments are not a JSON object.
    pub fn argument_object(&self) -> infer_core::Result<serde_json::Value> {
        let value: serde_json::Value = serde_json::from_str(&self.arguments).map_err(|error| {
            infer_core::Error::invalid(format!(
                "tool call arguments for {:?} are not JSON: {error}",
                self.name
            ))
        })?;
        if !value.is_object() {
            return Err(infer_core::Error::invalid(format!(
                "tool call arguments for {:?} must be a JSON object",
                self.name
            )));
        }
        Ok(value)
    }
}

/// One call as parsed from the model output, before an identifier is assigned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: String,
}

/// The model output split into publishable parts.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ParsedOutput {
    /// Assistant text with the reasoning and tool-call regions removed.
    pub content: String,
    /// Reasoning text when the dialect emitted a reasoning block.
    pub reasoning: Option<String>,
    /// Parsed calls in output order.
    pub calls: Vec<ParsedToolCall>,
    /// A marker opened without its closing counterpart, so the tail was withheld.
    pub truncated: bool,
}

impl ParsedOutput {
    /// Whether the output contains complete calls.
    #[must_use]
    pub const fn has_calls(&self) -> bool {
        !self.calls.is_empty()
    }
}

/// One published piece of a model response, in output order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    /// Assistant text outside the structured regions.
    Text(String),
    /// Reasoning text from a leading reasoning block.
    Reasoning(String),
    /// A completed tool call.
    Call(ParsedToolCall),
    /// A stable call identity, before all arguments have arrived.
    CallStart { index: usize, name: String },
    /// Append-only JSON argument bytes for a previously started call.
    CallArguments { index: usize, arguments: String },
    /// The complete block and its arguments passed validation.
    CallEnd { index: usize },
}
