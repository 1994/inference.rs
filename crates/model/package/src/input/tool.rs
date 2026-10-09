//! Model output dialect: tool calls and reasoning markers.
//!
//! The prompt-side template and the output-side parser must agree on one dialect, so the markers
//! here are bound to the model package rather than to the HTTP adapter. Two dialects ship with
//! the served packages and are detected from the template itself:
//!
//! ```text
//! JSON block (Qwen3-VL):
//!   <tool_call>
//!   {"name": <function-name>, "arguments": <args-json-object>}
//!   </tool_call>
//!
//! Function parameters (Qwen3.8):
//!   <tool_call>
//!   <function=<function-name>>
//!   <parameter=<name>>
//!   <value>
//!   </parameter>
//!   </function>
//!   </tool_call>
//! ```
//!
//! Parsing never repairs the model's output: an unterminated block or an argument payload that
//! does not decode is reported as truncated or left in the text instead of being completed with
//! invented characters. Callers decide whether that becomes a parameter error or plain text.

/// Opening marker of a tool-call block.
pub const TOOL_CALL_OPEN: &str = "<tool_call>";
/// Closing marker of a tool-call block.
pub const TOOL_CALL_CLOSE: &str = "</tool_call>";
/// Opening marker of a reasoning block, token 151667 of the target package.
pub const REASONING_OPEN: &str = "\u{3c}think\u{3e}";
/// Closing marker of a reasoning block, token 151668 of the target package.
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
    /// Whether any structural region was withheld because its closing marker never arrived.
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
}

/// Parser phase for one streamed response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Nothing published yet; deciding between a reasoning block and plain text.
    Start,
    Reasoning,
    Text,
    Call,
    Done,
}

/// Incremental parser for one model response.
///
/// `feed` consumes only newly committed text and never revisits it, so parsing a response costs
/// one pass over its length. Text is published lazily: a trailing whitespace run and a suffix
/// that could still become a marker are held back. That is what makes the concatenated deltas
/// equal the non-streaming result exactly, rather than approximately, and it is why
/// [`parse_with`] is defined as this parser fed the whole response.
#[derive(Debug, Clone)]
pub struct OutputStreamParser {
    dialect: ToolDialect,
    declared: Option<std::collections::BTreeSet<String>>,
    phase: Phase,
    /// Received text not yet published, consumed or held back.
    buffer: String,
    /// Bytes at the start of `buffer` already known not to open the closing marker.
    scanned: usize,
    /// Calls published so far, bounded by `MAX_TOOL_CALLS`.
    calls: usize,
    truncated: bool,
    text_published: bool,
    reasoning_published: bool,
}

impl OutputStreamParser {
    /// Create a parser for one response.
    #[must_use]
    pub const fn new(
        dialect: ToolDialect,
        declared: Option<std::collections::BTreeSet<String>>,
    ) -> Self {
        Self {
            dialect,
            declared,
            phase: Phase::Start,
            buffer: String::new(),
            scanned: 0,
            calls: 0,
            truncated: false,
            text_published: false,
            reasoning_published: false,
        }
    }
    /// Feed newly committed text and take whatever is now stable.
    ///
    /// # Errors
    /// Returns an invalid-input error when a tool-call body exceeds `MAX_TOOL_ARGUMENT_BYTES` or
    /// the response publishes more than `MAX_TOOL_CALLS` calls.
    pub fn feed(&mut self, chunk: &str) -> infer_core::Result<Vec<StreamEvent>> {
        self.buffer.push_str(chunk);
        self.drain(false)
    }
    /// Finish the response, publishing the remaining stable text and dropping held whitespace.
    ///
    /// # Errors
    /// Returns the same invalid-input errors as [`Self::feed`].
    pub fn finish(&mut self) -> infer_core::Result<Vec<StreamEvent>> {
        self.drain(true)
    }
    /// True when a structured region opened without closing, so its tail was withheld.
    #[must_use]
    pub const fn truncated(&self) -> bool {
        self.truncated
    }
    fn drain(&mut self, flush: bool) -> infer_core::Result<Vec<StreamEvent>> {
        let mut events = Vec::new();
        loop {
            let progressed = match self.phase {
                Phase::Done => break,
                Phase::Start => self.drain_start(flush),
                Phase::Reasoning => self.drain_reasoning(flush, &mut events),
                Phase::Text => self.drain_text(flush, &mut events),
                Phase::Call => self.drain_call(flush, &mut events)?,
            };
            if !progressed {
                break;
            }
        }
        Ok(events)
    }
    /// Decide between a reasoning block and plain text. Returns whether to keep draining.
    fn drain_start(&mut self, flush: bool) -> bool {
        // Hold while the buffer could still become the reasoning marker.
        if !flush && REASONING_OPEN.starts_with(self.buffer.as_str()) {
            return false;
        }
        if self.buffer.starts_with(REASONING_OPEN) {
            self.buffer.drain(..REASONING_OPEN.len());
            self.phase = Phase::Reasoning;
        } else {
            self.phase = Phase::Text;
        }
        true
    }
    fn drain_reasoning(&mut self, flush: bool, events: &mut Vec<StreamEvent>) -> bool {
        let Some(index) = self.buffer.find(REASONING_CLOSE) else {
            let hold = if flush {
                trailing_whitespace(&self.buffer)
            } else {
                hold_len(&self.buffer, REASONING_CLOSE)
            };
            let end = self.buffer.len() - hold;
            let ready = self.buffer[..end].to_string();
            self.buffer.drain(..end);
            self.publish_reasoning(&ready, events);
            if flush {
                // A reasoning block that never closed leaves no assistant text.
                self.truncated = true;
                self.buffer.clear();
                self.phase = Phase::Done;
            }
            return false;
        };
        let head = self.buffer[..index].trim_end().to_string();
        self.buffer.drain(..index + REASONING_CLOSE.len());
        self.publish_reasoning(&head, events);
        self.phase = Phase::Text;
        true
    }
    fn drain_text(&mut self, flush: bool, events: &mut Vec<StreamEvent>) -> bool {
        if let Some(index) = self.buffer.find(TOOL_CALL_OPEN) {
            let head = self.buffer[..index].trim_end().to_string();
            self.buffer.drain(..index + TOOL_CALL_OPEN.len());
            self.publish_text(&head, events);
            self.scanned = 0;
            self.phase = Phase::Call;
            return true;
        }
        let hold = if flush {
            trailing_whitespace(&self.buffer)
        } else {
            hold_len(&self.buffer, TOOL_CALL_OPEN)
        };
        let end = self.buffer.len() - hold;
        let ready = self.buffer[..end].to_string();
        self.buffer.drain(..end);
        self.publish_text(&ready, events);
        if flush {
            self.buffer.clear();
            self.phase = Phase::Done;
        }
        false
    }
    fn drain_call(
        &mut self,
        flush: bool,
        events: &mut Vec<StreamEvent>,
    ) -> infer_core::Result<bool> {
        // Resume the scan where the previous feed stopped, keeping one marker's worth of
        // overlap, so a body fed one byte at a time stays linear.
        let Some(found) = self.buffer[self.scanned..].find(TOOL_CALL_CLOSE) else {
            if self.buffer.len() > MAX_TOOL_ARGUMENT_BYTES {
                return Err(infer_core::Error::invalid(format!(
                    "tool call arguments exceed {MAX_TOOL_ARGUMENT_BYTES} bytes"
                )));
            }
            if flush {
                self.truncated = true;
                self.buffer.clear();
                self.phase = Phase::Done;
            } else {
                let overlap = self.buffer.len().saturating_sub(TOOL_CALL_CLOSE.len() - 1);
                self.scanned = floor_boundary(&self.buffer, overlap);
            }
            return Ok(false);
        };
        let index = self.scanned + found;
        let body = self.buffer[..index].to_string();
        self.buffer.drain(..index + TOOL_CALL_CLOSE.len());
        self.scanned = 0;
        self.phase = Phase::Text;
        if body.len() > MAX_TOOL_ARGUMENT_BYTES {
            return Err(infer_core::Error::invalid(format!(
                "tool call arguments exceed {MAX_TOOL_ARGUMENT_BYTES} bytes"
            )));
        }
        if self.calls >= MAX_TOOL_CALLS {
            return Err(infer_core::Error::invalid(format!(
                "response carries more than {MAX_TOOL_CALLS} tool calls"
            )));
        }
        let parsed = match self.dialect {
            ToolDialect::JsonBlock => parse_json_block(body.trim()),
            ToolDialect::FunctionParameters => parse_function_block(body.trim()),
        };
        match parsed {
            Some(call)
                if self
                    .declared
                    .as_ref()
                    .is_none_or(|names| names.contains(&call.name)) =>
            {
                self.calls += 1;
                events.push(StreamEvent::Call(call));
            }
            _ => {
                // Keep an undecodable or undeclared block visible as text rather than
                // publishing a call the service cannot vouch for.
                events.push(StreamEvent::Text(format!(
                    "{TOOL_CALL_OPEN}{body}{TOOL_CALL_CLOSE}"
                )));
                self.text_published = true;
            }
        }
        Ok(true)
    }
    fn publish_text(&mut self, chunk: &str, events: &mut Vec<StreamEvent>) {
        let text = if self.text_published {
            chunk
        } else {
            chunk.trim_start()
        };
        if text.is_empty() {
            return;
        }
        self.text_published = true;
        events.push(StreamEvent::Text(text.to_string()));
    }
    fn publish_reasoning(&mut self, chunk: &str, events: &mut Vec<StreamEvent>) {
        let text = if self.reasoning_published {
            chunk
        } else {
            chunk.trim_start()
        };
        if text.is_empty() {
            return;
        }
        self.reasoning_published = true;
        events.push(StreamEvent::Reasoning(text.to_string()));
    }
}

/// Assemble published events into the non-streaming result.
#[must_use]
pub fn assemble(events: Vec<StreamEvent>, truncated: bool) -> ParsedOutput {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut calls = Vec::new();
    for event in events {
        match event {
            StreamEvent::Text(text) => content.push_str(&text),
            StreamEvent::Reasoning(text) => reasoning.push_str(&text),
            StreamEvent::Call(call) => calls.push(call),
        }
    }
    ParsedOutput {
        content,
        reasoning: (!reasoning.is_empty()).then_some(reasoning),
        calls,
        truncated,
    }
}

/// Split reasoning, tool calls and assistant text out of one completed model response.
///
/// This uses the JSON-block dialect; prefer [`parse_with`] with the dialect the package's
/// template asks for.
///
/// # Errors
/// Returns an invalid-input error when a tool-call block carries more than
/// `MAX_TOOL_CALLS` calls or an argument payload above `MAX_TOOL_ARGUMENT_BYTES`.
pub fn parse(text: &str) -> infer_core::Result<ParsedOutput> {
    parse_with(text, ToolDialect::JsonBlock, None)
}

/// Parse a complete response in the dialect the package's template asks the model to emit.
///
/// This is [`OutputStreamParser`] fed the whole response at once, so the streamed and complete
/// forms of one response agree by construction. A call whose name is not in `declared` is left
/// in the content as text: the service does not publish an undeclared call and does not rewrite
/// the model's name into a declared one.
///
/// # Errors
/// Returns an invalid-input error when a tool-call block carries more than
/// `MAX_TOOL_CALLS` calls or an argument payload above `MAX_TOOL_ARGUMENT_BYTES`.
pub fn parse_with(
    text: &str,
    dialect: ToolDialect,
    declared: Option<&std::collections::BTreeSet<String>>,
) -> infer_core::Result<ParsedOutput> {
    let mut parser = OutputStreamParser::new(dialect, declared.cloned());
    let mut events = parser.feed(text)?;
    events.extend(parser.finish()?);
    Ok(assemble(events, parser.truncated()))
}

/// Largest char boundary at or below `index`.
fn floor_boundary(text: &str, index: usize) -> usize {
    let mut index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// Bytes of trailing whitespace in `text`.
fn trailing_whitespace(text: &str) -> usize {
    text.len() - text.trim_end().len()
}

/// Longest suffix of `text` that is a proper prefix of `marker`.
fn marker_prefix_len(text: &str, marker: &str) -> usize {
    let bytes = text.as_bytes();
    let marker = marker.as_bytes();
    let max = marker.len().min(bytes.len());
    (1..=max)
        .rev()
        .find(|&len| {
            text.is_char_boundary(text.len() - len) && bytes[text.len() - len..] == marker[..len]
        })
        .unwrap_or(0)
}

/// Bytes that must stay buffered before the next `feed`.
///
/// A marker prefix and any whitespace immediately before it are both held, because that
/// whitespace is a separator that only becomes real text if the marker never completes.
fn hold_len(buffer: &str, marker: &str) -> usize {
    let prefix = marker_prefix_len(buffer, marker);
    if prefix == 0 {
        return trailing_whitespace(buffer);
    }
    let mut start = buffer.len() - prefix;
    for (index, ch) in buffer[..start].char_indices().rev() {
        if !ch.is_whitespace() {
            break;
        }
        start = index;
    }
    buffer.len() - start
}

/// Decode one `<tool_call>` body as `{"name": ..., "arguments": ...}`.
fn parse_json_block(body: &str) -> Option<ParsedToolCall> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let object = value.as_object()?;
    let name = object.get("name")?.as_str()?.trim();
    if name.is_empty() {
        return None;
    }
    let arguments = match object.get("arguments") {
        // A string payload is preserved verbatim; an object is serialized back to compact JSON.
        Some(serde_json::Value::String(raw)) => raw.clone(),
        Some(value @ (serde_json::Value::Object(_) | serde_json::Value::Array(_))) => {
            serde_json::to_string(value).ok()?
        }
        _ => return None,
    };
    if arguments.len() > MAX_TOOL_ARGUMENT_BYTES {
        return None;
    }
    Some(ParsedToolCall {
        name: name.to_string(),
        arguments,
    })
}

/// Decode one `<tool_call>` body in the `<function=>`/`<parameter=>` dialect.
fn parse_function_block(body: &str) -> Option<ParsedToolCall> {
    let name = function_name(body)?;
    let mut arguments = serde_json::Map::new();
    let mut cursor = 0usize;
    while let Some(open) = body[cursor..].find(PARAMETER_MARKER) {
        let open = cursor + open;
        let header_start = open + PARAMETER_MARKER.len();
        let header_end = body[header_start..].find('>')? + header_start;
        let key = body[header_start..header_end].trim();
        if key.is_empty() {
            return None;
        }
        // The value runs to the matching close tag; a value may span multiple lines.
        let value_start = header_end + 1;
        let close = body[value_start..].find("</parameter>")? + value_start;
        let raw = body[value_start..close]
            .strip_prefix('\n')
            .unwrap_or_else(|| &body[value_start..close]);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        if arguments.contains_key(key) {
            // A duplicate parameter would silently drop one value; refuse the block instead.
            return None;
        }
        arguments.insert(key.to_string(), json_scalar(raw));
        cursor = close + "</parameter>".len();
    }
    let arguments = serde_json::to_string(&serde_json::Value::Object(arguments)).ok()?;
    if arguments.len() > MAX_TOOL_ARGUMENT_BYTES {
        return None;
    }
    Some(ParsedToolCall {
        name: name.to_string(),
        arguments,
    })
}

/// Read the function name out of a `<function=NAME>` header.
fn function_name(body: &str) -> Option<&str> {
    let start = body.find(FUNCTION_MARKER)? + FUNCTION_MARKER.len();
    let end = body[start..].find('>')? + start;
    let name = body[start..end].trim();
    (!name.is_empty()).then_some(name)
}

/// Interpret one parameter value as JSON when it is valid JSON, otherwise as a string.
///
/// The dialect has no type annotations, so a bare `21` becomes a number and `Paris` stays a
/// string. Nothing is rewritten: the text is either parsed by JSON rules or taken literally.
fn json_scalar(raw: &str) -> serde_json::Value {
    serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
}

#[cfg(test)]
#[path = "../../tests/unit/input_tool.rs"]
mod tests;
