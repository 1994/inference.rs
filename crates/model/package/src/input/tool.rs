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
mod tests {
    use super::*;

    #[test]
    fn the_reasoning_markers_match_the_served_package_tokens() {
        // Tokens 151667 and 151668 of the served packages spell these in plain ASCII.
        assert_eq!(REASONING_OPEN, "\u{3c}think\u{3e}");
        assert_eq!(REASONING_CLOSE, "\u{3c}/think\u{3e}");
        assert_eq!(TOOL_CALL_OPEN, "\u{3c}tool_call\u{3e}");
        assert_eq!(TOOL_CALL_CLOSE, "\u{3c}/tool_call\u{3e}");
    }

    #[test]
    fn a_json_block_becomes_one_call_with_verbatim_arguments() {
        let parsed = parse(
            "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"units\": \"c\"}}\n</tool_call>\n",
        )
        .unwrap();
        assert_eq!(parsed.content, "Let me check.");
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].name, "get_weather");
        assert_eq!(parsed.calls[0].arguments, r#"{"city":"Paris","units":"c"}"#);
        assert!(!parsed.truncated && parsed.reasoning.is_none());
    }

    #[test]
    fn a_string_argument_payload_is_never_rewritten() {
        let parsed =
            parse("<tool_call>{\"name\":\"f\",\"arguments\":\"{ not json }\"}</tool_call>")
                .unwrap();
        assert_eq!(parsed.calls[0].arguments, "{ not json }");
    }

    #[test]
    fn multiple_calls_keep_their_order_and_leave_no_marker_text() {
        let parsed = parse(concat!(
            "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>",
            "between",
            "<tool_call>{\"name\":\"b\",\"arguments\":{\"x\":1}}</tool_call>"
        ))
        .unwrap();
        assert_eq!(
            parsed
                .calls
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(parsed.content, "between");
    }

    #[test]
    fn an_unterminated_block_is_withheld_instead_of_repaired() {
        let parsed = parse("answer<tool_call>{\"name\":\"a\",\"arguments\":{}").unwrap();
        assert_eq!(parsed.content, "answer");
        assert_eq!(parsed.calls.len(), 0);
        assert!(parsed.truncated);
    }

    #[test]
    fn a_malformed_payload_stays_text() {
        let parsed = parse("<tool_call>{not json}</tool_call>").unwrap();
        assert_eq!(parsed.calls.len(), 0);
        assert!(parsed.content.contains(TOOL_CALL_OPEN));
        assert!(!parsed.truncated);
    }

    #[test]
    fn a_payload_without_a_declared_name_is_not_a_call() {
        for body in [
            r#"{"arguments":{"a":1}}"#,
            r#"{"name":"","arguments":{}}"#,
            r#"{"name":"f","arguments":12}"#,
            r#"[{"name":"f"}]"#,
        ] {
            let text = format!("{TOOL_CALL_OPEN}{body}{TOOL_CALL_CLOSE}");
            assert_eq!(parse(&text).unwrap().calls.len(), 0, "{body}");
        }
    }

    #[test]
    fn a_leading_reasoning_block_is_separated_from_the_answer() {
        let text = format!("{REASONING_OPEN}weigh the options{REASONING_CLOSE}\n\nFinal answer.");
        let parsed = parse(&text).unwrap();
        assert_eq!(parsed.reasoning.as_deref(), Some("weigh the options"));
        assert_eq!(parsed.content, "Final answer.");
    }

    #[test]
    fn an_unterminated_reasoning_block_publishes_nothing() {
        let text = format!("{REASONING_OPEN}still going");
        let parsed = parse(&text).unwrap();
        assert_eq!(parsed.reasoning.as_deref(), Some("still going"));
        assert_eq!(parsed.content, "");
    }

    #[test]
    fn ordinary_text_is_returned_unchanged() {
        let parsed = parse("plain answer").unwrap();
        assert_eq!(parsed.content, "plain answer");
        assert!(parsed.reasoning.is_none());
        assert_eq!(parsed.calls.len(), 0);
    }

    #[test]
    fn oversized_arguments_are_rejected_rather_than_truncated() {
        let huge = "x".repeat(MAX_TOOL_ARGUMENT_BYTES + 1);
        let text =
            format!("{TOOL_CALL_OPEN}{{\"name\":\"f\",\"arguments\":\"{huge}\"}}{TOOL_CALL_CLOSE}");
        assert!(parse(&text).is_err());
    }

    #[test]
    fn an_undeclared_name_is_left_as_text_instead_of_published() {
        let text = concat!(
            "<tool_call>{\"name\":\"declared\",\"arguments\":{}}</tool_call>",
            "<tool_call>{\"name\":\"invented\",\"arguments\":{}}</tool_call>"
        );
        let declared = std::iter::once("declared".to_string()).collect();
        let parsed = parse_with(text, ToolDialect::JsonBlock, Some(&declared)).unwrap();
        assert_eq!(
            parsed
                .calls
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            ["declared"]
        );
        assert!(parsed.content.contains("invented"));
        // Without a declaration list every structurally valid call is published.
        assert_eq!(parse(text).unwrap().calls.len(), 2);
    }

    #[test]
    fn the_dialect_is_detected_from_the_template_that_asks_for_it() {
        assert_eq!(
            ToolDialect::detect("<tool_call>\n<function=NAME>\n<parameter=K>"),
            ToolDialect::FunctionParameters
        );
        assert_eq!(
            ToolDialect::detect("{\"name\": <function-name>, \"arguments\": ...}"),
            ToolDialect::JsonBlock
        );
    }

    #[test]
    fn a_function_block_becomes_one_call_with_typed_parameters() {
        let text = concat!(
            "Checking.\n<tool_call>\n<function=get_weather>\n",
            "<parameter=city>\nParis\n</parameter>\n",
            "<parameter=days>\n3\n</parameter>\n",
            "<parameter=notes>\nline one\nline two\n</parameter>\n",
            "</function>\n</tool_call>"
        );
        let parsed = parse_with(text, ToolDialect::FunctionParameters, None).unwrap();
        assert_eq!(parsed.content, "Checking.");
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].name, "get_weather");
        let arguments: serde_json::Value =
            serde_json::from_str(&parsed.calls[0].arguments).unwrap();
        assert_eq!(arguments["city"], "Paris");
        // A bare number is valid JSON, so it stays a number; free text stays a string.
        assert_eq!(arguments["days"], 3);
        assert_eq!(arguments["notes"], "line one\nline two");
    }

    #[test]
    fn a_function_block_without_parameters_is_an_empty_object() {
        let text = "<tool_call><function=ping>\n</function></tool_call>";
        let parsed = parse_with(text, ToolDialect::FunctionParameters, None).unwrap();
        assert_eq!(parsed.calls[0].arguments, "{}");
    }

    #[test]
    fn malformed_function_blocks_are_left_as_text() {
        for body in [
            // No function header at all.
            "<parameter=city>\nParis\n</parameter>",
            // An empty function name.
            "<function=>\n<parameter=city>\nParis\n</parameter>",
            // A parameter that never closes.
            "<function=f>\n<parameter=city>\nParis",
            // A duplicate parameter would drop one value.
            "<function=f>\n<parameter=k>\n1\n</parameter>\n<parameter=k>\n2\n</parameter>",
            // A parameter without a name.
            "<function=f>\n<parameter=>\n1\n</parameter>",
        ] {
            let text = format!("{TOOL_CALL_OPEN}{body}{TOOL_CALL_CLOSE}");
            let parsed = parse_with(&text, ToolDialect::FunctionParameters, None).unwrap();
            assert_eq!(parsed.calls.len(), 0, "{body}");
            assert!(
                parsed.content.contains(FUNCTION_MARKER)
                    || parsed.content.contains(PARAMETER_MARKER)
            );
        }
    }

    #[test]
    fn a_json_block_is_not_a_function_block_and_the_reverse() {
        let json = "<tool_call>{\"name\":\"f\",\"arguments\":{\"k\":1}}</tool_call>";
        let parsed = parse_with(json, ToolDialect::FunctionParameters, None).unwrap();
        assert_eq!(parsed.calls.len(), 0);
        let function =
            "<tool_call><function=f>\n<parameter=k>\n1\n</parameter>\n</function></tool_call>";
        let parsed = parse_with(function, ToolDialect::JsonBlock, None).unwrap();
        assert_eq!(parsed.calls.len(), 0);
    }

    #[test]
    fn a_tool_call_argument_object_round_trips_for_the_template() {
        let call = ToolCall {
            id: "call_1".into(),
            name: "get_weather".into(),
            arguments: "{\"city\":\"Paris\"}".into(),
        };
        assert_eq!(call.argument_object().unwrap()["city"], "Paris");
        let malformed = ToolCall {
            arguments: "{not json}".into(),
            ..call
        };
        assert!(malformed.argument_object().is_err());
    }

    /// Feed `text` in fixed character groups and assemble the result.
    fn feed_all(text: &str, dialect: ToolDialect, chunk: usize) -> ParsedOutput {
        let chars: Vec<char> = text.chars().collect();
        let mut parser = OutputStreamParser::new(dialect, None);
        let mut events = Vec::new();
        for piece in chars.chunks(chunk.max(1)) {
            let piece: String = piece.iter().collect();
            events.extend(parser.feed(&piece).unwrap());
        }
        events.extend(parser.finish().unwrap());
        assemble(events, parser.truncated())
    }

    fn samples() -> Vec<(ToolDialect, String)> {
        let json_call = format!(
            "{TOOL_CALL_OPEN}{{\"name\":\"a\",\"arguments\":{{\"x\":1}}}}{TOOL_CALL_CLOSE}"
        );
        let function_call = format!(
            "{TOOL_CALL_OPEN}\n<function=b>\n<parameter=k>\nv\n</parameter>\n</function>\n{TOOL_CALL_CLOSE}"
        );
        vec![
            (ToolDialect::JsonBlock, "plain answer".to_string()),
            (ToolDialect::JsonBlock, "  padded \n\n".to_string()),
            (ToolDialect::JsonBlock, format!("Sure.\n{json_call}\n")),
            (ToolDialect::JsonBlock, format!("{json_call}and{json_call}")),
            (
                ToolDialect::JsonBlock,
                format!("{TOOL_CALL_OPEN}{{not json}}{TOOL_CALL_CLOSE}"),
            ),
            (
                ToolDialect::JsonBlock,
                format!("{TOOL_CALL_OPEN}{{\"name\":\"a\",\"arguments\":{{}}"),
            ),
            (
                ToolDialect::FunctionParameters,
                format!("Checking.\n{function_call}\nDone."),
            ),
            (
                ToolDialect::FunctionParameters,
                format!("{TOOL_CALL_OPEN}<function=ping>\n</function>{TOOL_CALL_CLOSE}"),
            ),
            (
                ToolDialect::FunctionParameters,
                format!("{REASONING_OPEN}weigh it{REASONING_CLOSE}\n\nAnswer."),
            ),
            (
                ToolDialect::FunctionParameters,
                format!("{REASONING_OPEN}never closed"),
            ),
            (ToolDialect::FunctionParameters, String::new()),
        ]
    }

    #[test]
    fn streaming_and_complete_parsing_agree_at_every_chunking() {
        for (dialect, text) in samples() {
            let complete = parse_with(&text, dialect, None).unwrap();
            for chunk in [1usize, 2, 3, 5, 17, 4096] {
                assert_eq!(
                    feed_all(&text, dialect, chunk),
                    complete,
                    "chunk {chunk} disagreed on {text:?}"
                );
            }
        }
    }

    #[test]
    fn a_marker_split_across_chunks_is_still_one_call() {
        let text =
            format!("hi{TOOL_CALL_OPEN}{{\"name\":\"a\",\"arguments\":{{}}}}{TOOL_CALL_CLOSE}");
        // Feed up to the middle of the opening marker, then the rest.
        let split = "hi<too".len();
        let mut parser = OutputStreamParser::new(ToolDialect::JsonBlock, None);
        let mut events = parser.feed(&text[..split]).unwrap();
        // The partial marker must be held, not published as text.
        assert_eq!(events, vec![StreamEvent::Text("hi".into())]);
        events.extend(parser.feed(&text[split..]).unwrap());
        events.extend(parser.finish().unwrap());
        assert_eq!(assemble(events, parser.truncated()).calls.len(), 1);
    }

    #[test]
    fn a_call_is_published_as_soon_as_its_block_closes() {
        let mut parser = OutputStreamParser::new(ToolDialect::JsonBlock, None);
        let head = format!("{TOOL_CALL_OPEN}{{\"name\":\"a\",\"arguments\":{{}}}}");
        assert_eq!(parser.feed(&head).unwrap().len(), 0);
        let events = parser.feed(TOOL_CALL_CLOSE).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::Call(_)));
    }

    #[test]
    fn whitespace_around_a_call_is_never_published() {
        let mut parser = OutputStreamParser::new(ToolDialect::JsonBlock, None);
        let mut events = parser.feed("  ").unwrap();
        events.extend(parser.feed("\n").unwrap());
        assert_eq!(events.len(), 0);
        events.extend(parser.feed("answer").unwrap());
        assert_eq!(events, vec![StreamEvent::Text("answer".into())]);
        events.extend(parser.finish().unwrap());
        assert_eq!(assemble(events, parser.truncated()).content, "answer");
    }

    #[test]
    fn a_reasoning_block_streams_before_the_answer() {
        let mut parser = OutputStreamParser::new(ToolDialect::FunctionParameters, None);
        let mut events = parser.feed(REASONING_OPEN).unwrap();
        assert!(events.is_empty(), "the marker alone is not reasoning yet");
        events.extend(parser.feed("why").unwrap());
        assert_eq!(events, vec![StreamEvent::Reasoning("why".into())]);
        events.extend(parser.feed(REASONING_CLOSE).unwrap());
        events.extend(parser.feed("Answer").unwrap());
        events.extend(parser.finish().unwrap());
        let parsed = assemble(events, parser.truncated());
        assert_eq!(parsed.reasoning.as_deref(), Some("why"));
        assert_eq!(parsed.content, "Answer");
    }

    #[test]
    fn an_unterminated_call_is_withheld_without_publishing_partial_arguments() {
        let mut parser = OutputStreamParser::new(ToolDialect::JsonBlock, None);
        let mut events = parser.feed("text").unwrap();
        events.extend(parser.feed(TOOL_CALL_OPEN).unwrap());
        events.extend(parser.feed("{\"name\":\"a\",").unwrap());
        events.extend(parser.finish().unwrap());
        let parsed = assemble(events, parser.truncated());
        assert_eq!(parsed.content, "text");
        assert_eq!(parsed.calls.len(), 0);
        assert!(parsed.truncated);
    }
}
