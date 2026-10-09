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

/// Split reasoning, tool calls and assistant text out of one completed model response.
///
/// # Errors
/// Returns an invalid-input error when a tool-call block carries more than
/// [`MAX_TOOL_CALLS`] calls or an argument payload above [`MAX_TOOL_ARGUMENT_BYTES`].
pub fn parse(text: &str) -> infer_core::Result<ParsedOutput> {
    parse_with(text, ToolDialect::JsonBlock, None)
}

/// Parse a response in the dialect the package's template asks the model to emit.
///
/// The scan is single-pass over the response with bounded buffers: each marker is located once,
/// the argument payload is bounded before it is decoded, and nothing is rewritten in place. A
/// call whose name is not in `declared` is left in the content as text: the service does not
/// publish an undeclared call and does not rewrite the model's name into a declared one.
///
/// # Errors
/// Returns an invalid-input error when a tool-call block carries more than
/// [`MAX_TOOL_CALLS`] calls or an argument payload above [`MAX_TOOL_ARGUMENT_BYTES`].
pub fn parse_with(
    text: &str,
    dialect: ToolDialect,
    declared: Option<&std::collections::BTreeSet<String>>,
) -> infer_core::Result<ParsedOutput> {
    let (reasoning, remainder) = split_reasoning(text);
    let mut content = String::with_capacity(remainder.len());
    let mut calls = Vec::new();
    let mut truncated = false;
    let mut cursor = 0usize;
    while let Some(open) = remainder[cursor..].find(TOOL_CALL_OPEN) {
        let open = cursor + open;
        content.push_str(&remainder[cursor..open]);
        let body_start = open + TOOL_CALL_OPEN.len();
        let Some(close) = remainder[body_start..].find(TOOL_CALL_CLOSE) else {
            // The block never closed. Withhold the tail rather than publishing half a call.
            truncated = true;
            cursor = remainder.len();
            break;
        };
        let close = body_start + close;
        let body = remainder[body_start..close].trim();
        if body.len() > MAX_TOOL_ARGUMENT_BYTES {
            return Err(infer_core::Error::invalid(format!(
                "tool call arguments exceed {MAX_TOOL_ARGUMENT_BYTES} bytes"
            )));
        }
        if calls.len() >= MAX_TOOL_CALLS {
            return Err(infer_core::Error::invalid(format!(
                "response carries more than {MAX_TOOL_CALLS} tool calls"
            )));
        }
        // A block that does not decode as the declared dialect is kept as text: the parser does
        // not guess at a malformed payload.
        let parsed = match dialect {
            ToolDialect::JsonBlock => parse_json_block(body),
            ToolDialect::FunctionParameters => parse_function_block(body),
        };
        match parsed {
            Some(call) if declared.is_none_or(|names| names.contains(&call.name)) => {
                calls.push(call)
            }
            _ => content.push_str(&remainder[open..close + TOOL_CALL_CLOSE.len()]),
        }
        cursor = close + TOOL_CALL_CLOSE.len();
    }
    content.push_str(&remainder[cursor.min(remainder.len())..]);
    Ok(ParsedOutput {
        content: content.trim().to_string(),
        reasoning,
        calls,
        truncated,
    })
}

/// Extract a leading reasoning block, returning it and the remaining text.
fn split_reasoning(text: &str) -> (Option<String>, &str) {
    let Some(open) = text.find(REASONING_OPEN) else {
        return (None, text);
    };
    let body_start = open + REASONING_OPEN.len();
    let Some(close) = text[body_start..].find(REASONING_CLOSE) else {
        // An unterminated reasoning block leaves no assistant text to publish.
        return (Some(text[body_start..].trim().to_string()), "");
    };
    let close = body_start + close;
    let reasoning = text[body_start..close].trim().to_string();
    let remainder = &text[close + REASONING_CLOSE.len()..];
    (Some(reasoning), remainder)
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
            .unwrap_or(&body[value_start..close]);
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
        assert!(parsed.calls.is_empty());
        assert!(parsed.truncated);
    }

    #[test]
    fn a_malformed_payload_stays_text() {
        let parsed = parse("<tool_call>{not json}</tool_call>").unwrap();
        assert!(parsed.calls.is_empty());
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
            assert!(parse(&text).unwrap().calls.is_empty(), "{body}");
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
        assert!(parsed.content.is_empty());
    }

    #[test]
    fn ordinary_text_is_returned_unchanged() {
        let parsed = parse("plain answer").unwrap();
        assert_eq!(parsed.content, "plain answer");
        assert!(parsed.reasoning.is_none() && parsed.calls.is_empty());
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
        let declared = ["declared".to_string()].into_iter().collect();
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
            assert!(parsed.calls.is_empty(), "{body}");
            assert!(
                parsed.content.contains(FUNCTION_MARKER)
                    || parsed.content.contains(PARAMETER_MARKER)
            );
        }
    }

    #[test]
    fn a_json_block_is_not_a_function_block_and_the_reverse() {
        let json = "<tool_call>{\"name\":\"f\",\"arguments\":{\"k\":1}}</tool_call>";
        assert!(
            parse_with(json, ToolDialect::FunctionParameters, None)
                .unwrap()
                .calls
                .is_empty()
        );
        let function =
            "<tool_call><function=f>\n<parameter=k>\n1\n</parameter>\n</function></tool_call>";
        assert!(
            parse_with(function, ToolDialect::JsonBlock, None)
                .unwrap()
                .calls
                .is_empty()
        );
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
}
