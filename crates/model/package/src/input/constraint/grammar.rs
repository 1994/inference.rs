use super::{ToolConstraints, schema};
use crate::ToolDialect;
use infer_core::{Error, Result};
use serde_json::{Value, json};
use std::fmt::Write;

const MAX_SCHEMA_BYTES: usize = 64 * 1024;
const MAX_CALLS: usize = 128;

pub(super) fn build(
    spec: &ToolConstraints,
    dialect: ToolDialect,
    thinking: bool,
    open: bool,
) -> Result<String> {
    if spec.tools.to_string().len() > MAX_SCHEMA_BYTES {
        return Err(Error::invalid("tool schemas exceed grammar byte budget"));
    }
    let tools = spec
        .tools
        .as_array()
        .ok_or_else(|| Error::invalid("tools must be an array"))?;
    let mut rules = String::new();
    let mut calls = Vec::new();
    for (index, tool) in tools.iter().enumerate() {
        let function = &tool["function"];
        let name = function["name"]
            .as_str()
            .ok_or_else(|| Error::invalid("tool requires name"))?;
        let parameters = function.get("parameters").cloned().unwrap_or_else(
            || json!({"type":"object","properties":{},"required":[],"additionalProperties":false}),
        );
        schema::validate(&parameters, function["strict"] == true)?;
        if spec.named.as_ref().is_some_and(|chosen| chosen != name) {
            continue;
        }
        let rule = format!("call_{index}");
        let body = match dialect {
            ToolDialect::JsonBlock => {
                let payload = json!({"type":"object","properties":{"name":{"type":"string","enum":[name]},"arguments":parameters},"required":["name","arguments"],"additionalProperties":false});
                let _ = writeln!(rules, "payload_{index}: %json {payload}");
                format!("\"<tool_call>\" payload_{index} \"</tool_call>\"")
            }
            ToolDialect::FunctionParameters => function_body(&mut rules, &parameters, name, index)?,
        };
        let _ = writeln!(rules, "{rule}: {body}");
        calls.push(rule);
    }
    if calls.is_empty() {
        return Err(Error::invalid(
            "constrained tool choice requires a declared tool",
        ));
    }
    let repetition = if spec.parallel && spec.named.is_none() {
        format!("(text? call)~0..{}", MAX_CALLS - 1)
    } else {
        String::new()
    };
    let reasoning = if !thinking {
        ""
    } else if open {
        "thought? \"</think>\""
    } else {
        "(\"<think>\" thought? \"</think>\")?"
    };
    let body = if spec.required || spec.named.is_some() {
        "text? calls text?"
    } else {
        "(text | text? calls text?)"
    };
    // Each text atom ends before a structural marker or includes its first mismatching byte.
    // Partial marker prefixes cannot become adjacent text atoms and evade the grammar.
    Ok(format!(
        r"start: {reasoning} {body}
calls: call {repetition}
call: {calls}
text: TEXT_ATOM+
TEXT_ATOM: /[^<]|<[^<t]|<t[^<oh]|<to[^<o]|<too[^<l]|<tool[^<_]|<tool_[^<c]|<tool_c[^<a]|<tool_ca[^<l]|<tool_cal[^<l]|<tool_call[^<>]|<th[^<i]|<thi[^<n]|<thin[^<k]|<think[^<>]/
thought: THOUGHT_ATOM+
THOUGHT_ATOM: /[^<]|<[^<\/]|<\/[^<t]|<\/t[^<h]|<\/th[^<i]|<\/thi[^<n]|<\/thin[^<k]|<\/think[^<>]/
{rules}",
        calls = calls.join(" | ")
    ))
}
fn function_body(
    rules: &mut String,
    parameters: &Value,
    name: &str,
    index: usize,
) -> Result<String> {
    if parameters.get("enum").is_some() || parameters.get("anyOf").is_some() {
        return Err(Error::unsupported(
            "function dialect cannot enforce root enum or anyOf",
        ));
    }
    let mut body = format!("{}", json!(format!("<tool_call><function={name}>")));
    let properties = parameters["properties"]
        .as_object()
        .ok_or_else(|| Error::invalid("function dialect requires object properties"))?;
    for (field, (key, schema)) in properties.iter().enumerate() {
        if key.is_empty() || key.trim() != key || key.contains(['<', '>', '\n', '\r']) {
            return Err(Error::unsupported(
                "property name cannot be encoded in function dialect",
            ));
        }
        let rule = format!("parameter_{index}_{field}");
        let _ = writeln!(rules, "{rule}: %json {schema}");
        let required = parameters["required"]
            .as_array()
            .is_some_and(|r| r.iter().any(|v| v == key));
        let _ = write!(
            body,
            " ({} {rule} \"</parameter>\"){}",
            json!(format!("<parameter={key}>")),
            if required { "" } else { "?" }
        );
    }
    body.push_str(" \"</function></tool_call>\"");
    Ok(body)
}
