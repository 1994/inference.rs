//! A forward-only JSON envelope scan. Argument bytes retain their original formatting.
use super::types::{ParsedToolCall, TOOL_CALL_CLOSE};
use serde::Deserialize;
use serde_json::value::RawValue;

#[derive(Debug, Clone, Default)]
pub(super) struct JsonScanner {
    cursor: usize,
    depth: usize,
    quoted: bool,
    string_start: usize,
    key_expected: bool,
    key: String,
    name: Option<String>,
    argument_start: Option<usize>,
    argument_end: Option<usize>,
}
impl JsonScanner {
    pub(super) const fn new() -> Self {
        Self {
            cursor: 0,
            depth: 0,
            quoted: false,
            string_start: 0,
            key_expected: false,
            key: String::new(),
            name: None,
            argument_start: None,
            argument_end: None,
        }
    }
    pub(super) fn scan(&mut self, body: &str) -> Option<usize> {
        while self.cursor < body.len() {
            let byte = body.as_bytes()[self.cursor];
            if self.quoted {
                let tail = &body[self.cursor..];
                let Some(offset) = tail.find('"') else {
                    self.cursor = body.len();
                    return None;
                };
                self.cursor += offset;
                // The body stays buffered, so escape parity also covers a backslash run that
                // ended in the previous chunk. Runs before distinct quotes never overlap.
                let slashes = body.as_bytes()[..self.cursor]
                    .iter()
                    .rev()
                    .take_while(|&&byte| byte == b'\\')
                    .count();
                if slashes % 2 == 0 {
                    self.end_string(body);
                }
            } else {
                if byte == b'<' {
                    let tail = &body[self.cursor..];
                    if tail.starts_with(TOOL_CALL_CLOSE) {
                        return Some(self.cursor);
                    }
                    if TOOL_CALL_CLOSE.starts_with(tail) {
                        return None;
                    }
                }
                self.structural_byte(byte);
            }
            self.cursor += 1;
        }
        None
    }
    fn end_string(&mut self, body: &str) {
        self.quoted = false;
        if self.depth == 1 {
            let value = serde_json::from_str::<String>(&body[self.string_start..=self.cursor]);
            if self.key_expected {
                self.key = value.unwrap_or_default();
                self.key_expected = false;
            } else if self.key == "name" {
                self.name = value.ok().filter(|name| !name.is_empty());
            }
        }
    }
    fn structural_byte(&mut self, byte: u8) {
        match byte {
            b'"' => {
                self.quoted = true;
                self.string_start = self.cursor;
            }
            b'{' | b'[' => {
                if self.depth == 1 && self.key == "arguments" && byte == b'{' {
                    self.argument_start.get_or_insert(self.cursor);
                }
                self.depth += 1;
                if self.depth == 1 {
                    self.key_expected = true;
                }
            }
            b'}' | b']' => {
                self.depth = self.depth.saturating_sub(1);
                if self.depth == 1 && self.argument_start.is_some() && self.argument_end.is_none() {
                    self.argument_end = Some(self.cursor + 1);
                }
            }
            b',' if self.depth == 1 => {
                self.key_expected = true;
            }
            _ => {}
        }
    }
    pub(super) fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    pub(super) fn arguments<'a>(&self, body: &'a str) -> Option<&'a str> {
        self.argument_start
            .map(|start| &body[start..self.argument_end.unwrap_or(self.cursor)])
    }
}

#[derive(Deserialize)]
struct Envelope<'a> {
    name: String,
    #[serde(borrow)]
    arguments: &'a RawValue,
}
/// `RawValue` validates JSON without allocating an argument object tree.
pub(super) fn complete(body: &str) -> Option<ParsedToolCall> {
    let envelope: Envelope<'_> = serde_json::from_str(body).ok()?;
    if envelope.name.is_empty() {
        return None;
    }
    let raw = envelope.arguments.get();
    let arguments = if raw.starts_with('{') {
        raw.to_string()
    } else if raw.starts_with('"') {
        let decoded: String = serde_json::from_str(raw).ok()?;
        let value: &RawValue = serde_json::from_str(&decoded).ok()?;
        if !value.get().starts_with('{') {
            return None;
        }
        decoded
    } else {
        return None;
    };
    Some(ParsedToolCall {
        name: envelope.name,
        arguments,
    })
}
