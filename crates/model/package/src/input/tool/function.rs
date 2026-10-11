//! Strict function/parameter envelope; values become append-only JSON at parameter boundaries.
use super::types::{FUNCTION_MARKER, PARAMETER_MARKER, ParsedToolCall, TOOL_CALL_CLOSE};
use std::collections::BTreeSet;
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_CLOSE: &str = "</parameter>";
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Start,
    Name,
    Between,
    Key,
    Value,
    Closed,
    Invalid,
}
#[derive(Debug, Clone)]
pub(super) struct FunctionScanner {
    phase: Phase,
    cursor: usize,
    start: usize,
    name: Option<String>,
    key: String,
    keys: BTreeSet<String>,
    arguments: String,
    json_value: Option<bool>,
    quoted: bool,
    escaped: bool,
    abandoned_close: Option<usize>,
}
impl FunctionScanner {
    pub(super) const fn new() -> Self {
        Self {
            phase: Phase::Start,
            cursor: 0,
            start: 0,
            name: None,
            key: String::new(),
            keys: BTreeSet::new(),
            arguments: String::new(),
            json_value: None,
            quoted: false,
            escaped: false,
            abandoned_close: None,
        }
    }
    pub(super) fn scan(&mut self, body: &str, flush: bool) -> Option<usize> {
        while self.cursor < body.len() {
            let byte = body.as_bytes()[self.cursor];
            if byte == b'<' && self.phase != Phase::Value {
                let tail = &body[self.cursor..];
                if tail.starts_with(TOOL_CALL_CLOSE) {
                    return Some(self.cursor);
                }
                if TOOL_CALL_CLOSE.starts_with(tail) {
                    return None;
                }
            }
            match self.phase {
                Phase::Start | Phase::Between | Phase::Closed => {
                    if !self.boundary(body) {
                        return None;
                    }
                }
                Phase::Name | Phase::Key => self.header(body, byte),
                Phase::Value => {
                    if !self.value(body, byte) {
                        return None;
                    }
                }
                Phase::Invalid => self.cursor += 1,
            }
        }
        if flush { self.abandoned_close } else { None }
    }
    fn boundary(&mut self, body: &str) -> bool {
        let tail = &body[self.cursor..];
        if body.as_bytes()[self.cursor].is_ascii_whitespace() {
            self.cursor += 1;
            return true;
        }
        let marker = match self.phase {
            Phase::Start => FUNCTION_MARKER,
            Phase::Between if tail.starts_with(FUNCTION_CLOSE) => {
                self.arguments.push('}');
                self.phase = Phase::Closed;
                self.cursor += FUNCTION_CLOSE.len();
                return true;
            }
            Phase::Between if FUNCTION_CLOSE.starts_with(tail) => return false,
            Phase::Between => PARAMETER_MARKER,
            _ => {
                self.phase = Phase::Invalid;
                return true;
            }
        };
        if marker.starts_with(tail) {
            return false;
        }
        if tail.starts_with(marker) {
            self.cursor += marker.len();
            self.start = self.cursor;
            self.phase = if self.phase == Phase::Start {
                Phase::Name
            } else {
                Phase::Key
            };
        } else {
            self.phase = Phase::Invalid;
        }
        true
    }
    fn header(&mut self, body: &str, byte: u8) {
        if byte == b'>' {
            let name = body[self.start..self.cursor].trim();
            if name.is_empty() {
                self.phase = Phase::Invalid;
            } else if self.phase == Phase::Name {
                self.name = Some(name.to_string());
                self.arguments.push('{');
                self.phase = Phase::Between;
            } else if self.keys.insert(name.to_string()) {
                self.key = name.to_string();
                self.phase = Phase::Value;
                self.start = self.cursor + 1;
                self.json_value = None;
                self.quoted = false;
                self.escaped = false;
            } else {
                self.phase = Phase::Invalid;
            }
        }
        self.cursor += 1;
    }
    fn value(&mut self, body: &str, byte: u8) -> bool {
        if byte == b'<' && !self.quoted {
            let tail = &body[self.cursor..];
            if tail.len() < PARAMETER_CLOSE.len() && PARAMETER_CLOSE.starts_with(tail) {
                return false;
            }
            if tail.starts_with(PARAMETER_CLOSE) {
                self.abandoned_close = None;
                self.append_value(&body[self.start..self.cursor]);
                self.cursor += PARAMETER_CLOSE.len();
                self.phase = Phase::Between;
                return true;
            }
            // A missing parameter closer must terminate as an invalid block, not swallow following calls.
            if tail.starts_with(TOOL_CALL_CLOSE) {
                self.abandoned_close.get_or_insert(self.cursor);
                self.cursor += TOOL_CALL_CLOSE.len();
                return true;
            }
            if TOOL_CALL_CLOSE.starts_with(tail) {
                return false;
            }
        }
        if self.json_value.is_none() && !byte.is_ascii_whitespace() {
            self.json_value = Some(matches!(byte, b'{' | b'[' | b'"'));
        }
        // Free text has no quote state. Skip its ordinary bytes with the string searcher's
        // optimized delimiter scan instead of branching on every byte of long parameters.
        if self.json_value == Some(false) && byte != b'<' {
            let tail = &body[self.cursor..];
            self.cursor += tail.find('<').unwrap_or(tail.len());
            return true;
        }
        if self.json_value == Some(true) {
            if self.escaped {
                self.escaped = false;
            } else if self.quoted && byte == b'\\' {
                self.escaped = true;
            } else if byte == b'"' {
                self.quoted = !self.quoted;
            }
        }
        self.cursor += 1;
        true
    }
    fn append_value(&mut self, raw: &str) {
        let raw = raw.strip_prefix('\n').unwrap_or(raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        let value = serde_json::from_str::<serde_json::Value>(raw)
            .unwrap_or_else(|_| serde_json::Value::String(raw.to_string()));
        if self.keys.len() > 1 {
            self.arguments.push(',');
        }
        if let Ok(key) = serde_json::to_string(&self.key) {
            self.arguments.push_str(&key);
        }
        self.arguments.push(':');
        if let Ok(value) = serde_json::to_string(&value) {
            self.arguments.push_str(&value);
        }
    }
    pub(super) fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    pub(super) fn arguments(&self) -> Option<&str> {
        self.name.as_ref().map(|_| self.arguments.as_str())
    }
    pub(super) fn complete(&self) -> Option<ParsedToolCall> {
        if self.phase != Phase::Closed {
            return None;
        }
        Some(ParsedToolCall {
            name: self.name.clone()?,
            arguments: self.arguments.clone(),
        })
    }
}
