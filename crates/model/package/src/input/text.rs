use crate::package_path;
use infer_core::{Error, ErrorCode, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};
use tokenizers::Tokenizer;

/// Maximum accepted `tokenizer.json` asset size.
const TOKENIZER_JSON_MAX_BYTES: u64 = 64 * crate::constants::MIB_U64;
/// Maximum accepted tokenizer-config or chat-template asset size.
const TEXT_ASSET_MAX_BYTES: u64 = 2 * crate::constants::MIB_U64;
/// Maximum accepted chat or input text size.
const TEXT_MAX_BYTES: usize = 2 * crate::constants::MIB;
/// Maximum chat messages accepted by the template renderer.
const MAX_CHAT_MESSAGES: usize = 256;
/// Template evaluation fuel bounding render work per chat request.
const TEMPLATE_FUEL: u64 = 100_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChatOptions {
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    pub tools: Option<serde_json::Value>,
}
impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: false,
            tools: None,
        }
    }
}
pub struct TextAssets {
    tokenizer: Tokenizer,
    template: Option<String>,
    special: BTreeMap<String, serde_json::Value>,
    pub fingerprint: String,
    max_tokens: usize,
    pub generation: crate::GenerationDefaults,
}
fn read(path: &Path, budget: u64) -> Result<Vec<u8>> {
    use std::io::Read;
    let file = std::fs::File::open(path).map_err(|e| Error::invalid(e.to_string()))?;
    if file
        .metadata()
        .map_err(|e| Error::invalid(e.to_string()))?
        .len()
        > budget
    {
        return Err(Error::new(
            ErrorCode::Capacity,
            "tokenizer/template asset exceeds budget",
        ));
    }
    let mut bytes = vec![];
    file.take(budget + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| Error::invalid(e.to_string()))?;
    if bytes.len() as u64 > budget {
        return Err(Error::new(
            ErrorCode::Capacity,
            "tokenizer asset grew beyond budget",
        ));
    }
    Ok(bytes)
}
impl TextAssets {
    ///
    /// # Errors
    /// Returns an I/O or invalid-input error for missing, malformed, unsupported, or oversized assets.
    pub fn open(root: impl AsRef<Path>, max_tokens: usize) -> Result<Self> {
        if max_tokens == 0 {
            return Err(Error::invalid("empty text token budget"));
        }
        let root = root.as_ref();
        let bytes = read(
            &package_path(root, "tokenizer.json")?,
            TOKENIZER_JSON_MAX_BYTES,
        )?;
        let mut tokenizer =
            Tokenizer::from_bytes(&bytes).map_err(|e| Error::invalid(e.to_string()))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| Error::invalid(e.to_string()))?;
        tokenizer.with_padding(None);
        let mut hash = Sha256::new();
        hash.update(&bytes);
        let mut special = BTreeMap::new();
        let mut template = None;
        if root.join("tokenizer_config.json").exists() {
            let config_bytes = read(
                &package_path(root, "tokenizer_config.json")?,
                TEXT_ASSET_MAX_BYTES,
            )?;
            hash.update(&config_bytes);
            let value: serde_json::Value =
                serde_json::from_slice(&config_bytes).map_err(|e| Error::invalid(e.to_string()))?;
            for key in ["bos_token", "eos_token", "pad_token", "unk_token"] {
                if let Some(v) = value.get(key) {
                    special.insert(key.into(), v.get("content").unwrap_or(v).clone());
                }
            }
            if let Some(v) = value.get("chat_template") {
                if let Some(s) = v.as_str() {
                    template = Some(s.to_owned());
                } else if let Some(list) = v.as_array() {
                    template = list
                        .iter()
                        .find(|v| v.get("name").and_then(|v| v.as_str()) == Some("default"))
                        .and_then(|v| v.get("template"))
                        .and_then(|v| v.as_str())
                        .map(str::to_owned);
                    if template.is_none() {
                        return Err(Error::unsupported(
                            "named chat templates require a default template",
                        ));
                    }
                } else if !v.is_null() {
                    return Err(Error::invalid("invalid chat template asset"));
                }
            }
        }
        if root.join("chat_template.jinja").exists() {
            let bytes = read(
                &package_path(root, "chat_template.jinja")?,
                TEXT_ASSET_MAX_BYTES,
            )?;
            hash.update(&bytes);
            template = Some(String::from_utf8(bytes).map_err(|e| Error::invalid(e.to_string()))?);
        }
        Ok(Self {
            generation: crate::GenerationDefaults::open(root)?,
            tokenizer,
            template,
            special,
            fingerprint: format!("{:x}", hash.finalize()),
            max_tokens,
        })
    }
    ///
    /// # Errors
    /// Returns an invalid-input or capacity error if tokenization fails or the token limit is exceeded.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        if text.len() > TEXT_MAX_BYTES {
            return Err(Error::new(ErrorCode::Capacity, "text input exceeds 2 MiB"));
        }
        let encoded = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| Error::invalid(e.to_string()))?;
        let tokens = encoded.get_ids().to_vec();
        if tokens.is_empty() || tokens.len() > self.max_tokens {
            return Err(Error::new(
                ErrorCode::Capacity,
                "text token count exceeds context budget",
            ));
        }
        Ok(tokens)
    }
    ///
    /// # Errors
    /// Returns an invalid-input error if token IDs cannot be decoded.
    pub fn decode(&self, tokens: &[u32], skip_special: bool) -> Result<String> {
        if tokens.len() > self.max_tokens {
            return Err(Error::new(
                ErrorCode::Capacity,
                "decode exceeds token budget",
            ));
        }
        self.tokenizer
            .decode(tokens, skip_special)
            .map_err(|e| Error::invalid(e.to_string()))
    }
    ///
    /// # Errors
    /// Returns an invalid-input or unsupported error for invalid messages, template errors, or exhausted template fuel.
    pub fn render_chat(&self, messages: &[ChatMessage], options: &ChatOptions) -> Result<String> {
        let template = self
            .template
            .as_ref()
            .ok_or_else(|| Error::unsupported("package has no chat template"))?;
        let bytes = messages
            .iter()
            .try_fold(0usize, |sum, m| sum.checked_add(m.content.len()))
            .ok_or_else(|| Error::invalid("chat length overflow"))?;
        if messages.is_empty()
            || messages.len() > MAX_CHAT_MESSAGES
            || bytes > TEXT_MAX_BYTES
            || messages
                .iter()
                .any(|m| !["system", "user", "assistant", "tool"].contains(&m.role.as_str()))
        {
            return Err(Error::invalid("invalid/budget-exceeding chat messages"));
        }
        let mut env = minijinja::Environment::new();
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
        env.set_fuel(Some(TEMPLATE_FUEL));
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    message,
                ))
            },
        );
        env.add_template("chat", template)
            .map_err(|e| Error::invalid(e.to_string()))?;
        let mut context = self.special.clone();
        context.insert(
            "messages".into(),
            serde_json::to_value(messages).map_err(|e| Error::invalid(e.to_string()))?,
        );
        context.insert(
            "add_generation_prompt".into(),
            options.add_generation_prompt.into(),
        );
        context.insert("enable_thinking".into(), options.enable_thinking.into());
        context.insert(
            "tools".into(),
            options.tools.clone().unwrap_or(serde_json::Value::Null),
        );
        let rendered = env
            .get_template("chat")
            .map_err(|e| Error::invalid(e.to_string()))?
            .render(context)
            .map_err(|e| Error::invalid(e.to_string()))?;
        if rendered.len() > TEXT_MAX_BYTES {
            return Err(Error::new(
                ErrorCode::Capacity,
                "rendered chat exceeds budget",
            ));
        }
        Ok(rendered)
    }
    ///
    /// # Errors
    /// Returns a template, tokenization, or capacity error if chat rendering or encoding fails.
    pub fn encode_chat(&self, messages: &[ChatMessage], options: &ChatOptions) -> Result<Vec<u32>> {
        self.encode(&self.render_chat(messages, options)?, false)
    }
}
