//! Server-sent events for the `OpenAI` chat and completion protocols.
//!
//! Both the streamed and the complete reply come from [`prepare`], so they accept the same
//! requests and resolve the same limits. Text is decoded and parsed incrementally on the CPU
//! pool: each committed token is detokenized once and fed to the parser once, and no published
//! delta is ever rewritten.
use super::execution::{finish_reason, prepare};
use super::{ApiResult, GenerationRequest, TextState};
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use infer_core::{Error, ErrorCode, RequestId};
use infer_models::{OutputStreamParser, StreamEvent, TextStreamDecoder};
use infer_runtime::EngineOutput;
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    convert::Infallible,
    sync::Arc,
    sync::atomic::{AtomicUsize, Ordering},
};
use tokio::sync::mpsc;

/// Terminal marker of an `OpenAI` event stream.
const DONE: &str = "[DONE]";

/// Lifecycle of one event stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// The role has not been published yet.
    Start,
    Running,
    /// The terminator has been queued.
    Done,
}

/// One streamed generation in progress.
struct Stream {
    request: RequestId,
    receiver: mpsc::Receiver<infer_core::Result<EngineOutput>>,
    delivery: crate::cpu::CpuPool,
    /// The decoder and parser leave this owner for the duration of a CPU job and come back.
    decoder: Option<TextStreamDecoder>,
    parser: Option<OutputStreamParser>,
    /// Serialized event payloads waiting to be written.
    queue: VecDeque<String>,
    id: String,
    object: &'static str,
    model: String,
    created: u64,
    chat: bool,
    include_usage: bool,
    /// Whether generated text is split into content, reasoning and calls. False publishes the
    /// decoded text unchanged, which is what the complete reply does when neither tools nor
    /// thinking mode are in play.
    parse: bool,
    prompt_tokens: Arc<AtomicUsize>,
    completion_tokens: usize,
    call_index: usize,
    phase: Phase,
}

impl Stream {
    /// Serialize one chunk of this response.
    fn chunk(&self, delta: &Value, finish_reason: Option<&str>) -> String {
        json!({
            "id":self.id, "object":self.object, "created":self.created, "model":self.model,
            "choices":[{"index":0, "delta":delta, "finish_reason":finish_reason, "logprobs":null}]
        })
        .to_string()
    }
    fn queue_chunk(&mut self, delta: &Value, finish_reason: Option<&str>) {
        let chunk = self.chunk(delta, finish_reason);
        self.queue.push_back(chunk);
    }
    /// Publish one parser event as a delta.
    fn publish(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::Text(text) => {
                let delta = if self.chat {
                    json!({"content": text})
                } else {
                    json!({"text": text})
                };
                self.queue_chunk(&delta, None);
            }
            StreamEvent::Reasoning(text) => {
                self.queue_chunk(&json!({"reasoning_content": text}), None);
            }
            StreamEvent::Call(call) => {
                // The function-parameter dialect needs the whole block before its arguments are
                // known, so a call is published once and complete. The index and the identifier
                // are the ones the complete reply would use, so the two forms agree.
                let delta = json!({"tool_calls":[{
                    "index": self.call_index,
                    "id": format!("call_{}_{}", self.request.get(), self.call_index),
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }]});
                self.call_index += 1;
                self.queue_chunk(&delta, None);
            }
        }
    }
    /// Queue an error event and end the stream.
    fn fail(&mut self, error: &Error) {
        self.queue.push_back(
            json!({"error":{"message":error.message, "code":format!("{:?}", error.code)}})
                .to_string(),
        );
        self.queue.push_back(DONE.to_string());
        self.phase = Phase::Done;
    }
    /// Terminal chunks: the finish reason, optional usage, and the stream terminator.
    fn finish(&mut self, reason: &str) {
        self.queue_chunk(&json!({}), Some(reason));
        if self.include_usage {
            let prompt = self.prompt_tokens.load(Ordering::Relaxed);
            self.queue.push_back(
                json!({
                    "id":self.id, "object":self.object, "created":self.created, "model":self.model,
                    "choices":[],
                    "usage":{
                        "prompt_tokens":prompt, "completion_tokens":self.completion_tokens,
                        "total_tokens":prompt + self.completion_tokens
                    }
                })
                .to_string(),
            );
        }
        self.queue.push_back(DONE.to_string());
        self.phase = Phase::Done;
    }
    /// Publish everything produced by one engine event.
    ///
    /// # Errors
    /// Returns a decoding or parsing error for a response that cannot be published incrementally.
    async fn advance(&mut self, event: EngineOutput) -> infer_core::Result<()> {
        match event {
            EngineOutput::Token { token, .. } => {
                let parse = self.parse;
                let delivery = self.delivery.clone();
                let decoder = self
                    .decoder
                    .take()
                    .ok_or_else(|| Error::invariant("stream decoder missing"))?;
                let parser = self
                    .parser
                    .take()
                    .ok_or_else(|| Error::invariant("stream parser missing"))?;
                let (decoder, parser, text, events) = delivery
                    .run(0, move |_| {
                        let mut decoder = decoder;
                        let mut parser = parser;
                        let text = decoder.push(token)?;
                        let events = if text.is_empty() {
                            Vec::new()
                        } else if parse {
                            parser.feed(&text)?
                        } else {
                            vec![StreamEvent::Text(text.clone())]
                        };
                        Ok((decoder, parser, text, events))
                    })
                    .await?;
                self.decoder = Some(decoder);
                self.parser = Some(parser);
                if !text.is_empty() {
                    self.completion_tokens += 1;
                }
                for event in events {
                    self.publish(event);
                }
            }
            EngineOutput::Finished(result) => {
                let reason = finish_reason(&result.reason)?;
                let parse = self.parse;
                let delivery = self.delivery.clone();
                let decoder = self
                    .decoder
                    .take()
                    .ok_or_else(|| Error::invariant("stream decoder missing"))?;
                let parser = self
                    .parser
                    .take()
                    .ok_or_else(|| Error::invariant("stream parser missing"))?;
                let (decoder, parser, tail, events) = delivery
                    .run(0, move |_| {
                        let mut decoder = decoder;
                        let mut parser = parser;
                        // Flush the held tail, then finish the parser so a withheld block is
                        // reported as truncated rather than published.
                        let tail = decoder.finish()?;
                        let mut events = if tail.is_empty() {
                            Vec::new()
                        } else if parse {
                            parser.feed(&tail)?
                        } else {
                            vec![StreamEvent::Text(tail.clone())]
                        };
                        if parse {
                            events.extend(parser.finish()?);
                        }
                        Ok((decoder, parser, tail, events))
                    })
                    .await?;
                self.decoder = Some(decoder);
                self.parser = Some(parser);
                if !tail.is_empty() {
                    self.completion_tokens += 1;
                }
                for event in events {
                    self.publish(event);
                }
                // A length stop stays `length` even when a complete call preceded the cap.
                let has_calls = self.call_index > 0;
                let reason = if has_calls && reason != "length" {
                    "tool_calls"
                } else {
                    reason
                };
                self.finish(reason);
            }
        }
        Ok(())
    }
}

/// Stream one chat or completion request as server-sent events.
///
/// # Errors
/// Returns the same validation errors as the complete reply, plus `unsupported` when the package
/// cannot publish a stable incremental decode.
pub(super) async fn stream(
    state: TextState,
    headers: HeaderMap,
    request: GenerationRequest,
    chat: bool,
) -> ApiResult<Response> {
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    let prepared = prepare(&state, &headers, request, chat).await?;
    let prefix = if chat { "chatcmpl" } else { "cmpl" };
    let object = if chat {
        "chat.completion.chunk"
    } else {
        "text_completion"
    };
    let events = futures_util::stream::unfold(
        Stream {
            request: prepared.request,
            receiver: prepared.receiver,
            delivery: state.delivery.clone(),
            decoder: Some(state.assets.stream_decoder()),
            parser: Some(OutputStreamParser::new(
                prepared.dialect,
                prepared.policy.declared().cloned(),
            )),
            queue: VecDeque::new(),
            id: format!("{prefix}-{}", prepared.request),
            object,
            model: prepared.model_name.clone(),
            created: prepared.created,
            chat,
            include_usage,
            parse: chat && (prepared.policy.parses() || prepared.enable_thinking),
            prompt_tokens: prepared.prompt_tokens,
            completion_tokens: 0,
            call_index: 0,
            phase: Phase::Start,
        },
        |mut stream| async move {
            loop {
                if let Some(data) = stream.queue.pop_front() {
                    return Some((Ok::<_, Infallible>(Event::default().data(data)), stream));
                }
                if stream.phase == Phase::Done {
                    return None;
                }
                if stream.phase == Phase::Start {
                    // The role is published before any content, as the protocol requires.
                    stream.phase = Phase::Running;
                    if stream.chat {
                        stream.queue_chunk(&json!({"role":"assistant"}), None);
                    }
                    continue;
                }
                match stream.receiver.recv().await {
                    None => {
                        let error = Error::new(
                            ErrorCode::Backend,
                            "request stream closed before completion",
                        );
                        stream.fail(&error);
                    }
                    Some(Err(error)) => stream.fail(&error),
                    Some(Ok(event)) => {
                        if let Err(error) = stream.advance(event).await {
                            stream.fail(&error);
                        }
                    }
                }
            }
        },
    );
    Ok(Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use infer_models::{ParsedToolCall, TextAssets};
    use std::sync::atomic::AtomicUsize;

    /// The GPT-2 byte alphabet: printable bytes keep their character, the rest are shifted past
    /// the printable range so every byte has a distinct character.
    fn byte_alphabet() -> Vec<char> {
        let mut printable: Vec<u8> = (b'!'..=b'~').collect();
        printable.extend(b'\xa1'..=b'\xac');
        printable.extend(b'\xae'..=b'\xff');
        let mut alphabet: Vec<char> = printable.iter().map(|byte| char::from(*byte)).collect();
        let shifted: Vec<u8> = (0u8..=255)
            .filter(|value| !printable.contains(value))
            .collect();
        alphabet.extend(shifted.iter().enumerate().map(|(index, _)| {
            char::from_u32(256 + u32::try_from(index).expect("alphabet index fits u32"))
                .expect("alphabet code is a scalar value")
        }));
        alphabet
    }

    /// A byte-level tokenizer over the whole byte alphabet, so any text round-trips one token per
    /// byte. The served packages use `ByteLevel` too; this one is small enough to build in a
    /// test.
    fn byte_assets() -> Arc<TextAssets> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let alphabet = byte_alphabet();
        let vocab: serde_json::Map<String, Value> = (0u16..=255)
            .map(|byte| (alphabet[byte as usize].to_string(), json!(u64::from(byte))))
            .collect();
        let tokenizer = json!({
            "version": "1.0",
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false},
            "post_processor": null,
            "decoder": {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false},
            "model": {"type":"BPE","dropout":null,"unk_token":null,
                "continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,
                "vocab":vocab,"merges":[]}
        });
        let root = std::env::temp_dir().join(format!(
            "infer-byte-tokenizer-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).expect("temporary package");
        std::fs::write(
            root.join("tokenizer.json"),
            serde_json::to_vec(&tokenizer).expect("tokenizer serializes"),
        )
        .expect("tokenizer written");
        Arc::new(TextAssets::open(&root, 4096).expect("byte tokenizer opens"))
    }

    /// A stream wired to a real decoder and parser, as the service runs it.
    fn parsing_stream(assets: &TextAssets, declared: &[&str]) -> Stream {
        let mut stream = stream();
        stream.decoder = Some(assets.stream_decoder());
        stream.parser = Some(OutputStreamParser::new(
            assets.tool_dialect(),
            Some(
                declared
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect::<std::collections::BTreeSet<_>>(),
            ),
        ));
        stream.parse = true;
        stream
    }

    fn stream() -> Stream {
        let (sender, receiver) = mpsc::channel(1);
        drop(sender);
        Stream {
            request: RequestId::new(7).unwrap(),
            receiver,
            delivery: crate::cpu::CpuPool::new(crate::cpu::CpuConfig::default()).unwrap(),
            decoder: None,
            parser: None,
            queue: VecDeque::new(),
            id: "chatcmpl-7".into(),
            object: "chat.completion.chunk",
            model: "served".into(),
            created: 1,
            chat: true,
            include_usage: false,
            prompt_tokens: Arc::new(AtomicUsize::new(3)),
            completion_tokens: 0,
            call_index: 0,
            parse: true,
            phase: Phase::Running,
        }
    }

    fn chunks(stream: &Stream) -> Vec<Value> {
        stream
            .queue
            .iter()
            .filter(|data| data.as_str() != DONE)
            .map(|data| serde_json::from_str(data).unwrap())
            .collect()
    }

    #[test]
    fn a_tool_call_delta_carries_the_complete_reply_identifier() {
        let mut stream = stream();
        stream.publish(StreamEvent::Call(ParsedToolCall {
            name: "get_weather".into(),
            arguments: "{\"city\":\"Paris\"}".into(),
        }));
        stream.publish(StreamEvent::Call(ParsedToolCall {
            name: "ping".into(),
            arguments: "{}".into(),
        }));
        stream.finish("tool_calls");
        let chunks = chunks(&stream);
        let first = &chunks[0]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(first["index"], 0);
        assert_eq!(first["id"], "call_7_0");
        assert_eq!(first["type"], "function");
        assert_eq!(first["function"]["name"], "get_weather");
        assert_eq!(first["function"]["arguments"], "{\"city\":\"Paris\"}");
        // A second call is appendable and keeps its own index and identifier.
        assert_eq!(
            chunks[1]["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_7_1"
        );
        assert_eq!(chunks[2]["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(stream.queue.back().unwrap(), DONE);
        assert_eq!(stream.phase, Phase::Done);
    }

    #[test]
    fn usage_is_published_once_when_requested() {
        let mut stream = stream();
        stream.include_usage = true;
        stream.completion_tokens = 4;
        stream.finish("stop");
        let chunks = chunks(&stream);
        assert_eq!(chunks[1]["choices"].as_array().unwrap().len(), 0);
        assert_eq!(chunks[1]["usage"]["prompt_tokens"], 3);
        assert_eq!(chunks[1]["usage"]["completion_tokens"], 4);
        assert_eq!(chunks[1]["usage"]["total_tokens"], 7);
    }

    #[test]
    fn a_failure_ends_the_stream_with_one_error_event() {
        let mut stream = stream();
        stream.fail(&Error::new(ErrorCode::Backend, "device fault"));
        let chunks = chunks(&stream);
        assert_eq!(chunks[0]["error"]["message"], "device fault");
        assert_eq!(chunks[0]["error"]["code"], "Backend");
        assert_eq!(stream.queue.back().unwrap(), DONE);
        assert_eq!(stream.phase, Phase::Done);
    }

    #[test]
    fn a_reasoning_delta_is_published_separately_from_content() {
        let mut stream = stream();
        stream.publish(StreamEvent::Reasoning("why".into()));
        stream.publish(StreamEvent::Text("answer".into()));
        let chunks = chunks(&stream);
        assert_eq!(chunks[0]["choices"][0]["delta"]["reasoning_content"], "why");
        assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "answer");
    }

    #[tokio::test]
    async fn streamed_tool_calls_are_decoded_parsed_and_published() {
        let assets = byte_assets();
        let text = concat!(
            "Checking.\n",
            "<tool_call>\n",
            "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n",
            "</tool_call>"
        );
        let tokens = assets.encode(text, false).expect("byte tokenizer encodes");
        assert!(tokens.len() > text.len() / 2, "one token per byte");
        let mut stream = parsing_stream(&assets, &["get_weather"]);
        for (index, token) in tokens.iter().enumerate() {
            stream
                .advance(EngineOutput::Token {
                    request: stream.request,
                    token: *token,
                    index,
                })
                .await
                .expect("token advances");
        }
        stream.finish("tool_calls");
        let chunks = chunks(&stream);
        let content: String = chunks
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(content, "Checking.");
        let call = chunks
            .iter()
            .find_map(|chunk| chunk["choices"][0]["delta"]["tool_calls"].as_array())
            .expect("one tool call delta")
            .first()
            .cloned()
            .expect("one call");
        let call = &call;
        assert_eq!(call["id"], "call_7_0");
        assert_eq!(call["function"]["name"], "get_weather");
        // The arguments are the model's own bytes, not a re-serialized object.
        assert_eq!(call["function"]["arguments"], "{\"city\":\"Paris\"}");
        assert_eq!(
            chunks.last().unwrap()["choices"][0]["finish_reason"],
            "tool_calls"
        );
        assert_eq!(stream.queue.back().unwrap(), DONE);
    }

    #[tokio::test]
    async fn a_streamed_call_that_is_not_declared_stays_in_the_content() {
        let assets = byte_assets();
        let text = "<tool_call>{\"name\":\"invented\",\"arguments\":{}}</tool_call>";
        let tokens = assets.encode(text, false).expect("byte tokenizer encodes");
        let mut stream = parsing_stream(&assets, &["declared"]);
        for (index, token) in tokens.iter().enumerate() {
            stream
                .advance(EngineOutput::Token {
                    request: stream.request,
                    token: *token,
                    index,
                })
                .await
                .expect("token advances");
        }
        stream.finish("stop");
        let chunks = chunks(&stream);
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk["choices"][0]["delta"]["tool_calls"].is_null())
        );
        let content: String = chunks
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert!(content.contains("invented"), "{content:?}");
    }
}
