//! Fixed tool-output replay: compilation/cache/mask costs, never model throughput.
use crate::allocator;
use infer_core::{Error, Result};
use infer_models::{SamplingOverrides, TextAssets, ToolConstraints, ToolDialect};
use serde_json::{Value, json};
use std::fs::read;
use std::path::Path;
use std::time::Instant;

const CONTEXT: usize = 4096;
const COLD_RUNS: usize = 10;
const HOT_RUNS: usize = 100;

fn ns(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}
fn summary(mut samples: Vec<u64>) -> Value {
    samples.sort_unstable();
    json!({"samples":samples.len(),"p50_ns":samples[samples.len()/2],"p95_ns":samples[samples.len()*95/100],"max_ns":samples.last()})
}
pub fn run(root: &Path) -> Result<Value> {
    let assets = TextAssets::open(root, CONTEXT)?;
    let sampling = assets
        .generation
        .resolve(&SamplingOverrides::default())?
        .sampling;
    let spec = ToolConstraints {
        required: true,
        named: None,
        parallel: false,
        tools: json!([
            {"type":"function","function":{"name":"ping","strict":true,"parameters":{"type":"object","properties":{},"required":[],"additionalProperties":false}}}
        ]),
    };
    let trace = match assets.tool_dialect() {
        ToolDialect::JsonBlock => r#"<tool_call>{"name":"ping","arguments":{}}</tool_call>"#,
        ToolDialect::FunctionParameters => "<tool_call><function=ping></function></tool_call>",
    };
    let mut tokens = assets.encode(trace, false)?;
    tokens.push(
        sampling
            .eos_token
            .or_else(|| sampling.eos_tokens.first().copied())
            .ok_or_else(|| Error::invalid("benchmark requires EOS"))?,
    );
    // Package vocabulary, including added tokens; backend padding is not measured.
    let tokenizer: Value = serde_json::from_slice(
        &read(root.join("tokenizer.json")).map_err(|e| Error::invalid(e.to_string()))?,
    )
    .map_err(|e| Error::invalid(e.to_string()))?;
    let ids = tokenizer["model"]["vocab"]
        .as_object()
        .ok_or_else(|| Error::invalid("benchmark requires vocabulary"))?
        .values()
        .filter_map(Value::as_u64);
    let added = tokenizer["added_tokens"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|value| value["id"].as_u64());
    let vocabulary = usize::try_from(ids.chain(added).max().unwrap_or(0))
        .map_err(|_| Error::invalid("vocabulary too large"))?
        .checked_add(1)
        .ok_or_else(|| Error::invalid("vocabulary overflow"))?;
    let mut cold = Vec::new();
    for _ in 0..COLD_RUNS {
        let fresh = TextAssets::open(root, CONTEXT)?;
        let start = Instant::now();
        let handle = fresh.tool_constraint(&spec, &sampling, false, false, vocabulary)?;
        cold.push(ns(start));
        drop(handle);
    }
    let mut hot = Vec::with_capacity(HOT_RUNS);
    let mut masks = Vec::with_capacity(HOT_RUNS * tokens.len());
    // Warm the shared grammar before timing cache hits.
    drop(assets.tool_constraint(&spec, &sampling, false, false, vocabulary)?);
    allocator::start();
    for _ in 0..HOT_RUNS {
        let start = Instant::now();
        let mut handle = assets.tool_constraint(&spec, &sampling, false, false, vocabulary)?;
        hot.push(ns(start));
        for (position, &token) in tokens.iter().enumerate() {
            let start = Instant::now();
            handle.select(position, |mask| {
                if mask[token as usize / 32] & (1 << (token % 32)) == 0 {
                    return Err(Error::invariant("trace token disallowed"));
                }
                Ok(token)
            })?;
            masks.push(ns(start));
        }
    }
    let counts = allocator::stop();
    Ok(
        json!({"scope":"Fixed ping tool trace, no GPU/network/sampling; cold excludes TextAssets loading; mask includes commit; allocations include cache hits, matchers excluding preallocated sample storage", "tokenizer_fingerprint":assets.fingerprint,"vocabulary":vocabulary,"trace_tokens":tokens.len(),"cold":summary(cold),"cached":summary(hot),"mask_commit":summary(masks),"allocator":counts}),
    )
}
