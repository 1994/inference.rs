//! Resident-model, sequential dataset diagnostic. Warmup and model load are excluded.
use super::{Model, decode, mtp::Mtp, options::Options};
use infer_models::{ChatMessage, ChatOptions, TextAssets};
use serde_json::{Value, json};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub fn run(options: &Options) -> Result<(), Box<dyn std::error::Error>> {
    if !(1..=1024).contains(&options.max_new_tokens) || options.mtp > 8 {
        return Err("invalid token limit or MTP depth".into());
    }
    let manifest: Value = serde_json::from_slice(&std::fs::read(
        options.dataset.as_ref().ok_or("dataset required")?,
    )?)?;
    let records = manifest["samples"].as_array().ok_or("samples required")?;
    if records.is_empty() {
        return Err("empty dataset".into());
    }
    let text = TextAssets::open(&options.model, 4096)?;
    let resolved = text.generation.resolve(&options.sampling())?;
    let mut inputs = Vec::new();
    for record in records {
        inputs.push(text.encode_chat(
            &[ChatMessage {
                role: "user".into(),
                content: record["prompt"].as_str().ok_or("prompt required")?.into(),
            }],
            &ChatOptions {
                enable_thinking: resolved.enable_thinking,
                ..Default::default()
            },
        )?);
    }
    let capacity =
        inputs.iter().map(Vec::len).max().unwrap_or(0) + options.max_new_tokens + options.mtp;
    let started = Instant::now();
    let mut model = Model::load(&options.model, capacity)?;
    let mut draft = if options.mtp > 0 {
        Some(Mtp::load(&options.model, &model, capacity)?)
    } else {
        None
    };
    let load_seconds = started.elapsed().as_secs_f64();
    let empty = model.state.clone();
    let draft_empty = draft.as_ref().map(Mtp::checkpoint);
    let mut results = Vec::new();
    let mut suite_started = Instant::now();
    let mut measured_start_unix = 0.0;
    // One complete request warms kernels; state is restored before every request.
    for index in 0..=inputs.len() {
        model.state.clone_from(&empty);
        model.hidden.clear();
        if let (Some(draft), Some(empty)) = (&mut draft, &draft_empty) {
            draft.restore(empty.clone());
            draft.model.hidden.clear();
        }
        if index == 1 {
            suite_started = Instant::now();
            measured_start_unix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
        }
        let sample = index.saturating_sub(1);
        let input = &inputs[sample];
        let request_start_unix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
        let started = Instant::now();
        let mut logits = Vec::new();
        for (position, token) in input.iter().enumerate() {
            if position > 0
                && let Some(draft) = &mut draft
            {
                draft.step(*token, &model.hidden, position, false)?;
            }
            logits = model.step(*token, position, position + 1 == input.len())?;
        }
        let prefill_seconds = started.elapsed().as_secs_f64();
        let decoded = decode::decode(
            &mut model,
            draft.as_mut(),
            input,
            logits,
            &resolved.sampling,
            options.max_new_tokens,
            options.mtp,
        )?;
        let seconds = started.elapsed().as_secs_f64();
        if index > 0 {
            results.push(json!({"id": records[sample]["id"], "input_tokens": input,
                "text": text.decode(&decoded.tokens, true)?, "decode": decoded,
                "prefill_seconds": prefill_seconds, "wall_seconds": seconds,
                "started_unix": request_start_unix,
                "finish_reason": if decoded.tokens.last().is_some_and(|t| resolved.sampling.is_eos(*t)) {
                    "stop" } else { "length" }}));
        }
        eprintln!("dataset {index}/{}: {seconds:.3}s", inputs.len());
    }
    let wall_seconds = suite_started.elapsed().as_secs_f64();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "framework": "rust-diagnostic", "model": options.model, "target": model.device.target(),
            "limitation": "GPU projections, CPU auxiliary ops; weight-only quantization; sequential MTP verification",
            "dataset": manifest, "generation": resolved, "mtp_depth": options.mtp,
            "max_new_tokens": options.max_new_tokens, "concurrency": 1, "warmup_requests": 1,
        "load_seconds": load_seconds, "wall_seconds": wall_seconds, "results": results,
        "measured_start_unix": measured_start_unix,
            "completed": true
        }))?
    );
    Ok(())
}
