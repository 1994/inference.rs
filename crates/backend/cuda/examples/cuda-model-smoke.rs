#[cfg(target_os = "linux")]
mod model_smoke;

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use infer_models::{ChatMessage, ChatOptions, TextAssets};
    use model_smoke::{mtp::Mtp, options::Options};
    use std::time::Instant;
    let options = Options::parse();
    if options.dataset.is_some() {
        return model_smoke::suite::run(&options);
    }
    if options.max_new_tokens == 0 || options.max_new_tokens > 1024 || options.mtp > 8 {
        return Err("max_new_tokens must be 1..=1024; mtp depth must be 0..=8".into());
    }
    let text = TextAssets::open(&options.model, 4096)?;
    let resolved = text.generation.resolve(&options.sampling())?;
    let input = text.encode_chat(
        &[ChatMessage {
            role: "user".into(),
            content: options.prompt.clone(),
        }],
        &ChatOptions {
            enable_thinking: resolved.enable_thinking,
            ..Default::default()
        },
    )?;
    let started = Instant::now();
    let capacity = input.len() + options.max_new_tokens + options.mtp;
    let mut model = model_smoke::Model::load(&options.model, capacity)?;
    let mut mtp = if options.mtp > 0 {
        Some(Mtp::load(&options.model, &model, capacity)?)
    } else {
        None
    };
    let load_seconds = started.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut logits = Vec::new();
    for (position, token) in input.iter().enumerate() {
        if position > 0
            && let Some(mtp) = &mut mtp
        {
            mtp.step(*token, &model.hidden, position, false)?;
        }
        logits = model.step(*token, position, position + 1 == input.len())?;
        eprintln!("prefill {}/{}", position + 1, input.len());
    }
    let prefill_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let result = model_smoke::decode::decode(
        &mut model,
        mtp.as_mut(),
        &input,
        logits,
        &resolved.sampling,
        options.max_new_tokens,
        options.mtp,
    )?;
    let decode_seconds = start.elapsed().as_secs_f64();
    let result = serde_json::json!({
        "mode": "diagnostic: GPU projections + Rust CPU auxiliary operations",
        "precision": "dequantized weights with F32 activations; dynamic activation quantization not enabled",
        "mtp_depth": options.mtp, "generation": resolved,
        "verification": "sequential target verification with exact rejection sampling; draft KV restore/replay",
        "timing_note": "diagnostic wall time includes host work, transfers and first-use JIT; not a production benchmark",
        "target": model.device.target(), "model": options.model, "prompt": options.prompt,
        "input_tokens": input, "decode": result, "text": text.decode(&result.tokens, true)?,
        "load_seconds": load_seconds, "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("CUDA model diagnostics require Linux");
    std::process::exit(1);
}
