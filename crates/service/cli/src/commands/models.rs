//! Models commands.
use super::{InspectModelOptions, InspectPackageOptions, TokenizeOptions, print, read_json};
use infer_core::{Error, ModelId, Result};
use infer_models::{QwenProvider, SafetensorsIndex, memory_estimate};

pub fn inspect_model(options: InspectModelOptions) -> Result<()> {
    let InspectModelOptions {
        config,
        index,
        context_tokens,
        sequences,
        device_memory_gib,
    } = options;
    let bytes = std::fs::read(config).map_err(|e| Error::invalid(e.to_string()))?;
    let model = QwenProvider.import_manifest(ModelId::new(1)?, &bytes)?;
    let index: Option<SafetensorsIndex> = index
        .map(|p| {
            std::fs::read(p)
                .map_err(|e| Error::invalid(e.to_string()))
                .and_then(|b| SafetensorsIndex::parse(&b))
        })
        .transpose()?;
    let bytes = device_memory_gib
        .checked_mul(1 << 30)
        .ok_or_else(|| Error::invalid("device memory overflow"))?;
    let memory = index
        .as_ref()
        .map(|i| {
            memory_estimate(
                &model.model,
                i.weight_bytes()?,
                context_tokens,
                sequences,
                1 << 30,
                bytes,
            )
        })
        .transpose()?;
    print(
        &serde_json::json!({"model":model,"memory":memory,"weight_shards":index.map(|i|i.shards()),"execution_supported":false,"next_requirement":"complete Qwen3.8 weight package and RTX 5090 CUDA/quantized execution validation"}),
    )?;

    Ok(())
}
pub fn inspect_package(options: InspectPackageOptions) -> Result<()> {
    let package = infer_models::QwenPackage::open(options.package, ModelId::new(1)?)?;
    let budget = options
        .device_memory_mib
        .map(|n| {
            n.checked_mul(1024 * 1024)
                .ok_or_else(|| Error::invalid("weight memory budget overflow"))
        })
        .transpose()?
        .unwrap_or(u64::MAX);
    let staging = options
        .staging_memory_mib
        .checked_mul(1024 * 1024)
        .ok_or_else(|| Error::invalid("weight staging budget overflow"))?;
    let plan = infer_models::WeightLoadPlan::build(
        &package,
        infer_models::LoadOptions {
            storage: if options.weights_f32 {
                infer_models::WeightStorage::F32
            } else {
                infer_models::WeightStorage::Native
            },
            resident_budget_bytes: budget,
            staging_budget_bytes: staging,
            chunk_bytes: (staging / if options.weights_f32 { 3 } else { 1 }).min(4 * 1024 * 1024),
        },
    )?;
    print(
        &serde_json::json!({"manifest":package.manifest,"program":package.graph,"weight_load":plan,"scope":"metadata preflight; weight residency excludes execution scratch, KV, and request state; vision/MTP excluded"}),
    )?;

    Ok(())
}
pub fn tokenize(options: TokenizeOptions) -> Result<()> {
    let TokenizeOptions {
        package,
        text,
        messages,
        enable_thinking,
    } = options;
    let config_bytes =
        std::fs::read(package.join("config.json")).map_err(|e| Error::invalid(e.to_string()))?;
    let model = QwenProvider.import_manifest(ModelId::new(1)?, &config_bytes)?;
    let assets = infer_models::TextAssets::open(package, model.model.max_sequence)?;
    let (rendered, tokens) = if let Some(text) = text {
        (text.clone(), assets.encode(&text, true)?)
    } else {
        let messages: Vec<infer_models::ChatMessage> = read_json(
            messages
                .as_deref()
                .ok_or_else(|| Error::invariant("validated argument"))?,
        )?;
        let rendered = assets.render_chat(
            &messages,
            &infer_models::ChatOptions {
                enable_thinking,
                ..Default::default()
            },
        )?;
        let tokens = assets.encode(&rendered, false)?;
        (rendered, tokens)
    };
    print(
        &serde_json::json!({"rendered":rendered,"tokens":tokens,"tokenizer_fingerprint":assets.fingerprint}),
    )?;

    Ok(())
}
