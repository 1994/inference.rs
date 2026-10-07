use clap::Parser;
use infer_models::SamplingOverrides;
use std::path::PathBuf;

#[expect(
    clippy::struct_excessive_bools,
    reason = "Independent diagnostic CLI switches have explicit clap dependency/conflict checks"
)]
#[derive(Parser)]
pub struct Options {
    pub model: PathBuf,
    pub prompt: String,
    pub max_new_tokens: usize,
    #[arg(long, default_value_t = 0)]
    pub mtp: usize,
    #[arg(long)]
    pub temperature: Option<f32>,
    #[arg(long)]
    pub top_k: Option<usize>,
    #[arg(long)]
    pub top_p: Option<f32>,
    #[arg(long)]
    pub min_p: Option<f32>,
    #[arg(long)]
    pub presence_penalty: Option<f32>,
    #[arg(long)]
    pub repetition_penalty: Option<f32>,
    #[arg(long)]
    pub thinking: Option<bool>,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long)]
    pub dataset: Option<PathBuf>,
    #[arg(long)]
    pub mlp_graph: bool,
    #[arg(long, conflicts_with = "mlp_graph")]
    pub device_graph: bool,
    #[arg(long, requires = "device_graph")]
    pub fp8_kv: bool,
    #[arg(long, requires = "device_graph")]
    pub sequential_verify: bool,
    #[arg(long, default_value_t = 1)]
    pub prefill_batch: usize,
    #[arg(long, conflicts_with_all = ["device_graph", "dataset"])]
    pub tune_projections: bool,
    #[arg(long, requires = "device_graph")]
    pub tuning: Option<PathBuf>,
    #[arg(long, requires = "mlp_graph")]
    pub mlp_pdl: bool,
}

impl Options {
    pub fn validate(&self) -> infer_core::Result<()> {
        if ![1, 3, 32].contains(&self.prefill_batch)
            || (self.prefill_batch > 1 && !self.device_graph)
        {
            return Err(infer_core::Error::invalid(
                "prefill batch must be 1, or 3/32 with device graph",
            ));
        }
        if self.prefill_batch == 3 && self.mtp > 0 && self.mtp != 2 && !self.sequential_verify {
            return Err(infer_core::Error::invalid(
                "prefill batch 3 requires MTP depth 0 or 2, or sequential verification",
            ));
        }
        if !(1..=1024).contains(&self.max_new_tokens) || self.mtp > 8 {
            return Err(infer_core::Error::invalid(
                "max_new_tokens must be 1..=1024; mtp depth must be 0..=8",
            ));
        }
        Ok(())
    }

    pub const fn execution_mode(&self) -> &'static str {
        if self.device_graph {
            "CUDA resident token graphs; target and MTP auxiliary operations on GPU"
        } else {
            "diagnostic: GPU projections + CPU auxiliary operations"
        }
    }

    pub const fn limitation(&self) -> &'static str {
        if self.prefill_batch == 32 {
            "BF16-rounded prompt projection inputs, F32 accumulation; decode uses F32 GEMV; CPU sampling; KV format recorded separately"
        } else if self.fp8_kv {
            "F32 activations; target FP8 KV, draft F32 KV; weight-only quantization; CPU sampling; GEMV projections (prefill_batch records prompt grouping)"
        } else if self.device_graph {
            "F32 activations and KV; weight-only quantization; CPU sampling; GEMV projections (prefill_batch records prompt grouping)"
        } else {
            "GPU projections, CPU auxiliary ops; weight-only quantization; sequential MTP verification"
        }
    }

    pub const fn prefill_math(&self) -> &'static str {
        if self.prefill_batch == 32 {
            "BF16 tensor-core GEMM with F32 accumulation, including masked tail and lm_head"
        } else {
            "F32 GEMV accumulation; three-row weight reuse when prefill_batch=3"
        }
    }

    pub const fn verification(&self) -> &'static str {
        if self.device_graph && self.mtp > 0 && !self.sequential_verify {
            "node-major batched GEMV verification; device recurrent prefix checkpoints; exact rejection sampling"
        } else {
            "sequential target verification with exact rejection sampling"
        }
    }

    pub const fn sampling(&self) -> SamplingOverrides {
        SamplingOverrides {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            presence_penalty: self.presence_penalty,
            repetition_penalty: self.repetition_penalty,
            enable_thinking: self.thinking,
            seed: self.seed,
            do_sample: None,
            eos_token: None,
            eos_tokens: None,
        }
    }
}
