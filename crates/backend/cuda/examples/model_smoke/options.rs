use clap::Parser;
use infer_models::SamplingOverrides;
use std::path::PathBuf;

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
}

impl Options {
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
