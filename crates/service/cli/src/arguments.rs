//! Arguments executable responsibilities.
use super::{backend, commands};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "infer",
    version,
    about = "Rust-native inference runtime with Safetensors-backed incremental execution across native backends."
)]
pub struct Cli {
    #[arg(long, global = true, value_enum, default_value = "auto")]
    pub(super) backend: backend::BackendChoice,
    /// Number of GPU blocks to use, overriding the profiled count; automatic when omitted.
    #[arg(long, global = true)]
    pub(super) num_gpu_blocks_override: Option<usize>,
    /// Token block size for contiguous chunks of tokens.
    #[arg(long, global = true)]
    pub(super) block_size: Option<usize>,
    /// Maximum number of tokens batched together per step.
    #[arg(long, global = true)]
    pub(super) max_num_batched_tokens: Option<usize>,
    /// Bounded temporary weight upload memory, independent of device residency.
    #[arg(long, global = true)]
    pub(super) upload_staging_mib: Option<usize>,
    /// Keep the conservative built-in tile instead of measuring the best one per projection.
    #[arg(long, global = true)]
    pub(super) no_autotune: bool,
    /// Draft tokens proposed per step by the MTP head; 0 disables speculation.
    #[arg(long, global = true, default_value_t = 0)]
    pub(super) num_speculative_tokens: usize,
    /// Fraction of device memory the engine may use, as in vLLM (0 disables the cap).
    #[arg(long, global = true, default_value_t = crate::constants::DEFAULT_GPU_MEMORY_UTILIZATION)]
    pub(super) gpu_memory_utilization: f64,
    #[command(subcommand)]
    pub(super) command: Command,
}
#[derive(Subcommand)]
pub enum Command {
    /// Serve native token HTTP/SSE and package tokenizer text requests.
    Serve(commands::ServeOptions),
    /// Run canonical requests with a Safetensors package or reference fixture.
    Run(commands::RunOptions),
    /// Exercise Generate, Embed, Rerank and Decision together.
    #[cfg(feature = "test-backends")]
    Demo(commands::DemoOptions),
    /// Compare numeric arrays, or verify batch/chunk invariance of the fixture.
    Verify(commands::VerifyOptions),
    /// Replay journal actions, or resume a quiescent execution checkpoint.
    Replay(commands::ReplayOptions),
    /// Export a CPU/control timeline in Chrome Trace / Perfetto JSON format.
    Profile(commands::ProfileOptions),
    /// Measure B3 CPU reference engine throughput and request latency.
    Benchmark(commands::BenchmarkOptions),
    /// Gate a candidate benchmark with correctness and latency constraints.
    Compare(commands::CompareOptions),
    /// Import an HF Qwen configuration and optionally check a weight index budget.
    InspectModel(commands::InspectModelOptions),
    /// Inspect supported execution paths and host environment.
    Doctor,
    /// Local JSON-RPC over stdin/stdout for inspect, explain, replay and control.
    Agent(commands::AgentOptions),
    /// Read shard headers and validate every text-backbone weight binding.
    InspectPackage(commands::InspectPackageOptions),
    /// Encode plain text or package chat template with native Rust assets.
    Tokenize(commands::TokenizeOptions),
}
