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
    /// Physical GPU KV block count; automatic when omitted.
    #[arg(long, global = true)]
    pub(super) kv_cache_blocks: Option<usize>,
    /// Tokens per KV block and logical state page.
    #[arg(long, global = true)]
    pub(super) kv_page_tokens: Option<usize>,
    /// Maximum token rows per native GPU prefill chunk.
    #[arg(long, global = true)]
    pub(super) prefill_chunk_tokens: Option<usize>,
    /// Bounded temporary weight upload memory, independent of device residency.
    #[arg(long, global = true)]
    pub(super) upload_staging_mib: Option<usize>,
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
