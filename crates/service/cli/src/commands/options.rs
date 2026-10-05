//! Options commands.
use std::path::PathBuf;

#[derive(clap::Args)]
pub struct ServeOptions {
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub listen: std::net::SocketAddr,
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub package: Option<PathBuf>,
    #[arg(long, alias = "host-memory-mib", default_value_t = 512)]
    pub device_memory_mib: u64,
}
#[derive(clap::Args)]
pub struct RunOptions {
    #[arg(long)]
    pub requests: PathBuf,
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub model: Option<PathBuf>,
    #[arg(long, conflicts_with = "model")]
    pub package: Option<PathBuf>,
    #[arg(long, alias = "host-memory-mib", default_value_t = 512)]
    pub device_memory_mib: u64,
    #[arg(long)]
    pub op_trace: Option<PathBuf>,
    #[arg(long)]
    pub journal: Option<PathBuf>,
    #[arg(long)]
    pub snapshot: Option<PathBuf>,
    #[arg(long)]
    pub events: Option<PathBuf>,
}
#[cfg(feature = "test-backends")]
#[derive(clap::Args)]
pub struct DemoOptions {
    #[arg(long)]
    pub journal: Option<PathBuf>,
    #[arg(long)]
    pub snapshot: Option<PathBuf>,
}
#[derive(clap::Args)]
pub struct VerifyOptions {
    #[arg(long, requires = "candidate")]
    pub reference: Option<PathBuf>,
    #[arg(long, requires = "reference")]
    pub candidate: Option<PathBuf>,
    #[arg(long, default_value_t = 1e-5)]
    pub atol: f64,
    #[arg(long, default_value_t = 1e-5)]
    pub rtol: f64,
    #[arg(long, conflicts_with = "reference", requires = "golden")]
    pub package: Option<PathBuf>,
    #[arg(long, requires = "package")]
    pub golden: Option<PathBuf>,
    #[arg(long, alias = "host-memory-mib", default_value_t = 512)]
    pub device_memory_mib: u64,
}
#[derive(clap::Args)]
pub struct ReplayOptions {
    #[arg(
        long,
        conflicts_with = "snapshot",
        required_unless_present = "snapshot"
    )]
    pub journal: Option<PathBuf>,
    #[arg(long, conflicts_with = "journal")]
    pub snapshot: Option<PathBuf>,
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub model: Option<PathBuf>,
    #[arg(long, conflicts_with = "model")]
    pub package: Option<PathBuf>,
    #[arg(long, alias = "host-memory-mib", default_value_t = 512)]
    pub device_memory_mib: u64,
}
#[derive(clap::Args)]
pub struct ProfileOptions {
    #[arg(long)]
    pub journal: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub package: Option<PathBuf>,
    #[arg(long, alias = "host-memory-mib", default_value_t = 512)]
    pub device_memory_mib: u64,
}
#[derive(clap::Args)]
pub struct BenchmarkOptions {
    #[arg(long, default_value_t = 32)]
    pub requests: usize,
    #[arg(long, default_value_t = 16)]
    pub input_tokens: usize,
    #[arg(long, default_value_t = 16)]
    pub output_tokens: usize,
    #[arg(long, default_value_t = 1_000_000)]
    pub ttft_slo_us: u64,
    #[arg(long, default_value_t = 100_000)]
    pub tpot_slo_us: u64,
    #[arg(long)]
    pub output: Option<PathBuf>,
    #[arg(long)]
    pub package: Option<PathBuf>,
    #[arg(long, alias = "host-memory-mib", default_value_t = 512)]
    pub device_memory_mib: u64,
}
#[derive(clap::Args)]
pub struct CompareOptions {
    #[arg(long)]
    pub baseline: PathBuf,
    #[arg(long)]
    pub candidate: PathBuf,
    #[arg(long)]
    pub correctness_passed: bool,
    #[arg(long, default_value_t = 0.05)]
    pub max_p99_regression: f64,
}
#[derive(clap::Args)]
pub struct InspectModelOptions {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub index: Option<PathBuf>,
    #[arg(long, default_value_t = 4096)]
    pub context_tokens: usize,
    #[arg(long, default_value_t = 1)]
    pub sequences: usize,
    #[arg(long, default_value_t = 32)]
    pub device_memory_gib: u64,
}
#[derive(clap::Args)]
pub struct AgentOptions {
    #[arg(long)]
    pub package: Option<PathBuf>,
    #[arg(long, alias = "host-memory-mib", default_value_t = 512)]
    pub device_memory_mib: u64,
    #[arg(long, default_value_t = 0)]
    pub probe_memory_mib: u64,
}
#[derive(clap::Args)]
pub struct InspectPackageOptions {
    #[arg(long)]
    pub package: PathBuf,
    /// Fail inspection when resident weights exceed this optional budget.
    #[arg(long)]
    pub device_memory_mib: Option<u64>,
    #[arg(long, default_value_t = 4)]
    pub staging_memory_mib: usize,
    /// Inspect expanded F32 storage instead of the source storage format.
    #[arg(long)]
    pub weights_f32: bool,
}
#[derive(clap::Args)]
pub struct TokenizeOptions {
    #[arg(long)]
    pub package: PathBuf,
    #[arg(
        long,
        conflicts_with = "messages",
        required_unless_present = "messages"
    )]
    pub text: Option<String>,
    #[arg(long, conflicts_with = "text")]
    pub messages: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    pub enable_thinking: bool,
}
