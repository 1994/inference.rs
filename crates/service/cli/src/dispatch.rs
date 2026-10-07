//! Dispatch executable responsibilities.
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
use crate::support::agent;
use crate::{arguments::Cli, arguments::Command, backend, commands, support::print};
use clap::Parser;
use infer_core::Result;
use std::io;

pub fn run() {
    let cli = Cli::parse();
    let selection = backend::Selection {
        kind: cli.backend,
        num_gpu_blocks_override: cli.num_gpu_blocks_override,
        block_size: cli.block_size,
        max_num_batched_tokens: cli.max_num_batched_tokens,
        upload_staging_mib: cli.upload_staging_mib,
        num_speculative_tokens: cli.num_speculative_tokens,
        gpu_memory_utilization: cli.gpu_memory_utilization,
        autotune: !cli.no_autotune,
    };
    if let Err(error) = execute(
        cli.into_command().unwrap_or_else(|error| error.exit()),
        selection,
    ) {
        let _ = serde_json::to_writer(io::stderr().lock(), &error);
        eprintln!();
        std::process::exit(1);
    }
}

// Host-only builds route the selection into subcommands that take it by value but never
// consume it, which the ownership lint flags there and only there.
#[cfg_attr(
    not(any(
        target_os = "macos",
        feature = "test-backends",
        all(target_os = "linux", feature = "cuda")
    )),
    expect(
        clippy::needless_pass_by_value,
        reason = "The dispatcher owns the single request-local selection in host-only builds"
    )
)]
pub fn execute(command: Command, backend_choice: backend::Selection) -> Result<()> {
    match command {
        #[cfg(not(any(
            target_os = "macos",
            feature = "test-backends",
            all(target_os = "linux", feature = "cuda")
        )))]
        Command::Run(options) if options.package.is_none() => Err(infer_core::Error::invalid(
            "device backend requires --package; CPU fixtures require a test-backends build",
        )),
        #[cfg(any(
            target_os = "macos",
            feature = "test-backends",
            all(target_os = "linux", feature = "cuda")
        ))]
        Command::Serve(options) => commands::serve(options, backend_choice),
        #[cfg(any(
            target_os = "macos",
            feature = "test-backends",
            all(target_os = "linux", feature = "cuda")
        ))]
        Command::Run(options) => commands::run(options, backend_choice),
        #[cfg(feature = "test-backends")]
        Command::Demo(options) => commands::demo(options, backend_choice),
        Command::Verify(options) => commands::verify(options, &backend_choice),
        #[cfg(any(
            target_os = "macos",
            feature = "test-backends",
            all(target_os = "linux", feature = "cuda")
        ))]
        Command::Replay(options) => commands::replay(options, backend_choice),
        #[cfg(any(
            target_os = "macos",
            feature = "test-backends",
            all(target_os = "linux", feature = "cuda")
        ))]
        Command::Profile(options) => commands::profile(options, backend_choice),
        #[cfg(any(
            target_os = "macos",
            feature = "test-backends",
            all(target_os = "linux", feature = "cuda")
        ))]
        Command::Benchmark(options) => commands::benchmark(options, backend_choice),
        Command::Compare(options) => commands::compare(options),
        Command::InspectModel(options) => commands::inspect_model(options),
        Command::InspectPackage(options) => commands::inspect_package(options),
        Command::Tokenize(options) => commands::tokenize(options),
        Command::Doctor => print(
            &serde_json::json!({"host_arch":std::env::consts::ARCH,"host_os":std::env::consts::OS,"backends":backend::catalog(),"qwen38_config_import":true,"safetensors_weight_binding":true,"native_tokenizer":true,"incremental_hybrid_state":true,"cuda_execution":"resident CUDA BackendProvider; requires Linux --features cuda","qwen38_27b_execution_validated":false,"native_workloads":["generate","embed","rerank","decision"]}),
        ),
        #[cfg(any(
            target_os = "macos",
            feature = "test-backends",
            all(target_os = "linux", feature = "cuda")
        ))]
        Command::Agent(options) => agent(
            options.package.as_deref(),
            options.host_memory_mib,
            options.probe_memory_mib,
            backend_choice,
        ),
        #[cfg(not(any(
            target_os = "macos",
            feature = "test-backends",
            all(target_os = "linux", feature = "cuda")
        )))]
        _ => Err(infer_core::Error::unsupported(match backend_choice.kind {
            backend::BackendChoice::Cuda => {
                "CUDA execution requires Linux and a build with --features cuda"
            }
            backend::BackendChoice::Metal => {
                "Metal backend requires macOS and a supported Metal device"
            }
            backend::BackendChoice::Auto => {
                "no supported GPU backend available (CUDA or Metal); CPU execution is test-only"
            }
        })),
    }
}
