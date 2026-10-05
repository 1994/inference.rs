//! Dispatch executable responsibilities.
#[cfg(any(target_os = "macos", feature = "test-backends"))]
use crate::support::agent;
use crate::{arguments::Cli, arguments::Command, backend, commands, support::print};
use clap::Parser;
use infer_core::Result;
use std::io;

pub fn run() {
    let cli = Cli::parse();
    let selection = backend::Selection {
        kind: cli.backend,
        kv_cache_blocks: cli.kv_cache_blocks,
        page_tokens: cli.kv_page_tokens,
        prefill_chunk_tokens: cli.prefill_chunk_tokens,
        upload_staging_mib: cli.upload_staging_mib,
    };
    if let Err(error) = execute(cli.command, selection) {
        let _ = serde_json::to_writer(io::stderr().lock(), &error);
        eprintln!();
        std::process::exit(1);
    }
}

pub fn execute(command: Command, backend_choice: backend::Selection) -> Result<()> {
    match command {
        #[cfg(any(target_os = "macos", feature = "test-backends"))]
        Command::Serve(options) => commands::serve(options, backend_choice),
        #[cfg(any(target_os = "macos", feature = "test-backends"))]
        Command::Run(options) => commands::run(options, backend_choice),
        #[cfg(feature = "test-backends")]
        Command::Demo(options) => commands::demo(options, backend_choice),
        Command::Verify(options) => commands::verify(options, backend_choice),
        #[cfg(any(target_os = "macos", feature = "test-backends"))]
        Command::Replay(options) => commands::replay(options, backend_choice),
        #[cfg(any(target_os = "macos", feature = "test-backends"))]
        Command::Profile(options) => commands::profile(options, backend_choice),
        #[cfg(any(target_os = "macos", feature = "test-backends"))]
        Command::Benchmark(options) => commands::benchmark(options, backend_choice),
        Command::Compare(options) => commands::compare(options),
        Command::InspectModel(options) => commands::inspect_model(options),
        Command::InspectPackage(options) => commands::inspect_package(options),
        Command::Tokenize(options) => commands::tokenize(options),
        Command::Doctor => print(
            &serde_json::json!({"host_arch":std::env::consts::ARCH,"host_os":std::env::consts::OS,"backends":backend::catalog(),"qwen38_config_import":true,"safetensors_weight_binding":true,"native_tokenizer":true,"incremental_hybrid_state":true,"cuda_execution":"deferred until RTX 5090 migration","qwen38_27b_execution_validated":false,"native_workloads":["generate","embed","rerank","decision"]}),
        ),
        #[cfg(any(target_os = "macos", feature = "test-backends"))]
        Command::Agent(options) => agent(
            options.package.as_deref(),
            options.device_memory_mib,
            options.probe_memory_mib,
            backend_choice,
        ),
        #[cfg(not(any(target_os = "macos", feature = "test-backends")))]
        _ => Err(Error::unsupported(match backend_choice.kind {
            backend::BackendChoice::Cuda => {
                "native NVIDIA CUDA executor is not installed; CUDA implementation is deferred to RTX 5090 migration"
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
