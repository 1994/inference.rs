//! Server commands.
use super::{ServeOptions, axum_serve, backend};
use crate::support::{configured_engine, read_json};
#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
use infer_core::ErrorCode;
use infer_core::{Error, Result};

#[cfg(any(
    target_os = "macos",
    feature = "test-backends",
    all(target_os = "linux", feature = "cuda")
))]
pub fn serve(options: ServeOptions, backend_choice: backend::Selection) -> Result<()> {
    let ServeOptions {
        listen,
        config: config_path,
        package,
        host_memory_mib,
    } = options;
    let engine = configured_engine(
        config_path.as_deref().map(read_json).transpose()?,
        None,
        package.as_deref(),
        host_memory_mib,
        backend_choice,
    )?;
    let assets = package
        .as_deref()
        .filter(|p| p.join("tokenizer.json").exists())
        .map(|p| infer_models::TextAssets::open(p, engine.model().max_sequence))
        .transpose()?
        .map(std::sync::Arc::new);
    tokio::runtime::Runtime::new()
        .map_err(|e| Error::new(ErrorCode::Backend, e.to_string()))?
        .block_on(async move {
            let listener = tokio::net::TcpListener::bind(listen)
                .await
                .map_err(|e| Error::new(ErrorCode::Backend, e.to_string()))?;
            let handle = infer_frontdoor::RuntimeHandle::start(engine)?;
            let shutdown = handle.clone();
            eprintln!("Native inference server listening on http://{listen}");
            axum_serve(listener, handle, shutdown, assets).await
        })?;

    Ok(())
}
