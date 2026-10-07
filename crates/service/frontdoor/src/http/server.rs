//! Server HTTP adapter.
use super::{RuntimeHandle, logging::log_request, router, router_with_text};
use axum::middleware::from_fn;
use infer_core::{Error, ErrorCode, Result};

///
/// # Errors
/// Returns an I/O error if the HTTP server fails.
pub async fn serve(
    listener: tokio::net::TcpListener,
    handle: RuntimeHandle,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    axum::serve(listener, router(handle).layer(from_fn(log_request)))
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| Error::new(ErrorCode::Backend, e.to_string()))
}
///
/// # Errors
/// Returns an I/O error if the HTTP server fails.
pub async fn serve_with_text(
    listener: tokio::net::TcpListener,
    handle: RuntimeHandle,
    assets: std::sync::Arc<infer_models::TextAssets>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    axum::serve(
        listener,
        router_with_text(handle, assets).layer(from_fn(log_request)),
    )
    .with_graceful_shutdown(shutdown)
    .await
    .map_err(|e| Error::new(ErrorCode::Backend, e.to_string()))
}
