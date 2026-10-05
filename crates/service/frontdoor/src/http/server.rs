//! Server HTTP adapter.
use super::{RuntimeHandle, router, router_with_text};
use infer_core::{Error, ErrorCode, Result};

///
/// # Errors
/// Returns an I/O error if the HTTP server fails.
pub async fn serve(
    listener: tokio::net::TcpListener,
    handle: RuntimeHandle,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    axum::serve(listener, router(handle))
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
    axum::serve(listener, router_with_text(handle, assets))
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| Error::new(ErrorCode::Backend, e.to_string()))
}
