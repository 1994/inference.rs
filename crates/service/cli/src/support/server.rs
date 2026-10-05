//! Server CLI support.
#[cfg(any(target_os = "macos", feature = "test-backends"))]
use infer_core::Result;

#[cfg(any(target_os = "macos", feature = "test-backends"))]
pub async fn axum_serve(
    listener: tokio::net::TcpListener,
    handle: infer_frontdoor::RuntimeHandle,
    shutdown: infer_frontdoor::RuntimeHandle,
    assets: Option<std::sync::Arc<infer_models::TextAssets>>,
) -> Result<()> {
    let stop = async move {
        let _ = tokio::signal::ctrl_c().await;
        let _ = shutdown.shutdown().await;
    };
    if let Some(assets) = assets {
        infer_frontdoor::serve_with_text(listener, handle, assets, stop).await
    } else {
        infer_frontdoor::serve(listener, handle, stop).await
    }
}
