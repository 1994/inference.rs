//! Server CLI support.
#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "cuda")))]
use infer_core::Result;
#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "cuda")))]
use std::sync::Arc;
#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "cuda")))]
use tokio::net::TcpListener;
#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "cuda")))]
use tokio::signal::ctrl_c;

#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "cuda")))]
pub async fn axum_serve(
    listener: TcpListener,
    handle: infer_frontdoor::RuntimeHandle,
    shutdown: infer_frontdoor::RuntimeHandle,
    assets: Option<Arc<infer_models::TextAssets>>,
) -> Result<()> {
    let stop = async move {
        let _ = ctrl_c().await;
        let _ = shutdown.shutdown().await;
    };
    if let Some(assets) = assets {
        infer_frontdoor::serve_with_text(listener, handle, assets, stop).await
    } else {
        infer_frontdoor::serve(listener, handle, stop).await
    }
}
