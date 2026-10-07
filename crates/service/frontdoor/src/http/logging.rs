//! HTTP request logging.
//!
//! One line per request, produced by a router layer so every route is covered. Success is
//! `info`, a client rejection is `warn` and a server failure is `error`, so an operator sees a
//! rejected request without turning on debug logging. Request identifiers and token counts stay
//! in the engine's own `infer::request` events, which the access line already correlates by time.
use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use std::time::Instant;

/// Log method, path, status and latency for one request.
pub(super) async fn log_request(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let started = Instant::now();
    let response = next.run(request).await;
    let status = response.status();
    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    if status.is_server_error() {
        tracing::error!(
            target: "infer::http",
            %method,
            path,
            status = status.as_u16(),
            elapsed_ms,
            "request failed"
        );
    } else if status.is_client_error() {
        tracing::warn!(
            target: "infer::http",
            %method,
            path,
            status = status.as_u16(),
            elapsed_ms,
            "request rejected"
        );
    } else {
        tracing::info!(
            target: "infer::http",
            %method,
            path,
            status = status.as_u16(),
            elapsed_ms,
            "request"
        );
    }
    response
}
