//! Bounded engine service with asynchronous CPU workers and HTTP adapters.
mod actor;
mod constants;
pub mod cpu;
mod http;
mod ingress;
pub use actor::RuntimeHandle;
pub use http::{
    HttpError, NativeJsonAdapter, NativeTextRequest, router, router_with_text, serve,
    serve_with_text,
};
