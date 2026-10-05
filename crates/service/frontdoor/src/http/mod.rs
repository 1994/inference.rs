//! Native JSON, text and observability HTTP adapters.
//! Native HTTP and SSE adapters backed by a bounded, single-owner engine actor.
#[cfg(test)]
mod tests;
use crate::{RuntimeHandle, cpu};
use error::HttpResult;
use native::trace_parent;
use observability::{diagnostics, events, metrics, observability, otlp, timeline};

mod error;
mod native;
mod observability;
mod openai;
mod protocol;
mod server;
mod text;

pub use error::HttpError;
pub use native::router;
pub use protocol::NativeJsonAdapter;
pub use server::{serve, serve_with_text};
pub use text::{NativeTextRequest, router_with_text};
