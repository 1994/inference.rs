//! Process logging setup and human-readable sizes for log lines.
//!
//! The binary owns the subscriber; backends, the runtime and the HTTP frontdoor only emit
//! `tracing` events. Lines go to stderr, so a command's JSON result on stdout stays
//! machine-readable. The default level is `info`; `INFER_LOG` (or `RUST_LOG`) selects more
//! detail, for example `INFER_LOG=infer=debug` adds per-weight load progress and per-request
//! detail. A second `init` is ignored, so a re-entrant path cannot replace its own reporting.
use std::io::{IsTerminal, Write};
use tracing_subscriber::EnvFilter;

/// Install the process subscriber, reporting to stderr.
pub fn init() {
    let filter = EnvFilter::try_from_env("INFER_LOG")
        .or_else(|_| EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_ansi(std::io::stderr().is_terminal())
        .try_init();
}

/// Flush stderr before a hard exit, so a reported failure is never lost to buffering.
pub fn flush() {
    let _ = std::io::stderr().flush();
}

/// MiB in one GiB, for sizes that round below a GiB.
#[cfg(all(target_os = "linux", feature = "cuda"))]
const MIB_PER_GIB: f64 = 1024.0;

/// GiB as a floating-point divisor for human-readable sizes.
#[cfg(all(target_os = "linux", feature = "cuda"))]
const GIB: f64 = MIB_PER_GIB * MIB_PER_GIB * MIB_PER_GIB;

/// Human-readable size, rounded to one decimal place above a GiB.
#[cfg(all(target_os = "linux", feature = "cuda"))]
#[must_use]
pub fn human_bytes(bytes: u64) -> String {
    #[expect(
        clippy::cast_precision_loss,
        reason = "Device and payload sizes are far below f64 mantissa precision"
    )]
    let gib = bytes as f64 / GIB;
    if gib >= 1.0 {
        format!("{gib:.1} GiB")
    } else {
        #[expect(
            clippy::cast_precision_loss,
            reason = "Device and payload sizes are far below f64 mantissa precision"
        )]
        let mib = bytes as f64 / (GIB / MIB_PER_GIB);
        format!("{mib:.0} MiB")
    }
}

#[cfg(all(test, target_os = "linux", feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn sizes_stay_readable() {
        assert_eq!(human_bytes(0), "0 MiB");
        assert_eq!(
            human_bytes(crate::constants::GIB_U64 + crate::constants::GIB_U64 / 2),
            "1.5 GiB"
        );
    }
}
