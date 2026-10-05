#[cfg(target_os = "linux")]
mod affinity;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "linux")]
mod mask;
#[cfg(target_os = "linux")]
mod numa;
#[cfg(target_os = "linux")]
pub(super) use linux::{Guard, allowed_cpus};
#[cfg(target_os = "macos")]
pub(super) use macos::{Guard, allowed_cpus};
