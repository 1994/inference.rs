//! GPU numerical checks require Linux and an NVIDIA device.
#[cfg(target_os = "linux")]
#[path = "resident_checks/mod.rs"]
mod checks;

#[cfg(target_os = "linux")]
fn main() -> infer_core::Result<()> {
    checks::run()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("CUDA resident checks require Linux");
    std::process::exit(1);
}
