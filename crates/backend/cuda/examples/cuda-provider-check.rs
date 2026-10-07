//! Actual-model acceptance through the engine SPI, including request ownership checks.
#[cfg(target_os = "linux")]
#[path = "provider_check/mod.rs"]
mod check;
#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    check::run()
}
#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("CUDA provider check requires Linux");
    std::process::exit(1);
}
