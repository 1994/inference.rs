//! Json CLI support.
use infer_core::{Error, ErrorCode, Result};
use serde::{Serialize, de::DeserializeOwned};
use std::{io, io::Write, path::Path};

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let file = std::fs::File::open(path)
        .map_err(|e| Error::invalid(format!("{}: {e}", path.display())))?;
    if file
        .metadata()
        .map_err(|e| Error::invalid(e.to_string()))?
        .len()
        > 64 * 1024 * 1024
    {
        return Err(Error::new(ErrorCode::Capacity, "input JSON exceeds 64 MiB"));
    }
    serde_json::from_reader(file).map_err(|e| Error::invalid(format!("{}: {e}", path.display())))
}
#[cfg(any(target_os = "macos", feature = "test-backends"))]
pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let file = std::fs::File::create(path).map_err(|e| Error::invalid(e.to_string()))?;
    serde_json::to_writer_pretty(file, value).map_err(|e| Error::invalid(e.to_string()))
}
pub fn print(value: &impl Serialize) -> Result<()> {
    let mut out = io::stdout().lock();
    serde_json::to_writer_pretty(&mut out, value).map_err(|e| Error::invalid(e.to_string()))?;
    writeln!(out).map_err(|e| Error::invalid(e.to_string()))
}
