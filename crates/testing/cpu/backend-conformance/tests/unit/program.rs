use super::*;

use crate::support::{MockBackend, caps};
use infer_core::Result;

fn failed(report: &Report, name: &str) -> bool {
    report.failures().any(|check| check.name == name)
}

#[test]
fn accepts_a_consistent_program() -> Result<()> {
    let backend = MockBackend::new()?;
    let report = program_contract(&backend, backend.model(), backend.program(), &caps());
    assert!(report.is_ok(), "{}", report.summary());
    Ok(())
}

#[test]
fn rejects_a_backend_that_accepts_every_program() -> Result<()> {
    let backend = MockBackend::loose()?;
    let report = program_contract(&backend, backend.model(), backend.program(), &caps());
    for name in [
        "program.rejects_model",
        "program.rejects_backend",
        "program.rejects_precision",
        "program.rejects_kernel",
        "program.rejects_dataflow",
    ] {
        assert!(
            failed(&report, name),
            "missing {name}: {}",
            report.summary()
        );
    }
    assert!(report.require().is_err());
    Ok(())
}

#[test]
fn rejects_a_program_with_another_kernel_id() -> Result<()> {
    let backend = MockBackend::new()?;
    let mut program = backend.program().clone();
    if let Some(first) = program.operations.first_mut() {
        first.kernel = KernelId::new(UNKNOWN_KERNEL_ID).unwrap_or(first.kernel);
    }
    assert!(backend.validate_program(backend.model(), &program).is_err());
    Ok(())
}
