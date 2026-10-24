use super::*;

use crate::support::MockBackend;
use infer_core::Result;

fn failed(report: &Report, name: &str) -> bool {
    report.failures().any(|check| check.name == name)
}

#[test]
fn accepts_a_consistent_backend() -> Result<()> {
    let mut backend = MockBackend::new()?;
    let report = backend_contract(&mut backend, 8);
    assert!(report.is_ok(), "{}", report.summary());
    Ok(())
}

#[test]
fn rejects_an_empty_identity() -> Result<()> {
    let mut backend = MockBackend::anonymous()?;
    let report = backend_contract(&mut backend, 8);
    assert!(failed(&report, "state.identity"), "{}", report.summary());
    Ok(())
}

#[test]
fn rejects_a_quote_below_its_own_recipe() -> Result<()> {
    let mut backend = MockBackend::new()?.with_discount(1);
    let report = backend_contract(&mut backend, 8);
    assert!(failed(&report, "state.quote_floor"), "{}", report.summary());
    Ok(())
}

#[test]
fn rejects_a_zero_capacity() -> Result<()> {
    let mut backend = MockBackend::new()?;
    let report = backend_contract(&mut backend, 0);
    assert!(failed(&report, "state.capacity"), "{}", report.summary());
    Ok(())
}
