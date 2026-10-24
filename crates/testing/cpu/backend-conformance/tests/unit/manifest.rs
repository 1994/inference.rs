use super::*;

use crate::support::{MockKernels, caps};

fn failed(report: &Report, name: &str) -> bool {
    report.failures().any(|check| check.name == name)
}

#[test]
fn accepts_a_consistent_manifest() {
    let report = kernel_manifest(&MockKernels::good(), &caps());
    assert!(report.is_ok(), "{}", report.summary());
}

#[test]
fn rejects_a_kernel_for_another_backend() {
    let report = kernel_manifest(&MockKernels::foreign(), &caps());
    assert!(
        failed(&report, "manifest.descriptors"),
        "{}",
        report.summary()
    );
    assert!(report.require().is_err());
}

#[test]
fn rejects_a_registration_the_capabilities_cannot_select() {
    let report = kernel_manifest(&MockKernels::unselectable(), &caps());
    assert!(
        failed(&report, "manifest.selectable"),
        "{}",
        report.summary()
    );
}

#[test]
fn rejects_duplicate_kernel_identifiers() {
    let report = kernel_manifest(&MockKernels::duplicate(), &caps());
    assert!(
        failed(&report, "manifest.identifiers"),
        "{}",
        report.summary()
    );
    assert!(
        failed(&report, "manifest.registration"),
        "{}",
        report.summary()
    );
}
