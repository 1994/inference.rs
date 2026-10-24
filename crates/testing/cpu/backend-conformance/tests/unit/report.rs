use super::*;

#[test]
fn required_failures_block_optional_ones_do_not() {
    let mut report = Report::new();
    report.pass("passes");
    report.optional("optional", Status::Failed("informative".into()));
    report.skip("skipped", "not applicable");
    assert!(report.is_ok());
    assert!(report.summary().contains("0 failed"));

    report.fail("required", "broken");
    assert!(!report.is_ok());
    let error = report.require().unwrap_err();
    assert!(error.message.contains("required"));
}

#[test]
fn summary_lists_required_failures_only() {
    let mut report = Report::new();
    report.fail("first", "alpha");
    report.optional("second", Status::Failed("beta".into()));
    let summary = report.summary();
    assert!(summary.contains("first: alpha"));
    assert!(!summary.contains("second: beta"));
}

#[test]
fn extended_reports_keep_every_check() {
    let mut left = Report::new();
    left.pass("left");
    let mut right = Report::new();
    right.skip("right", "not applicable");
    left.extend(right);
    assert_eq!(left.checks().len(), 2);
    assert!(left.is_ok());
}
