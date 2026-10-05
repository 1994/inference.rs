use super::*;
#[test]
fn rejects_nonfinite_and_detects_divergence() {
    assert!(compare(&[1.0], &[f32::NAN], 0.0, 0.0).is_err());
    assert!(!compare(&[1.0], &[1.1], 0.01, 0.0).unwrap().passed);
}
#[test]
fn calibration_known_values() {
    let r = calibration(&[vec![0.5, 0.5], vec![0.0, 1.0]], &[0, 1], 10).unwrap();
    assert_eq!(r.brier, 0.25);
    assert_eq!(r.accuracy, 1.0);
    assert_eq!(r.ece, 0.25);
}
#[test]
fn progress_guard_only_trips_for_feasible_runnable_work() {
    let mut g = ProgressGuard::new(2).unwrap();
    g.observe(1, false, false).unwrap();
    g.observe(1, true, false).unwrap();
    assert!(g.observe(1, true, false).is_err());
    g.observe(1, true, true).unwrap();
}
