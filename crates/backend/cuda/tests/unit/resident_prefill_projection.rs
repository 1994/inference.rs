use super::{DENSE_REFERENCE, narrow_dense_layout, with_dense_reference};

#[test]
fn narrow_dense_scope_matches_only_measured_profile() {
    assert!(narrow_dense_layout("sm_120", (64, 48, 5120)));
    for architecture in ["sm_100", "sm_90", "sm_121", ""] {
        assert!(!narrow_dense_layout(architecture, (64, 48, 5120)));
    }
    for shape in [
        (1, 48, 5120),
        (3, 48, 5120),
        (4, 48, 5120),
        (32, 48, 5120),
        (128, 48, 5120),
        (256, 48, 5120),
        (64, 32, 5120),
        (64, 64, 5120),
        (64, 48, 4096),
    ] {
        assert!(!narrow_dense_layout("sm_120", shape));
    }
}

#[test]
fn dense_reference_restores_after_nested_scope_and_unwind() {
    assert!(!DENSE_REFERENCE.get());
    with_dense_reference(|| {
        assert!(DENSE_REFERENCE.get());
        with_dense_reference(|| assert!(DENSE_REFERENCE.get()));
        assert!(DENSE_REFERENCE.get());
    });
    assert!(!DENSE_REFERENCE.get());
    let failure = std::panic::catch_unwind(|| with_dense_reference(|| panic!("reference unwind")));
    assert!(failure.is_err());
    assert!(!DENSE_REFERENCE.get());
}
