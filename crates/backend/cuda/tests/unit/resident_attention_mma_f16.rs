use super::*;

#[test]
fn fp16_selection_identity_is_distinct_from_bf16_and_canonical() {
    assert_ne!(SCHEMA, crate::tuning::attention::mma::SCHEMA);
    assert_ne!(SCHEMA, crate::tuning::attention::SCHEMA);
    assert!(SCHEMA.contains("fa4-f16"));
    assert!(crate::tuning::attention::mma::SCHEMA.contains("fa4-bf16"));
    assert!(crate::tuning::attention::SCHEMA.contains("prefill-fp8-kv"));
    assert_ne!(
        artifact_identity(),
        crate::resident::attention_mma::artifact_identity()
    );
    assert_ne!(
        Choice::MmaF16,
        Choice::Mma {
            variant: crate::resident::attention_mma::catalog::Variant::Sm120
        }
    );
    assert_eq!(
        serde_json::to_string(&Choice::MmaF16).unwrap(),
        "\"MmaF16\""
    );
}

#[test]
fn fp16_support_is_only_the_validated_sm120_geometry() {
    let compiled = cfg!(all(target_os = "linux", target_arch = "x86_64"));
    assert_eq!(
        supported(
            "sm_120",
            crate::resident::attention_mma::catalog::HEADS,
            crate::resident::attention_mma::catalog::KV_HEADS,
            crate::resident::attention_mma::catalog::DIM,
            128,
            0
        ),
        compiled
    );
    assert!(!supported(
        "sm_120",
        crate::resident::attention_mma::catalog::HEADS,
        crate::resident::attention_mma::catalog::KV_HEADS,
        crate::resident::attention_mma::catalog::DIM,
        128,
        64
    ));
    assert!(!supported(
        "sm_120",
        8,
        crate::resident::attention_mma::catalog::KV_HEADS,
        crate::resident::attention_mma::catalog::DIM,
        128,
        0
    ));
    assert!(!supported(
        "sm_90",
        crate::resident::attention_mma::catalog::HEADS,
        crate::resident::attention_mma::catalog::KV_HEADS,
        crate::resident::attention_mma::catalog::DIM,
        128,
        0
    ));
    assert!(!supported(
        "sm_100",
        crate::resident::attention_mma::catalog::HEADS,
        crate::resident::attention_mma::catalog::KV_HEADS,
        crate::resident::attention_mma::catalog::DIM,
        128,
        0
    ));
    assert!(!supported(
        "sm_103",
        crate::resident::attention_mma::catalog::HEADS,
        crate::resident::attention_mma::catalog::KV_HEADS,
        crate::resident::attention_mma::catalog::DIM,
        128,
        0
    ));
    // Lane zero, misaligned lanes and windows are canonical fallbacks.
    assert!(!supported(
        "sm_120",
        crate::resident::attention_mma::catalog::HEADS,
        crate::resident::attention_mma::catalog::KV_HEADS,
        crate::resident::attention_mma::catalog::DIM,
        0,
        0
    ));
    assert!(!supported(
        "sm_120",
        crate::resident::attention_mma::catalog::HEADS,
        crate::resident::attention_mma::catalog::KV_HEADS,
        crate::resident::attention_mma::catalog::DIM,
        65,
        0
    ));
}

#[test]
fn fp16_choice_never_takes_the_bf16_tiled_or_oxide_paths() {
    assert!(!Choice::MmaF16.uses_oxide());
    assert_eq!(Choice::MmaF16.query_tile(), None);
}
