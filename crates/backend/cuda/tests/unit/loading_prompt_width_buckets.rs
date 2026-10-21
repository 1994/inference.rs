use super::LoadOptions;
use infer_core::Result;

#[test]
fn every_explicit_prompt_bucket_resolves_with_or_without_mtp() -> Result<()> {
    for prefill_width in [32, 64, 128, 256, 512, 1024, 2048] {
        for mtp_depth in 0..=2 {
            let mut options = LoadOptions {
                prefill_width,
                mtp_depth,
                ..LoadOptions::default()
            };
            options.resolve_verification()?;
            assert_eq!(options.prefill_width, prefill_width);
            assert_eq!(
                options.verification_width,
                if mtp_depth == 0 { 0 } else { mtp_depth + 1 }
            );
        }
    }
    Ok(())
}

#[test]
fn unimplemented_prompt_widths_stay_rejected() {
    for prefill_width in [2, 31, 33, 129, 511, 513, 1023, 1025, 2049] {
        let options = LoadOptions {
            prefill_width,
            ..LoadOptions::default()
        };
        assert!(options.validate().is_err(), "width {prefill_width}");
    }
}
