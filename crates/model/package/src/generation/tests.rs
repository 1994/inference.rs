use super::*;

#[test]
fn mode_defaults_and_explicit_zero_overrides_are_distinct() -> Result<()> {
    let defaults = GenerationDefaults {
        base: Sampling {
            temperature: 1.0,
            top_p: 0.95,
            top_k: Some(20),
            eos_tokens: vec![10, 11],
            ..Sampling::default()
        },
        thinking: true,
        qwen38: true,
        sources: BTreeMap::new(),
    };
    let thinking = defaults.resolve(&SamplingOverrides::default())?;
    assert!(thinking.enable_thinking);
    assert_eq!(thinking.sampling.temperature, 1.0);
    let overrides = SamplingOverrides {
        enable_thinking: Some(false),
        ..SamplingOverrides::default()
    };
    let instruct = defaults.resolve(&overrides)?;
    assert_eq!(instruct.sampling.temperature, 0.7);
    assert_eq!(instruct.sampling.top_p, 0.8);
    assert_eq!(instruct.sampling.presence_penalty, 1.5);
    let custom = defaults.resolve(&SamplingOverrides {
        temperature: Some(0.0),
        top_k: Some(0),
        presence_penalty: Some(0.0),
        ..overrides
    })?;
    assert_eq!(custom.sampling.temperature, 0.0);
    assert_eq!(custom.sampling.top_k, None);
    assert_eq!(custom.sampling.presence_penalty, 0.0);
    assert!(custom.sampling.is_eos(11));
    assert!(
        defaults
            .resolve(&SamplingOverrides {
                top_p: Some(0.0),
                ..SamplingOverrides::default()
            })
            .is_err()
    );
    Ok(())
}
