use super::*;
use std::path::Path;

#[test]
fn model_path_serves_with_automatic_defaults() {
    let cli = Cli::try_parse_from(["infer", "/models/example"]).unwrap();
    assert_eq!(cli.backend, backend::BackendChoice::Auto);
    assert!(!cli.no_autotune);
    assert!(cli.block_size.is_none());
    assert!(cli.max_num_batched_tokens.is_none());
    let Command::Serve(options) = cli.into_command().unwrap() else {
        panic!("expected server")
    };
    assert_eq!(
        options.package.as_deref(),
        Some(Path::new("/models/example"))
    );
    assert!(options.config.is_none());
    assert_eq!(options.listen.to_string(), "127.0.0.1:8080");
}

#[test]
fn shorthand_preserves_overrides_and_legacy_commands() {
    for args in [
        vec![
            "infer",
            "./model",
            "--listen",
            "127.0.0.1:9000",
            "--config",
            "custom.json",
        ],
        vec![
            "infer",
            "serve",
            "--package",
            "./model",
            "--listen",
            "127.0.0.1:9000",
            "--config",
            "custom.json",
        ],
    ] {
        let Command::Serve(options) = Cli::try_parse_from(args).unwrap().into_command().unwrap()
        else {
            panic!("expected server")
        };
        assert_eq!(options.package.as_deref(), Some(Path::new("./model")));
        assert_eq!(options.config.as_deref(), Some(Path::new("custom.json")));
        assert_eq!(options.listen.port(), 9000);
    }
    assert!(matches!(
        Cli::try_parse_from(["infer", "--backend", "cuda", "doctor"])
            .unwrap()
            .into_command()
            .unwrap(),
        Command::Doctor
    ));
}

#[test]
fn ambiguous_or_missing_launch_input_is_rejected() {
    for args in [
        vec!["infer"],
        vec!["infer", "--listen", "127.0.0.1:9000"],
        vec!["infer", "./model", "--package", "./other"],
    ] {
        assert!(Cli::try_parse_from(args).is_err());
    }
    assert!(
        Cli::try_parse_from(["infer", "./model", "doctor"])
            .unwrap()
            .into_command()
            .is_err()
    );
    assert!(Cli::try_parse_from(["infer", "./model", "--backend", "cuda"]).is_ok());
    assert!(Cli::try_parse_from(["infer", "--backend", "cuda", "./model"]).is_ok());
    assert!(Cli::try_parse_from(["infer", "--", "./serve"]).is_ok());
}

#[test]
fn deployment_length_limits_are_parsed_as_separate_budgets() {
    let cli = Cli::try_parse_from([
        "infer",
        "./model",
        "--max-model-len",
        "8192",
        "--max-output-tokens",
        "1024",
    ])
    .unwrap();
    assert_eq!(cli.max_model_len, Some(8192));
    assert_eq!(cli.max_output_tokens, Some(1024));
    // Zero is a configuration error the length resolver reports, not a clap special value.
    let zero = Cli::try_parse_from(["infer", "./model", "--max-model-len", "0"]).unwrap();
    assert_eq!(zero.max_model_len, Some(0));
    // The total context and output cap stay distinct from the per-step batching budget.
    let batching =
        Cli::try_parse_from(["infer", "./model", "--max-num-batched-tokens", "2048"]).unwrap();
    assert_eq!(batching.max_num_batched_tokens, Some(2048));
    assert!(batching.max_model_len.is_none());
    assert!(batching.max_output_tokens.is_none());
}
