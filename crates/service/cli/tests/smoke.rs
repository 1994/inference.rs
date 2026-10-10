use std::process::Command;
#[cfg(feature = "test-backends")]
use std::{io::Write, process::Stdio};

#[cfg(feature = "test-backends")]
#[test]
fn verify_command_reports_runtime_invariance() {
    let output = Command::new(env!("CARGO_BIN_EXE_infer"))
        .args(["--backend", "test-cpu", "verify"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["passed"], true);
    assert_eq!(report["state_leaks"], 0);
}
#[cfg(feature = "test-backends")]
#[test]
fn agent_validates_envelopes_and_uses_standard_error_codes() {
    let mut process = Command::new(env!("CARGO_BIN_EXE_infer"))
        .args(["--backend", "test-cpu", "agent"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    process.stdin.take().unwrap().write_all(b"{\"id\":1,\"method\":\"runtime.inspect\"}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"unknown\"}\n{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"runtime.inspect\"}\n").unwrap();
    let output = process.wait_with_output().unwrap();
    assert!(output.status.success());
    let lines = String::from_utf8(output.stdout).unwrap();
    let responses: Vec<serde_json::Value> = lines
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(responses[0]["error"]["code"], -32600);
    assert_eq!(responses[1]["error"]["code"], -32601);
    assert_eq!(responses[2]["result"]["active_requests"], 0);
}

#[test]
fn cuda_remains_primary_and_explicit_unavailable_backend_does_not_fallback() {
    let binary = env!("CARGO_BIN_EXE_infer");
    let doctor = Command::new(binary).arg("doctor").output().unwrap();
    assert!(doctor.status.success());
    let report: serde_json::Value = serde_json::from_slice(&doctor.stdout).unwrap();
    assert_eq!(report["backends"]["primary_target"], "cuda");
    assert_eq!(
        report["backends"]["auto_priority"],
        serde_json::json!(["cuda", "metal"])
    );
    assert_eq!(
        report["backends"]["supported"],
        serde_json::json!(["cuda", "metal"])
    );
    assert!(report["backends"].get("host").is_none());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples");
    let output = Command::new(binary)
        // Make the unavailable-device case deterministic even on GPU hosts.
        .env("CUDA_VISIBLE_DEVICES", "-1")
        .args(["--backend", "cuda", "run", "--package"])
        .arg(root.join("qwen-hybrid-tiny"))
        .arg("--requests")
        .arg(root.join("requests.json"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    let expected_code = if cfg!(all(target_os = "linux", feature = "cuda")) {
        "Backend"
    } else {
        "Unsupported"
    };
    assert_eq!(error["code"], expected_code, "{error}");
    assert!(
        output.stdout.is_empty(),
        "failed backend must not emit results"
    );
    for backend in ["metal", "cuda"] {
        let output = Command::new(binary)
            .args(["--backend", backend, "verify"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("requires --package"));
    }
}

#[cfg(not(feature = "test-backends"))]
#[test]
fn production_binary_rejects_cpu_choices_and_implicit_fixture_execution() {
    let binary = env!("CARGO_BIN_EXE_infer");
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples");
    for backend in ["host", "cpu", "test-cpu"] {
        let output = Command::new(binary)
            .args(["--backend", backend, "run", "--package"])
            .arg(root.join("qwen-hybrid-tiny"))
            .arg("--requests")
            .arg(root.join("requests.json"))
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("invalid value"));
    }
    let output = Command::new(binary)
        .args(["run", "--requests"])
        .arg(root.join("requests.json"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires --package"));
    let output = Command::new(binary).arg("--help").output().unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(!help.contains("test-cpu") && !help.contains("host-memory-mib"));
}
