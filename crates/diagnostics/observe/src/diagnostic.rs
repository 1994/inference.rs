use infer_core::{Error, RequestId, StepId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticCode {
    AdmissionRejected,
    SubmissionFailed,
    CompletionFailed,
    InvalidCompletion,
    SubmissionTimeout,
    ProgressStall,
    InvariantViolation,
    OutputBackpressure,
    DeadlineExceeded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub sequence: u64,
    pub timestamp_us: u64,
    pub code: DiagnosticCode,
    pub severity: Severity,
    pub request: Option<RequestId>,
    pub step: Option<StepId>,
    pub error: Error,
    pub progress_epoch: u64,
    pub source: String,
    pub checkpoint_available: bool,
    pub resource_release_pending: bool,
}
