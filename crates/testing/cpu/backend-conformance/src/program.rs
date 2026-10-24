//! Compiled-program identity checks.
use crate::report::Report;
use infer_core::{KernelId, ModelId, Result};
use infer_ir::{BackendKind, DeviceCapabilities, ExecutionProgram, ModelIr, PrecisionPlan};
use infer_spi::BackendProvider;

/// Kernel id guaranteed not to belong to a provider table.
const UNKNOWN_KERNEL_ID: u64 = 999_999;

/// Check that a backend accepts its compiled program and rejects mismatched identities.
///
/// The adapter supplies the program it compiled for the same model. The checks probe the four
/// mismatches that otherwise surface as wrong numbers or device faults much later: a foreign
/// backend, a foreign model, a precision the device cannot compute, and an unknown kernel id.
#[must_use]
pub fn program_contract<B: BackendProvider>(
    backend: &B,
    model: &ModelIr,
    program: &ExecutionProgram,
    caps: &DeviceCapabilities,
) -> Report {
    let mut report = Report::new();
    if let Err(error) = backend.validate_program(model, program) {
        report.fail("program.accepts", error.to_string());
        return report;
    }
    report.pass("program.accepts");
    check_graph(backend, model, program, &mut report);
    check_model_rejection(backend, model, program, &mut report);
    check_backend_rejection(backend, model, program, &mut report);
    check_precision_rejection(backend, model, program, caps, &mut report);
    check_kernel_rejection(backend, model, program, &mut report);
    check_dataflow_rejection(backend, model, program, &mut report);
    report
}

fn check_graph<B: BackendProvider>(
    backend: &B,
    model: &ModelIr,
    program: &ExecutionProgram,
    report: &mut Report,
) {
    match backend.execution_graph(model) {
        Ok(graph) if graph == program.dataflow => report.pass("program.execution_graph"),
        Ok(_) => report.fail(
            "program.execution_graph",
            "execution_graph did not return the validated dataflow",
        ),
        Err(error) => report.fail("program.execution_graph", error.to_string()),
    }
    let foreign = next_model_id(model.id);
    match backend.execution_graph(&foreign_model(model, foreign)) {
        Err(_) => report.pass("program.execution_graph_model"),
        Ok(_) => report.fail(
            "program.execution_graph_model",
            "execution_graph accepted a different model",
        ),
    }
}

fn check_model_rejection<B: BackendProvider>(
    backend: &B,
    model: &ModelIr,
    program: &ExecutionProgram,
    report: &mut Report,
) {
    let foreign = next_model_id(model.id);
    expect_reject(
        report,
        "program.rejects_model",
        &backend.validate_program(&foreign_model(model, foreign), program),
    );
}

fn check_backend_rejection<B: BackendProvider>(
    backend: &B,
    model: &ModelIr,
    program: &ExecutionProgram,
    report: &mut Report,
) {
    let mut wrong = program.clone();
    wrong.backend = other_backend(program.backend);
    expect_reject(
        report,
        "program.rejects_backend",
        &backend.validate_program(model, &wrong),
    );
}

fn check_precision_rejection<B: BackendProvider>(
    backend: &B,
    model: &ModelIr,
    program: &ExecutionProgram,
    caps: &DeviceCapabilities,
    report: &mut Report,
) {
    let Some(precision) = unsupported_precision(program, caps) else {
        report.skip(
            "program.rejects_precision",
            "capabilities declare every precision candidate",
        );
        return;
    };
    let mut wrong = program.clone();
    wrong.precision = precision;
    expect_reject(
        report,
        "program.rejects_precision",
        &backend.validate_program(model, &wrong),
    );
}

fn check_kernel_rejection<B: BackendProvider>(
    backend: &B,
    model: &ModelIr,
    program: &ExecutionProgram,
    report: &mut Report,
) {
    let Some(first) = program.operations.first() else {
        report.skip(
            "program.rejects_kernel",
            "program has no compiled operation",
        );
        return;
    };
    let unknown = KernelId::new(UNKNOWN_KERNEL_ID).unwrap_or(first.kernel);
    let mut wrong = program.clone();
    wrong.operations[0].kernel = unknown;
    expect_reject(
        report,
        "program.rejects_kernel",
        &backend.validate_program(model, &wrong),
    );
}

fn check_dataflow_rejection<B: BackendProvider>(
    backend: &B,
    model: &ModelIr,
    program: &ExecutionProgram,
    report: &mut Report,
) {
    let mut wrong = program.clone();
    wrong.dataflow.scratch_elements = if wrong.dataflow.scratch_elements == usize::MAX {
        0
    } else {
        wrong.dataflow.scratch_elements + 1
    };
    expect_reject(
        report,
        "program.rejects_dataflow",
        &backend.validate_program(model, &wrong),
    );
}

fn expect_reject(report: &mut Report, name: &'static str, result: &Result<()>) {
    match result {
        Err(_) => report.pass(name),
        Ok(()) => report.fail(name, "backend accepted a mismatched program"),
    }
}

fn foreign_model(model: &ModelIr, id: ModelId) -> ModelIr {
    let mut foreign = model.clone();
    foreign.id = id;
    foreign
}

fn next_model_id(id: ModelId) -> ModelId {
    id.get()
        .checked_add(1)
        .and_then(|next| ModelId::new(next).ok())
        .unwrap_or(ModelId::ONE)
}

const fn other_backend(kind: BackendKind) -> BackendKind {
    match kind {
        BackendKind::Cuda => BackendKind::Metal,
        BackendKind::Metal => BackendKind::Cuda,
    }
}

fn unsupported_precision(
    program: &ExecutionProgram,
    caps: &DeviceCapabilities,
) -> Option<PrecisionPlan> {
    [
        PrecisionPlan::f32(),
        PrecisionPlan::bf16(),
        PrecisionPlan::fp8_channel(),
        PrecisionPlan::nvfp4_block(),
    ]
    .into_iter()
    .find(|plan| *plan != program.precision && !caps.compute_dtypes.contains(&plan.compute))
}

#[cfg(test)]
#[path = "../tests/unit/program.rs"]
mod tests;
