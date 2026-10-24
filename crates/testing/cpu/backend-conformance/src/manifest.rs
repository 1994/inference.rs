//! Kernel manifest checks.
use crate::report::Report;
use infer_core::{ErrorCode, KernelId, OpId};
use infer_ir::{
    BackendRequirements, CapabilityRequirements, DeviceBackend, DeviceCapabilities, Operation,
    OperationIr, PrecisionPlan,
};
use infer_kernel_api::KernelRegistry;
use infer_spi::{KernelProvider, KernelRegistration};
use std::collections::BTreeSet;

/// Every operation the IR can lower, in a stable order.
const OPERATIONS: [Operation; 20] = [
    Operation::TokenEmbedding,
    Operation::RmsNorm,
    Operation::Attention,
    Operation::LinearAttention,
    Operation::Convolution,
    Operation::MatMul,
    Operation::Silu,
    Operation::Residual,
    Operation::LmHead,
    Operation::Pool,
    Operation::Rank,
    Operation::Decision,
    Operation::Split,
    Operation::Rope,
    Operation::Sigmoid,
    Operation::Multiply,
    Operation::GatedNorm,
    Operation::MoeRoute,
    Operation::ExpertMatMul,
    Operation::MoeCombine,
];

/// Kernels that share one operation and precision; selection must return one of them.
struct Group {
    operation: Operation,
    precision: PrecisionPlan,
    ids: Vec<KernelId>,
    workspace_bytes: u64,
}

/// Check a backend's declared kernel table against its device capabilities.
///
/// The check is device-free: it validates descriptor metadata, registration, and that every
/// declared operation is selectable for the capability sample the adapter passes in.
#[must_use]
pub fn kernel_manifest(provider: &impl KernelProvider, caps: &DeviceCapabilities) -> Report {
    let kernels = provider.kernels();
    let mut report = Report::new();
    check_descriptors(&kernels, caps, &mut report);
    check_registration(provider, &kernels, caps, &mut report);
    report
}

fn check_descriptors(
    kernels: &[KernelRegistration],
    caps: &DeviceCapabilities,
    report: &mut Report,
) {
    if kernels.is_empty() {
        report.fail("manifest.non_empty", "provider registered no kernels");
        return;
    }
    report.pass("manifest.non_empty");
    check_identifier(kernels, report);
    let mut issues = Vec::new();
    for kernel in kernels {
        if kernel.backend != caps.backend_kind() {
            issues.push(format!(
                "kernel {} declares {:?}, capabilities are {:?}",
                kernel.id,
                kernel.backend,
                caps.backend_kind()
            ));
        }
        if !requirement_matches(&kernel.requirements, &caps.backend) {
            issues.push(format!(
                "kernel {} requires a different backend than the capability sample",
                kernel.id
            ));
        }
        if !caps.compute_dtypes.contains(&kernel.precision.compute) {
            issues.push(format!(
                "kernel {} computes {:?}, capabilities list {:?}",
                kernel.id, kernel.precision.compute, caps.compute_dtypes
            ));
        }
        if let Some(dtype) = kernel
            .requirements
            .compute_dtypes
            .iter()
            .find(|dtype| !caps.compute_dtypes.contains(dtype))
        {
            issues.push(format!(
                "kernel {} requires {dtype:?}, capabilities list {:?}",
                kernel.id, caps.compute_dtypes
            ));
        }
        if kernel.max_shape_elements == 0
            || kernel.estimated_ns == 0
            || kernel.source.file.is_empty()
            || kernel.source.function.is_empty()
        {
            issues.push(format!(
                "kernel {} has an empty shape/cost/source descriptor",
                kernel.id
            ));
        }
    }
    if issues.is_empty() {
        report.pass("manifest.descriptors");
    } else {
        report.fail("manifest.descriptors", issues.join("; "));
    }
}

fn check_identifier(kernels: &[KernelRegistration], report: &mut Report) {
    let mut ids = BTreeSet::new();
    let duplicate = kernels.iter().find(|kernel| !ids.insert(kernel.id));
    match duplicate {
        Some(kernel) => report.fail(
            "manifest.identifiers",
            format!("duplicate kernel id {}", kernel.id),
        ),
        None => report.pass("manifest.identifiers"),
    }
}

const fn requirement_matches(
    requirements: &CapabilityRequirements,
    backend: &DeviceBackend,
) -> bool {
    matches!(
        (&requirements.backend, backend),
        (None, _)
            | (Some(BackendRequirements::Cuda(_)), DeviceBackend::Cuda(_))
            | (Some(BackendRequirements::Metal(_)), DeviceBackend::Metal(_))
    )
}

fn check_registration(
    provider: &impl KernelProvider,
    kernels: &[KernelRegistration],
    caps: &DeviceCapabilities,
    report: &mut Report,
) {
    let mut registry = KernelRegistry::default();
    if let Err(error) = registry.register(provider) {
        report.fail("manifest.registration", error.to_string());
        return;
    }
    report.pass("manifest.registration");
    let groups = groups(kernels);
    let issues = group_issues(&registry, &groups, caps);
    if issues.is_empty() {
        report.pass("manifest.selectable");
    } else {
        report.fail("manifest.selectable", issues.join("; "));
    }
    check_unregistered(&registry, &groups, caps, report);
}

fn groups(kernels: &[KernelRegistration]) -> Vec<Group> {
    let mut groups: Vec<Group> = Vec::new();
    for kernel in kernels {
        let existing = groups.iter_mut().find(|group| {
            group.operation == kernel.operation && group.precision == kernel.precision
        });
        if let Some(group) = existing {
            group.ids.push(kernel.id);
            group.workspace_bytes = group.workspace_bytes.max(kernel.workspace_bytes);
        } else {
            groups.push(Group {
                operation: kernel.operation,
                precision: kernel.precision.clone(),
                ids: vec![kernel.id],
                workspace_bytes: kernel.workspace_bytes,
            });
        }
    }
    groups
}

fn group_issues(
    registry: &KernelRegistry,
    groups: &[Group],
    caps: &DeviceCapabilities,
) -> Vec<String> {
    let mut issues = Vec::new();
    for group in groups {
        let operation = operation_ir(group.operation, &group.precision);
        match registry.select(&operation, &group.precision, caps, group.workspace_bytes) {
            Ok(selected) if group.ids.contains(&selected.id) => {}
            Ok(selected) => issues.push(format!(
                "{:?} selected unregistered kernel {}",
                group.operation, selected.id
            )),
            Err(error) => issues.push(format!(
                "{:?} is registered but unselectable: {error}",
                group.operation
            )),
        }
    }
    issues
}

fn check_unregistered(
    registry: &KernelRegistry,
    groups: &[Group],
    caps: &DeviceCapabilities,
    report: &mut Report,
) {
    let registered: BTreeSet<_> = groups.iter().map(|group| group.operation).collect();
    let Some(missing) = OPERATIONS
        .iter()
        .copied()
        .find(|op| !registered.contains(op))
    else {
        report.skip(
            "manifest.unregistered",
            "the provider registers every known operation",
        );
        return;
    };
    let precision = PrecisionPlan::f32();
    let operation = operation_ir(missing, &precision);
    match registry.select(&operation, &precision, caps, 0) {
        Err(error) if error.code == ErrorCode::Unsupported => report.pass("manifest.unregistered"),
        Err(error) => report.fail(
            "manifest.unregistered",
            format!("unregistered {missing:?} returned {error}"),
        ),
        Ok(selected) => report.fail(
            "manifest.unregistered",
            format!("unregistered {missing:?} selected {}", selected.id),
        ),
    }
}

fn operation_ir(operation: Operation, precision: &PrecisionPlan) -> OperationIr {
    OperationIr {
        id: OpId::ONE,
        operation,
        shape: vec![1],
        layer: None,
        requirements: CapabilityRequirements {
            compute_dtypes: vec![precision.compute],
            backend: None,
        },
    }
}

#[cfg(test)]
#[path = "../tests/unit/manifest.rs"]
mod tests;
