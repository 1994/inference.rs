use infer_core::{Error, ErrorCode, KernelId, Result};
use infer_ir::{DeviceCapabilities, OperationIr, PrecisionPlan};
use infer_spi::{KernelProvider, KernelRegistration};
use std::collections::BTreeMap;

#[derive(Default)]
pub struct KernelRegistry {
    kernels: BTreeMap<KernelId, KernelRegistration>,
}
impl KernelRegistry {
    /// Registration is transactional: an invalid provider cannot partially install.
    ///
    /// # Errors
    /// Returns an invalid-input or conflict error for incompatible provider metadata or duplicate registrations.
    pub fn register(&mut self, provider: &impl KernelProvider) -> Result<()> {
        let candidates = provider.kernels();
        let mut seen = std::collections::BTreeSet::new();
        for kernel in &candidates {
            if self.kernels.contains_key(&kernel.id) || !seen.insert(kernel.id) {
                return Err(Error::new(ErrorCode::Conflict, "duplicate kernel ID"));
            }
            if kernel.max_shape_elements == 0
                || kernel.estimated_ns == 0
                || kernel.source.file.is_empty()
            {
                return Err(Error::invalid("invalid kernel descriptor"));
            }
        }
        for candidate in candidates {
            self.kernels.insert(candidate.id, candidate);
        }
        Ok(())
    }
    #[must_use]
    pub fn get(&self, id: KernelId) -> Option<&KernelRegistration> {
        self.kernels.get(&id)
    }
    ///
    /// # Errors
    /// Returns an unsupported error if no registered kernel satisfies the operation, precision, backend, and workspace requirements.
    pub fn select(
        &self,
        operation: &OperationIr,
        precision: &PrecisionPlan,
        caps: &DeviceCapabilities,
        workspace: u64,
    ) -> Result<&KernelRegistration> {
        let elements = operation
            .shape
            .iter()
            .try_fold(1usize, |v, d| v.checked_mul(*d))
            .ok_or_else(|| Error::invalid("op shape overflow"))?;
        if elements == 0 {
            return Err(Error::invalid("empty op shape"));
        }
        self.kernels
            .values()
            .filter(|k| {
                k.operation == operation.operation
                    && k.backend == caps.backend_kind()
                    && k.precision == *precision
                    && caps.require(&k.requirements).is_ok()
                    && caps.require(&operation.requirements).is_ok()
                    && elements <= k.max_shape_elements
                    && k.workspace_bytes <= workspace
            })
            .min_by_key(|k| (std::cmp::Reverse(k.priority), k.estimated_ns, k.id))
            .ok_or_else(|| {
                Error::unsupported(format!(
                    "no compatible kernel for {:?} with {:?}",
                    operation.operation, precision.compute
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infer_core::OpId;
    use infer_ir::{
        BackendKind, BackendRequirements, CapabilityRequirements, CudaRequirements, DeviceBackend,
        MetalCapabilities, NvidiaArchitecture, NvidiaCapabilities, Operation,
    };
    use infer_spi::SourceLocation;
    struct Provider;
    impl KernelProvider for Provider {
        fn kernels(&self) -> Vec<KernelRegistration> {
            (1..=2)
                .map(|id| KernelRegistration {
                    backend: BackendKind::TestCpu,
                    id: KernelId::new(id).unwrap(),
                    operation: Operation::MatMul,
                    precision: PrecisionPlan::f32(),
                    requirements: CapabilityRequirements {
                        backend: (id == 1).then_some(BackendRequirements::Cuda(CudaRequirements {
                            tma: true,
                            ..Default::default()
                        })),
                        ..Default::default()
                    },
                    max_shape_elements: 100,
                    workspace_bytes: 0,
                    priority: 1,
                    estimated_ns: id,
                    source: SourceLocation {
                        crate_name: "test".into(),
                        file: "test.rs".into(),
                        function: "matmul".into(),
                    },
                })
                .collect()
        }
    }
    #[test]
    fn filters_capability_and_shape() {
        let mut registry = KernelRegistry::default();
        registry.register(&Provider).unwrap();
        let mut op = OperationIr {
            id: OpId::new(1).unwrap(),
            operation: Operation::MatMul,
            shape: vec![10],
            layer: None,
            requirements: CapabilityRequirements::default(),
        };
        assert_eq!(
            registry
                .select(
                    &op,
                    &PrecisionPlan::f32(),
                    &DeviceCapabilities::reference(),
                    0
                )
                .unwrap()
                .id
                .get(),
            2
        );
        op.shape = vec![101];
        assert!(
            registry
                .select(
                    &op,
                    &PrecisionPlan::f32(),
                    &DeviceCapabilities::reference(),
                    0
                )
                .is_err()
        );
        assert!(registry.register(&Provider).is_err());
    }
    #[test]
    fn backend_target_prevents_cpu_kernels_from_binding_to_metal_or_cuda() {
        let mut registry = KernelRegistry::default();
        registry.register(&Provider).unwrap();
        let op = OperationIr {
            id: OpId::new(1).unwrap(),
            operation: Operation::MatMul,
            shape: vec![10],
            layer: None,
            requirements: CapabilityRequirements::default(),
        };
        for backend in [
            DeviceBackend::Metal(MetalCapabilities { simd_width: 32 }),
            DeviceBackend::Cuda(NvidiaCapabilities {
                architecture: NvidiaArchitecture {
                    compute_major: 12,
                    compute_minor: 0,
                    multiprocessors: 170,
                },
                tensor_core_generation: Some(5),
                warp_size: 32,
                graphs: true,
                tma: true,
                clusters: true,
                pinned_transfer: true,
                cuda_ipc: true,
                nvlink: false,
                gpu_direct: false,
            }),
        ] {
            let mut caps = DeviceCapabilities::reference();
            caps.backend = backend;
            assert!(
                registry
                    .select(&op, &PrecisionPlan::f32(), &caps, 0)
                    .is_err()
            );
        }
    }
}
