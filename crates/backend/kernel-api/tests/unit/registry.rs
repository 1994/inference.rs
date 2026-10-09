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
