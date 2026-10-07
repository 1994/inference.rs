//! A model provider owns topology; loading and runtime must preserve it verbatim.
use infer_backend_host::{HostBackend, HostConfig, HostKernels};
use infer_core::{ModelId, Result};
use infer_ir::{DataflowGraph, ModelIr, PrecisionPlan, TensorOp};
use infer_kernel_api::KernelRegistry;
use infer_models::{ModelPackage, ModelRegistry, QwenProvider};
use infer_runtime::{Engine, RuntimeConfig};
use infer_spi::{BackendProvider, ImportedModel, ModelProvider, ProviderMetadata};
use std::{path::Path, sync::Arc};

struct WithoutFinalNorm;

impl ModelProvider for WithoutFinalNorm {
    fn metadata(&self) -> ProviderMetadata {
        QwenProvider.metadata()
    }
    fn architectures(&self) -> &'static [&'static str] {
        QwenProvider.architectures()
    }
    fn import(&self, id: ModelId, config: &[u8]) -> Result<ImportedModel> {
        QwenProvider.import(id, config)
    }
    fn weight_source(&self, slot: &str, prefix: &str) -> String {
        QwenProvider.weight_source(slot, prefix)
    }
    fn graph(&self, model: &ModelIr) -> Result<DataflowGraph> {
        let mut graph = QwenProvider.graph(model)?;
        let final_norm = graph.nodes.len() - 2;
        let norm = graph.nodes.remove(final_norm);
        assert!(matches!(norm.op, TensorOp::Norm { .. }));
        let hidden = norm.inputs[0];
        graph.nodes[final_norm].inputs[0] = hidden;
        graph.hidden = Some(hidden);
        graph
            .tensors
            .retain(|tensor| !norm.outputs.contains(&tensor.id));
        graph.plan_lifetimes()?;
        Ok(graph)
    }
}

#[test]
fn provider_topology_survives_loading_compilation_and_runtime() -> Result<()> {
    let mut providers = ModelRegistry::new();
    providers.register(Arc::new(WithoutFinalNorm))?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../examples/qwen-hybrid-tiny");
    let mut package = ModelPackage::open_with(&providers, root, ModelId::ONE)?;
    let model = package.imported.model.clone();
    let graph = package.graph.clone();
    assert_ne!(graph, QwenProvider.graph(&model)?);
    package.validate_weight_bindings()?;
    let backend = HostBackend::from_package(&mut package, HostConfig::default())?;
    assert_eq!(backend.execution_graph(&model)?, graph);
    let mut kernels = KernelRegistry::default();
    kernels.register(&HostKernels)?;
    let engine = Engine::new(
        backend,
        model,
        PrecisionPlan::f32(),
        &kernels,
        RuntimeConfig::default(),
    )?;
    assert_eq!(engine.program().dataflow, graph);
    Ok(())
}
