use infer_backend_host::{HostBackend, HostConfig, HostKernels};
use infer_kernel_api::KernelRegistry;
use infer_models::ModelPackage;
use infer_runtime::{Engine, RuntimeConfig};
use infer_spi::BackendProvider;
use serde::Deserialize;
use std::path::Path;

use infer_core::*;
use infer_ir::*;

#[derive(Deserialize)]
struct Golden {
    prefixes: Vec<Prefix>,
    greedy_tokens: Vec<u32>,
}
#[derive(Deserialize)]
struct Prefix {
    tokens: Vec<u32>,
    logits: Vec<f32>,
    hidden: Vec<Vec<f32>>,
    layers: Vec<Vec<Vec<f32>>>,
}
fn load() -> (HostBackend, KernelRegistry, Golden) {
    load_package("qwen-hybrid-tiny")
}
#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn load_package(name: &str) -> (HostBackend, KernelRegistry, Golden) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../examples")
        .join(name);
    let mut package = ModelPackage::open(&root, ModelId::new(1).unwrap()).unwrap();
    let backend = HostBackend::from_package(&mut package, HostConfig::default()).unwrap();
    let mut registry = KernelRegistry::default();
    registry.register(&HostKernels).unwrap();
    let golden = serde_json::from_slice(&std::fs::read(root.join("golden.json")).unwrap()).unwrap();
    (backend, registry, golden)
}
#[test]
fn official_transformers_prefix_logits_and_hidden_match_incremental_dataflow() {
    check_prefixes("qwen-hybrid-tiny");
}
#[test]
fn grouped_multi_head_mapping_matches_official_layer_logits_and_hidden() {
    check_prefixes("qwen-hybrid-grouped");
}
#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn check_prefixes(package: &str) {
    let (mut backend, registry, golden) = load_package(package);
    backend.enable_layer_probes(1024 * 1024).unwrap();
    let model = backend.model().clone();
    let program = infer_compiler::compile(
        ProgramId::new(1).unwrap(),
        infer_compiler::lower(
            &model,
            infer_model_recipes::decoder::lower(&model).unwrap(),
            PrecisionPlan::f32(),
        )
        .unwrap(),
        &registry,
        &backend.capabilities(),
        1 << 20,
    )
    .unwrap();
    let state = StateId::new(1).unwrap();
    backend.reserve_state(state, 32).unwrap();
    for prefix in &golden.prefixes {
        let request = RequestId::new(1).unwrap();
        let step = StepPlan {
            id: StepId::new(prefix.tokens.len() as u64).unwrap(),
            decision: DecisionId::new(1).unwrap(),
            program: program.id,
            role: ExecutionRole::Prefill,
            work: vec![PlannedWork {
                request,
                state,
                token_count: 1,
                role: ExecutionRole::Prefill,
            }],
            cost: CostEstimate::default(),
            graph: None,
            quantum_overrun: false,
        };
        let mut ticket = backend
            .submit(
                &program,
                &step,
                vec![ExecutionTask {
                    request,
                    state,
                    tokens: prefix.tokens.clone().into(),

                    sampling: None,
                }],
            )
            .unwrap();
        let output = backend.poll(&mut ticket).unwrap().unwrap().remove(0).output;
        let probes = backend.drain_layer_probes();
        assert_eq!(probes.len(), prefix.layers.len());
        for probe in probes {
            assert_eq!(probe.position, prefix.tokens.len() - 1);
            let metric = infer_quality::compare(
                &prefix.layers[probe.layer][probe.position],
                &probe.hidden,
                2e-6,
                2e-5,
            )
            .unwrap();
            assert!(
                metric.passed,
                "layer {} position {} {:?}",
                probe.layer, probe.position, metric
            );
        }
        let logits = infer_quality::compare(&prefix.logits, &output.logits, 2e-6, 2e-5).unwrap();
        assert!(
            logits.passed,
            "prefix {} logits {:?}",
            prefix.tokens.len(),
            logits
        );
        let a: Vec<_> = prefix.hidden.iter().flatten().copied().collect();
        let b: Vec<_> = output.hidden.iter().flatten().copied().collect();
        let hidden = infer_quality::compare(&a, &b, 2e-6, 2e-5).unwrap();
        assert!(
            hidden.passed,
            "prefix {} hidden {:?}",
            prefix.tokens.len(),
            hidden
        );
    }
    assert_eq!(
        backend.inspect().tokens_executed,
        golden.prefixes.last().unwrap().tokens.len() as u64
    );
    assert!(backend.traces().iter().any(|t| !t.states.is_empty()));
    backend.release_state(state).unwrap();
    assert_eq!(backend.inspect().allocated_bytes, 0);
}
#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn request(tokens: Vec<u32>, id: u64) -> CanonicalRequest {
    CanonicalRequest {
        id: RequestId::new(id).unwrap(),
        model: ModelId::new(1).unwrap(),
        session: None,
        input: RequestInput::Sequence {
            tokens: tokens.into(),
            media: vec![],
        },
        workload: Workload::Generate { max_new_tokens: 5 },
        qos: Qos::default(),
        sampling: Sampling::default(),
        extensions: std::collections::BTreeMap::new(),
    }
}
#[test]
fn generation_matches_official_cached_greedy_and_chunk_invariance() {
    for chunk in [1, 3, 64] {
        let (backend, registry, golden) = load();
        let model = backend.model().clone();
        let mut engine = Engine::new(
            backend,
            model,
            PrecisionPlan::f32(),
            &registry,
            RuntimeConfig {
                max_num_batched_tokens: chunk,
                ..Default::default()
            },
        )
        .unwrap();
        let tokens = golden.prefixes.last().unwrap().tokens.clone();
        engine.submit(request(tokens.clone(), 1)).unwrap();
        for now in 0..100 {
            if engine.is_idle() {
                break;
            }
            engine.tick(now).unwrap();
        }
        let done = engine
            .request(RequestId::new(1).unwrap())
            .unwrap()
            .completed
            .as_ref()
            .unwrap();
        assert_eq!(
            done.output,
            Some(WorkloadOutput::Tokens(golden.greedy_tokens.into()))
        );
        assert_eq!(
            engine.backend().inspect().tokens_executed,
            (tokens.len() + 4) as u64
        );
        assert_eq!(engine.backend().inspect().sequences, 0);
        assert_eq!(engine.inspect().state.allocated_pages, 0);
        assert!(engine.inspect().cost_model.observations > 0);
        assert_eq!(
            engine.inspect().cost_model.last_source,
            Some(TimingSource::CpuWall)
        );
    }
}
#[test]
fn hybrid_execution_checkpoint_restores_mid_generation() {
    let (backend, registry, golden) = load();
    let model = backend.model().clone();
    let mut original = Engine::new(
        backend,
        model,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig::default(),
    )
    .unwrap();
    original
        .submit(request(golden.prefixes.last().unwrap().tokens.clone(), 1))
        .unwrap();
    original.tick(0).unwrap();
    let (_, snapshot) = original.quiesce(1).unwrap();
    let (backend, _, _) = load();
    let mut restored = Engine::restore(backend, &registry, snapshot.unwrap()).unwrap();
    for now in 2..100 {
        if !original.is_idle() {
            original.tick(now).unwrap();
        }
        if !restored.is_idle() {
            restored.tick(now).unwrap();
        }
    }
    assert_eq!(
        original
            .request(RequestId::new(1).unwrap())
            .unwrap()
            .completed,
        restored
            .request(RequestId::new(1).unwrap())
            .unwrap()
            .completed
    );
    assert_eq!(restored.backend().inspect().allocated_bytes, 0);
}
#[test]
fn prefix_reuse_restores_hybrid_state_and_does_not_recompute_cached_tokens() {
    let (backend, registry, golden) = load();
    let mut package = ModelPackage::open(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../examples/qwen-hybrid-tiny"),
        ModelId::new(1).unwrap(),
    )
    .unwrap();
    let model = backend.model().clone();
    let weights = package.load_host_weights(1024 * 1024).unwrap();
    let backend = HostBackend::new(
        model.clone(),
        weights,
        HostConfig {
            block_size: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let mut engine = Engine::new(
        backend,
        model,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig::default(),
    )
    .unwrap();
    for id in 1..=2 {
        engine
            .submit(request(golden.prefixes.last().unwrap().tokens.clone(), id))
            .unwrap();
        for now in (id - 1) * 100..id * 100 {
            if engine.is_idle() {
                break;
            }
            engine.tick(now).unwrap();
        }
        assert_eq!(
            engine
                .request(RequestId::new(id).unwrap())
                .unwrap()
                .completed
                .as_ref()
                .unwrap()
                .output,
            Some(WorkloadOutput::Tokens(golden.greedy_tokens.clone().into()))
        );
    }
    assert_eq!(engine.backend().inspect().prefix_hits, 1);
    // APC attaches two full blocks and leaves the last prompt token for logits.
    assert_eq!(engine.backend().inspect().tokens_executed, 16);
    assert!(engine.backend().inspect().prefix_entries > 0);
    assert_eq!(engine.backend().inspect().sequences, 0);
}
#[test]
fn physical_admission_failure_is_atomic_and_cancel_releases_hybrid_tensors() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../examples/qwen-hybrid-tiny");
    let mut package = ModelPackage::open(&root, ModelId::new(1).unwrap()).unwrap();
    let memory = package.manifest.host_f32_bytes + package.graph.scratch_elements as u64 * 4;
    let model = package.imported.model.clone();
    let weights = package.load_host_weights(1024 * 1024).unwrap();
    let backend = HostBackend::new(
        model.clone(),
        weights,
        HostConfig {
            memory_bytes: memory,
            prefix_cache_bytes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let mut registry = KernelRegistry::default();
    registry.register(&HostKernels).unwrap();
    let mut engine = Engine::new(
        backend,
        model,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig::default(),
    )
    .unwrap();
    assert_eq!(
        engine.submit(request(vec![1, 2], 1)).unwrap_err().code,
        ErrorCode::Capacity
    );
    assert_eq!(engine.inspect().active_requests, 0);
    assert_eq!(engine.inspect().state.allocated_pages, 0);
    assert_eq!(engine.backend().inspect().sequences, 0);
    let (backend, registry, _) = load();
    let model = backend.model().clone();
    let mut engine = Engine::new(
        backend,
        model,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig::default(),
    )
    .unwrap();
    engine.submit(request(vec![1, 2], 1)).unwrap();
    engine.tick(0).unwrap();
    engine.cancel(RequestId::new(1).unwrap()).unwrap();
    engine.tick(1).unwrap();
    assert_eq!(engine.backend().inspect().allocated_bytes, 0);
    assert_eq!(engine.inspect().state.allocated_pages, 0);
}
