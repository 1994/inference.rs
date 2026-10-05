use infer_backend_host::{HostBackend, HostConfig, HostKernels};
use infer_kernel_api::KernelRegistry;
use infer_models::QwenPackage;
use infer_runtime::{Engine, RuntimeConfig};
use infer_workloads::ProjectionWorkloads;
use std::path::PathBuf;

use infer_core::*;
use infer_ir::*;
#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
)]
fn actual_projection_weights_match_torch_embedding_rank_and_typed_decisions() {
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../examples/qwen-hybrid-tiny");
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("readout-golden.json")).unwrap()).unwrap();
    let requests: Vec<CanonicalRequest> =
        serde_json::from_slice(include_bytes!("../../../../../examples/requests.json")).unwrap();
    for chunk in [1, 64] {
        let mut package = QwenPackage::open(&root, ModelId::new(1).unwrap()).unwrap();
        let model = package.imported.model.clone();
        let backend = HostBackend::from_package(&mut package, HostConfig::default()).unwrap();
        let provider =
            ProjectionWorkloads::open(&root, model.hidden_size, model.vocab_size).unwrap();
        let mut registry = KernelRegistry::default();
        registry.register(&HostKernels).unwrap();
        let mut engine = Engine::new(
            backend,
            model,
            PrecisionPlan::f32(),
            &registry,
            RuntimeConfig {
                token_budget: chunk,
                ..Default::default()
            },
        )
        .unwrap()
        .with_workloads(provider)
        .unwrap();
        for r in &requests {
            engine.submit(r.clone()).unwrap();
        }
        for now in 0..500 {
            if engine.is_idle() {
                break;
            }
            engine.tick(now).unwrap();
        }
        let get = |id| {
            engine
                .request(RequestId::new(id).unwrap())
                .unwrap()
                .completed
                .as_ref()
                .unwrap()
                .output
                .clone()
                .unwrap()
        };
        let WorkloadOutput::Embedding(embedding) = get(2) else {
            panic!("expected embedding")
        };
        let expected: Vec<f32> = serde_json::from_value(golden["embedding"].clone()).unwrap();
        assert!(
            infer_quality::compare(&expected, &embedding, 2e-6, 2e-5)
                .unwrap()
                .passed
        );
        let WorkloadOutput::Ranking(ranking) = get(3) else {
            panic!("expected rank")
        };
        let scores: Vec<f32> = serde_json::from_value(golden["rank_scores"].clone()).unwrap();
        let mut order: Vec<_> = (0..scores.len()).collect();
        order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b)));
        for (got, index) in ranking.iter().zip(order) {
            assert_eq!(got.index, index);
            assert!((got.score - scores[index]).abs() < 2e-6);
        }
        let WorkloadOutput::Decisions(answers) = get(4) else {
            panic!("expected decision")
        };
        let probabilities: Vec<Vec<f32>> =
            serde_json::from_value(golden["decision_probabilities"].clone()).unwrap();
        for (got, expected) in answers.iter().zip(probabilities) {
            if expected.is_empty() {
                assert_eq!(got.probabilities, [] as [f32; 0]);
                assert!(
                    (got.expected.unwrap()
                        - golden["continuous_expected"].as_f64().unwrap() as f32)
                        .abs()
                        < 2e-6
                );
                continue;
            }
            assert!(
                infer_quality::compare(&expected, &got.probabilities, 2e-6, 2e-5)
                    .unwrap()
                    .passed
            );
            assert!(got.abstained);
        }
        assert_eq!(engine.backend().inspect().sequences, 0);
        assert_eq!(engine.inspect().state.allocated_pages, 0);
    }
}
