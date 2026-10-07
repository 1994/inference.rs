mod backend;
use crate::allocator::{self, Counts};
use infer_backend_reference::{ReferenceKernels, ReferenceModel};
use infer_core::{Error, ModelId, RequestId, Result};
use infer_ir::{CanonicalRequest, PrecisionPlan};
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, RuntimeConfig};
use serde::Serialize;
#[derive(Serialize)]
pub struct Case {
    requests: usize,
    batch: usize,
    prefill: Counts,
    decode: Counts,
    completion: Counts,
    cancellation: Counts,
}
fn engine(requests: usize, batch: usize) -> Result<Engine<backend::Backend>> {
    let mut model = ReferenceModel::fixture(ModelId::ONE, 7).ir;
    model.max_sequence = 32768;
    let backend = backend::Backend::new(requests, batch, model.vocab_size)?;
    let mut registry = KernelRegistry::default();
    registry.register(&ReferenceKernels)?;
    Engine::new(
        backend,
        model,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig {
            max_requests: requests,
            max_num_seqs: batch,
            candidate_limit: requests,
            history_capacity: 32,
            admission: infer_ir::AdmissionConfig {
                default_tenant: infer_ir::TenantQuota {
                    max_reserved_tokens: requests * 32768,
                    max_state_pages: requests * 2048,
                    ..infer_ir::TenantQuota::default()
                },
                ..infer_ir::AdmissionConfig::default()
            },
            max_num_batched_tokens: batch * 64,
            num_gpu_blocks: requests * 2048,
            ..RuntimeConfig::default()
        },
    )
}
fn request(id: usize, tokens: usize, generated: usize) -> Result<CanonicalRequest> {
    serde_json::from_value(serde_json::json!({
        "id":id,"model":1,"input":{"Sequence":{"tokens":vec![1; tokens]}},
        "sampling":{"temperature":0.7,"top_k":8,"seed":42},
        "workload":{"Generate":{"max_new_tokens":generated}}
    }))
    .map_err(|error| Error::invalid(error.to_string()))
}
fn measured(operation: impl FnOnce() -> Result<()>) -> Result<Counts> {
    allocator::start();
    let result = operation();
    let counts = allocator::stop();
    result?;
    Ok(counts)
}
const fn add(total: &mut Counts, sample: Counts) {
    total.allocations += sample.allocations;
    total.reallocations += sample.reallocations;
    total.deallocations += sample.deallocations;
    total.requested_bytes += sample.requested_bytes;
}
pub fn run(requests: usize, batch: usize) -> Result<Case> {
    let mut engine = engine(requests, batch)?;
    for id in 1..=requests {
        engine.submit(request(id, 1024, 16384)?)?;
    }
    let mut output = Vec::with_capacity(batch * 2);
    let mut prefill = Counts::default();
    let mut decode = Counts::default();
    allocator::arm_trap();
    for tick in 1..=10000 {
        let generating = !engine.request(RequestId::ONE)?.generated.is_empty();
        let counts = measured(|| engine.tick_into(tick, &mut output))?;
        add(
            if generating {
                &mut decode
            } else {
                &mut prefill
            },
            counts,
        );
        output.clear();
    }
    if engine.request(RequestId::ONE)?.status.terminal() {
        return Err(Error::invariant("cancellation test lost its live request"));
    }
    let mut cancellation = Counts::default();
    for id in 1..=requests {
        add(
            &mut cancellation,
            measured(|| engine.cancel_into(RequestId::new(id as u64)?, &mut output))?,
        );
    }
    for tick in 10001..11000 {
        add(
            &mut cancellation,
            measured(|| engine.tick_into(tick, &mut output))?,
        );
        output.clear();
        if engine.is_idle() {
            break;
        }
    }
    engine.check_invariants()?;
    if !engine.is_idle() {
        return Err(Error::invariant("live cancellation did not drain"));
    }
    let mut terminal = self::engine(requests, batch)?;
    for id in 1..=requests {
        terminal.submit(request(id, 1, 1)?)?;
    }
    let mut completion = Counts::default();
    for tick in 1..1000 {
        add(
            &mut completion,
            measured(|| terminal.tick_into(tick, &mut output))?,
        );
        output.clear();
        if terminal.is_idle() {
            break;
        }
    }
    terminal.check_invariants()?;
    if !terminal.is_idle() {
        return Err(Error::invariant("completion did not drain"));
    }
    let case = Case {
        requests,
        batch,
        prefill,
        decode,
        completion,
        cancellation,
    };
    for (phase, counts) in [
        ("prefill", &case.prefill),
        ("decode", &case.decode),
        ("completion", &case.completion),
        ("cancellation", &case.cancellation),
    ] {
        if counts.allocations != 0 || counts.reallocations != 0 || counts.deallocations != 0 {
            return Err(Error::invariant(format!(
                "Engine CPU allocation gate failed in {phase} for R={requests}, B={batch}: {} alloc, {} realloc, {} dealloc\n{}",
                counts.allocations,
                counts.reallocations,
                counts.deallocations,
                allocator::trapped_stack().unwrap_or("no trapped stack")
            )));
        }
    }
    Ok(case)
}
