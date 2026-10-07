//! CPU planner microbenchmark; no model execution or throughput claims.
use infer_core::{DecisionId, ProgramId, RequestId, Result, StateId, StepId};
use infer_ir::{
    BackendKind, CostEstimate, CostQuery, ExecutionRole, ReadyWork, ResourceSnapshot,
    SchedulerConfig,
};
use infer_scheduler::{CostAwarePolicy, FallbackCosts};
use infer_spi::{CostModelProvider, SchedulingPolicy};
use std::{cell::Cell, hint::black_box, time::Instant};

#[derive(Default)]
struct Counted {
    calls: Cell<usize>,
    queries: Cell<usize>,
}
impl CostModelProvider for Counted {
    fn identity(&self) -> &'static str {
        "counted-fallback"
    }
    fn estimate(&self, work: &[CostQuery]) -> Result<CostEstimate> {
        self.calls.set(self.calls.get() + 1);
        self.queries.set(self.queries.get() + work.len());
        FallbackCosts.estimate(work)
    }
}
fn ready(count: usize, mixed: bool) -> Result<Vec<ReadyWork>> {
    (1..=count)
        .map(|n| {
            let program = ProgramId::new(1)?;
            let role = if mixed && n % 4 == 0 {
                ExecutionRole::Decode
            } else {
                ExecutionRole::Prefill
            };
            let unit = CostEstimate {
                gpu_us: 1,
                ..Default::default()
            };
            Ok(ReadyWork {
                request: RequestId::new(n as u64)?,
                state: StateId::new(n as u64)?,
                program,
                role,
                remaining_tokens: if role == ExecutionRole::Decode {
                    1
                } else {
                    512
                },
                tenant: format!("t{}", n % 8),
                weight: 1,
                deadline_us: None,
                virtual_finish: 0,
                cost_per_token: unit,
                cost_query: CostQuery::from_unit(program, BackendKind::Metal, role, 1, 1, unit),
                latency_deadline_us: None,
                remaining_latency_us: 512,
                remaining_completion_us: 512,
                last_service_us: 0,
            })
        })
        .collect()
}
fn run(count: usize, mixed: bool, quantum: usize) -> Result<()> {
    let items = ready(count, mixed)?;
    let resources = ResourceSnapshot {
        max_num_seqs: 32,
        max_num_batched_tokens: 128,
        gpu_budget_us: 1000,
        workspace_bytes: 1 << 20,
        free_state_pages: 4096,
        free_logical_pages: 4096,
        free_state_bytes: 1 << 30,
        transfer_budget_us: 1000,
        encoder_budget_us: 1000,
        graphs: vec![],
        scheduler: SchedulerConfig {
            fair_quantum_tokens: quantum,
            ..Default::default()
        },
    };
    let costs = Counted::default();
    let mut workspace = <CostAwarePolicy as SchedulingPolicy>::Workspace::default();
    let started = Instant::now();
    let mut digest = 0_u64;
    for _ in 0..200 {
        let plan = CostAwarePolicy.plan_reusing(
            infer_spi::PlanningContext {
                ready: black_box(&items),
                resources: &resources,
                now_us: 0,
                decision: DecisionId::new(1)?,
                step: StepId::new(2)?,
                costs: &costs,
            },
            &mut workspace,
        )?;
        digest = plan.step.as_ref().map_or(0, |step| {
            step.work.iter().fold(0, |sum, work| {
                sum.wrapping_mul(31)
                    .wrapping_add(work.request.get() * 1000 + work.token_count as u64)
            })
        });
        black_box(plan);
    }
    println!(
        "{count},{mixed},{quantum},{},{},{},{digest}",
        started.elapsed().as_nanos() / 200,
        costs.calls.get() / 200,
        costs.queries.get() / 200
    );
    Ok(())
}
fn main() -> Result<()> {
    println!("ready,mixed,quantum,ns_per_plan,estimate_calls,query_visits,decision_digest");
    for count in [32, 256] {
        for mixed in [false, true] {
            for quantum in [1, 8] {
                run(count, mixed, quantum)?;
            }
        }
    }
    Ok(())
}
