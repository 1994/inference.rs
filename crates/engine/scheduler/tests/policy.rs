use infer_core::*;
use infer_ir::*;
use infer_scheduler::*;
use infer_spi::SchedulingPolicy;

#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn work(id: u64, role: ExecutionRole, tokens: usize) -> ReadyWork {
    let program = ProgramId::new(1).unwrap();
    let unit = CostEstimate {
        gpu_us: 1,
        ..Default::default()
    };
    ReadyWork {
        request: RequestId::new(id).unwrap(),
        state: StateId::new(id).unwrap(),
        program,
        role,
        remaining_tokens: tokens,
        tenant: format!("t{id}"),
        weight: 1,
        deadline_us: None,
        virtual_finish: 0,
        cost_per_token: unit,
        cost_query: CostQuery::from_unit(program, BackendKind::Metal, role, 1, 1, unit),
        latency_deadline_us: None,
        remaining_latency_us: tokens as u64,
        remaining_completion_us: tokens as u64,
        last_service_us: 0,
    }
}
fn resources() -> ResourceSnapshot {
    ResourceSnapshot {
        max_num_seqs: 8,
        max_num_batched_tokens: 16,
        gpu_budget_us: 100,
        workspace_bytes: 1024,
        free_state_pages: 4,
        free_logical_pages: 4,
        free_state_bytes: 4096,
        transfer_budget_us: 100,
        encoder_budget_us: 100,
        graphs: vec![],
        scheduler: SchedulerConfig {
            fair_quantum_tokens: 1,
            ..Default::default()
        },
    }
}
#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn plan(items: &[ReadyWork], r: &ResourceSnapshot, now: u64) -> SchedulingDecision {
    CostAwarePolicy
        .plan(
            items,
            r,
            now,
            DecisionId::new(1).unwrap(),
            StepId::new(2).unwrap(),
        )
        .unwrap()
}
#[test]
fn slack_uses_remaining_service_instead_of_only_absolute_deadline() {
    let mut short = work(1, ExecutionRole::Prefill, 1);
    short.deadline_us = Some(20);
    let mut long = work(2, ExecutionRole::Prefill, 30);
    long.deadline_us = Some(40);
    let mut r = resources();
    r.max_num_batched_tokens = 1;
    let result = plan(&[short, long], &r, 0);
    assert_eq!(result.step.unwrap().work[0].request.get(), 2);
    assert_eq!(result.selected[0].slack_us, Some(10));
    assert_eq!(result.selected[0].reason, SelectionReason::UrgentSlo);
}
#[test]
fn mixed_batch_protects_decode_and_caps_long_prefill() {
    let mut r = resources();
    r.scheduler.prefill_chunk_tokens = 4;
    let result = plan(
        &[
            work(1, ExecutionRole::Prefill, 100),
            work(2, ExecutionRole::Decode, 1),
            work(3, ExecutionRole::Forward, 20),
        ],
        &r,
        0,
    );
    let step = result.step.unwrap();
    assert_eq!(step.role, ExecutionRole::Mixed);
    assert_eq!(step.work[0].role, ExecutionRole::Decode);
    assert_eq!(
        step.work
            .iter()
            .find(|w| w.request.get() == 1)
            .unwrap()
            .token_count,
        4
    );
    assert_eq!(
        step.work
            .iter()
            .find(|w| w.request.get() == 2)
            .unwrap()
            .token_count,
        1
    );
    assert!(step.work.iter().any(|w| w.role == ExecutionRole::Forward));
    assert_eq!(step.cost.gpu_us, 16);
}
#[test]
fn wfq_updates_tenant_service_inside_a_batch() {
    let a = work(1, ExecutionRole::Prefill, 32);
    let mut b = work(2, ExecutionRole::Prefill, 32);
    b.weight = 3;
    let mut r = resources();
    r.max_num_batched_tokens = 12;
    let step = plan(&[a, b], &r, 0).step.unwrap();
    let count = |id| {
        step.work
            .iter()
            .find(|w| w.request.get() == id)
            .unwrap()
            .token_count
    };
    assert_eq!(count(1), 3);
    assert_eq!(count(2), 9);
}
#[test]
fn aging_prevents_starvation_even_under_urgent_decode_traffic() {
    let mut long = work(1, ExecutionRole::Forward, 100);
    long.virtual_finish = 1_000_000_000;
    let mut decode = work(2, ExecutionRole::Decode, 1);
    decode.last_service_us = 100;
    decode.latency_deadline_us = Some(101);
    let mut r = resources();
    r.max_num_batched_tokens = 1;
    r.scheduler.max_wait_us = 100;
    let decision = plan(&[long, decode], &r, 100);
    assert_eq!(decision.step.unwrap().work[0].request.get(), 1);
    assert_eq!(decision.selected[0].reason, SelectionReason::Aging);
}
#[test]
fn incompatible_or_resource_heavy_candidate_does_not_block_others() {
    for field in 0..4 {
        let mut bad = work(1, ExecutionRole::Prefill, 1);
        match field {
            0 => bad.cost_query.workspace_bytes = 1025,
            1 => bad.cost_query.num_gpu_blocks = 5,
            2 => bad.cost_query.transfer_us = 101,
            _ => bad.cost_query.encoder_us = 101,
        }
        let result = plan(&[bad, work(2, ExecutionRole::Prefill, 1)], &resources(), 0);
        assert_eq!(result.step.unwrap().work[0].request.get(), 2);
        assert_eq!(result.deferred.len(), 1);
        assert!(result.deferred[0].required > result.deferred[0].available);
    }
    let mut other = work(2, ExecutionRole::Prefill, 1);
    other.program = ProgramId::new(2).unwrap();
    other.cost_query.program = other.program;
    let result = plan(&[work(1, ExecutionRole::Decode, 1), other], &resources(), 0);
    assert_eq!(result.step.unwrap().work.len(), 1);
    assert_eq!(result.deferred[0].reason, DeferReason::IncompatibleProgram);
}
#[test]
fn multi_resource_cost_is_aggregated_for_the_complete_batch() {
    let mut a = work(1, ExecutionRole::Decode, 1);
    a.cost_query.state_bytes = 3000;
    a.cost_query.transfer_us = 60;
    let mut b = work(2, ExecutionRole::Decode, 1);
    b.cost_query.state_bytes = 3000;
    b.cost_query.transfer_us = 60;
    let decision = plan(&[a, b], &resources(), 0);
    assert_eq!(decision.deferred[0].reason, DeferReason::StateCapacity);
    assert_eq!(decision.deferred[0].required, 6000);
    assert_eq!(decision.deferred[0].available, 4096);
    let step = decision.step.unwrap();
    assert_eq!(step.work.len(), 1);
    assert_eq!(step.cost.state_bytes, 3000);
    assert_eq!(step.cost.transfer_us, 60);
}
#[test]
fn minimum_quantum_escape_is_singleton_and_never_relaxes_memory_limits() {
    let mut item = work(1, ExecutionRole::Prefill, 100);
    item.cost_query.fallback_per_token_us = 200;
    let step = plan(&[item.clone()], &resources(), 0).step.unwrap();
    assert!(step.quantum_overrun);
    assert_eq!(step.work[0].token_count, 1);
    assert_eq!(step.cost.gpu_us, 200);
    item.cost_query.state_bytes = 4097;
    assert!(plan(&[item], &resources(), 0).step.is_none());
}
#[test]
fn graph_descriptor_is_program_and_shape_compatible_and_decode_only() {
    let mut r = resources();
    r.graphs = vec![
        GraphVariant {
            batch_capacity: 2,
            max_tokens: 2,
            program: Some(ProgramId::new(2).unwrap()),
            workspace_bytes: 0,
        },
        GraphVariant {
            batch_capacity: 4,
            max_tokens: 4,
            program: Some(ProgramId::new(1).unwrap()),
            workspace_bytes: 500,
        },
    ];
    let step = plan(
        &[
            work(1, ExecutionRole::Decode, 1),
            work(2, ExecutionRole::Decode, 1),
        ],
        &r,
        0,
    )
    .step
    .unwrap();
    assert_eq!(step.graph.unwrap().batch_capacity, 4);
    assert_eq!(step.cost.workspace_bytes, 500);
    let step = plan(
        &[
            work(1, ExecutionRole::Decode, 1),
            work(2, ExecutionRole::Prefill, 2),
        ],
        &r,
        0,
    )
    .step
    .unwrap();
    assert!(step.graph.is_none());
}
#[test]
fn planning_is_input_order_invariant_and_checked_against_overflow() {
    let items = vec![
        work(1, ExecutionRole::Prefill, 32),
        work(2, ExecutionRole::Decode, 1),
        work(3, ExecutionRole::Forward, 32),
    ];
    let original = plan(&items, &resources(), 0);
    let mut reverse = items;
    reverse.reverse();
    assert_eq!(original, plan(&reverse, &resources(), 0));
    let mut item = work(1, ExecutionRole::Prefill, 4);
    item.cost_query.fallback_per_token_us = u64::MAX;
    let mut r = resources();
    r.scheduler.fair_quantum_tokens = 4;
    assert!(
        CostAwarePolicy
            .plan(
                &[item],
                &r,
                0,
                DecisionId::new(1).unwrap(),
                StepId::new(2).unwrap()
            )
            .is_err()
    );
}

#[test]
fn planning_budget_yields_partial_batch_with_explicit_cpu_deferrals() -> Result<()> {
    let ready = vec![
        work(1, ExecutionRole::Prefill, 256),
        work(2, ExecutionRole::Decode, 1),
    ];
    let mut r = resources();
    r.scheduler.max_planning_probes = 1;
    r.max_num_batched_tokens = 1024;
    r.gpu_budget_us = 10_000;
    let decision = CostAwarePolicy.plan(&ready, &r, 0, DecisionId::ONE, StepId::ONE)?;
    assert_eq!(decision.step.as_ref().map(|step| step.work.len()), Some(1));
    assert_eq!(decision.deferred.len(), 1);
    assert_eq!(decision.deferred[0].reason, DeferReason::PlanningBudget);
    validate_decision(&decision, &ready, &r, ProgramId::ONE, &FallbackCosts)
}

#[test]
fn exhausted_large_quantum_probe_still_dispatches_a_feasible_minimum() -> Result<()> {
    let ready = vec![work(1, ExecutionRole::Prefill, 100)];
    let mut r = resources();
    r.scheduler.max_planning_probes = 1;
    r.scheduler.prefill_chunk_tokens = 128;
    r.scheduler.fair_quantum_tokens = 128;
    r.max_num_batched_tokens = 128;
    r.gpu_budget_us = 2;
    let decision = CostAwarePolicy.plan(&ready, &r, 0, DecisionId::ONE, StepId::ONE)?;
    assert_eq!(
        decision
            .step
            .as_ref()
            .and_then(|step| step.work.first())
            .map(|work| work.token_count),
        Some(1)
    );
    validate_decision(&decision, &ready, &r, ProgramId::ONE, &FallbackCosts)
}
