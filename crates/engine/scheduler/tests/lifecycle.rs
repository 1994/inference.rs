use infer_core::{RequestId, Result, StepId};
use infer_ir::ExecutionRole;
use infer_scheduler::{BlockedOn, QueueRequest, QueueState, RequestQueue};

fn item(id: u64, phase: ExecutionRole) -> Result<QueueRequest> {
    Ok(QueueRequest {
        request: RequestId::new(id)?,
        phase,
        tenant: "t".into(),
        virtual_finish: id,
        last_service_us: 10,
        deadline_us: None,
        hard_deadline_us: Some(500),
        wait_deadline_us: 100,
    })
}
#[test]
fn cancellation_keeps_flight_owned_until_exact_completion_fence() -> Result<()> {
    let mut queue = RequestQueue::new(3)?;
    queue.enqueue(item(1, ExecutionRole::Prefill)?)?;
    queue.enqueue(item(2, ExecutionRole::Decode)?)?;
    queue.dispatch(StepId::ONE, &[RequestId::ONE])?;
    queue.cancel(RequestId::ONE)?;
    assert_eq!(queue.inspect().cancel_pending, 1);
    assert_eq!(queue.inspect().decode, 1);
    assert!(queue.remove(RequestId::ONE).is_err());
    assert!(queue.complete(RequestId::ONE, StepId::new(2)?).is_err());
    queue.check_invariants()?;
    assert!(queue.complete(RequestId::ONE, StepId::ONE)?);
    queue.remove(RequestId::ONE)?;
    assert!(queue.complete(RequestId::ONE, StepId::ONE).is_err());
    queue.check_invariants()
}
#[test]
fn dispatch_failure_is_atomic_and_memory_waiters_wake_by_epoch() -> Result<()> {
    let mut queue = RequestQueue::new(3)?;
    queue.enqueue(item(1, ExecutionRole::Prefill)?)?;
    queue.enqueue(item(2, ExecutionRole::Decode)?)?;
    queue.block(RequestId::ONE, BlockedOn::Memory, 7)?;
    assert!(
        queue
            .dispatch(StepId::ONE, &[RequestId::new(2)?, RequestId::ONE])
            .is_err()
    );
    assert_eq!(queue.state(RequestId::new(2)?), Some(QueueState::Ready));
    assert_eq!(queue.wake(BlockedOn::Transfer, 8), []);
    assert_eq!(queue.wake(BlockedOn::Memory, 7), []);
    assert_eq!(queue.wake(BlockedOn::Memory, 8), [RequestId::ONE]);
    queue.check_invariants()?;
    queue.dispatch(StepId::ONE, &[RequestId::ONE, RequestId::new(2)?])?;
    queue.check_invariants()
}
#[test]
fn queue_wait_deadlines_pause_during_execution_but_hard_deadlines_do_not() -> Result<()> {
    let mut queue = RequestQueue::new(2)?;
    queue.enqueue(item(1, ExecutionRole::Prefill)?)?;
    queue.enqueue(item(2, ExecutionRole::Decode)?)?;
    queue.dispatch(StepId::ONE, &[RequestId::ONE])?;
    assert_eq!(queue.expired_ids(100), [RequestId::new(2)?]);
    assert_eq!(queue.expired_ids(500), [RequestId::ONE, RequestId::new(2)?]);
    queue.cancel(RequestId::ONE)?;
    assert_eq!(queue.expired_ids(500), [RequestId::new(2)?]);
    queue.check_invariants()
}
#[test]
fn priority_indexes_are_bounded_and_phase_changes_do_not_leave_tombstones() -> Result<()> {
    let mut queue = RequestQueue::new(3)?;
    let mut aged = item(1, ExecutionRole::Prefill)?;
    aged.virtual_finish = 100;
    let mut urgent = item(2, ExecutionRole::Decode)?;
    urgent.last_service_us = 95;
    urgent.deadline_us = Some(110);
    queue.enqueue(aged.clone())?;
    queue.enqueue(urgent)?;
    queue.enqueue(item(3, ExecutionRole::Forward)?)?;
    let mut scratch = Vec::with_capacity(3);
    queue.candidates_into(&mut scratch, 2, 100, 50, 20)?;
    assert_eq!(scratch, [RequestId::ONE, RequestId::new(3)?]);
    queue.candidates_into(&mut scratch, 2, 20, 50, 100)?;
    assert_eq!(scratch[0], RequestId::new(2)?);
    aged.phase = ExecutionRole::Decode;
    queue.update_ready(&aged)?;
    assert_eq!(queue.inspect().prefill, 0);
    assert_eq!(queue.inspect().decode, 2);
    queue.update_tenant("t", 200);
    queue.check_invariants()
}

#[test]
fn resource_ack_wakes_only_its_owner_and_duplicate_ack_is_rejected() -> Result<()> {
    let mut queue = RequestQueue::new(2)?;
    queue.enqueue(item(1, ExecutionRole::Prefill)?)?;
    queue.enqueue(item(2, ExecutionRole::Decode)?)?;
    queue.block(RequestId::ONE, BlockedOn::Preparation, 7)?;
    queue.block(RequestId::new(2)?, BlockedOn::Preparation, 7)?;
    queue.unblock(RequestId::ONE)?;
    assert_eq!(queue.state(RequestId::ONE), Some(QueueState::Ready));
    assert!(matches!(
        queue.state(RequestId::new(2)?),
        Some(QueueState::Blocked {
            reason: BlockedOn::Preparation,
            ..
        })
    ));
    assert!(queue.unblock(RequestId::ONE).is_err());
    queue.check_invariants()?;
    queue.cancel(RequestId::new(2)?)?;
    assert!(queue.unblock(RequestId::new(2)?).is_err());
    queue.check_invariants()
}

#[test]
fn bounded_tenant_merge_matches_full_sort_across_head_changes_and_restore() -> Result<()> {
    use std::collections::BTreeMap;
    let mut queue = RequestQueue::new(128)?;
    let mut records = BTreeMap::new();
    let mut finishes = BTreeMap::new();
    for id in 1..=128 {
        let mut request = item(id, ExecutionRole::Decode)?;
        request.tenant = format!("tenant{}", id % 8).into();
        request.virtual_finish = id % 8;
        finishes
            .entry(request.tenant.clone())
            .or_insert(request.virtual_finish);
        queue.enqueue(request.clone())?;
        records.insert(request.request, request);
    }
    let mut scratch = Vec::with_capacity(17);
    for tick in 1..=256 {
        queue.candidates_into(&mut scratch, 17, 0, u64::MAX, 0)?;
        let mut expected: Vec<_> = records.values().collect();
        expected.sort_unstable_by_key(|r| (finishes[&r.tenant], r.last_service_us, r.request));
        assert_eq!(
            scratch,
            expected
                .iter()
                .take(17)
                .map(|r| r.request)
                .collect::<Vec<_>>()
        );
        let step = StepId::new(tick)?;
        queue.dispatch(step, &scratch[..7])?;
        for id in &scratch[..7] {
            let record = records
                .get_mut(id)
                .ok_or_else(|| infer_core::Error::invariant("reference owner missing"))?;
            record.last_service_us = tick;
            queue.complete_ready(
                *id,
                step,
                infer_scheduler::QueueTiming {
                    phase: record.phase,
                    last_service_us: tick,
                    deadline_us: None,
                    wait_deadline_us: record.wait_deadline_us,
                },
            )?;
            let finish = finishes
                .get_mut(&record.tenant)
                .ok_or_else(|| infer_core::Error::invariant("reference tenant missing"))?;
            *finish += 1;
            queue.update_tenant(&record.tenant, *finish);
        }
        if tick.is_multiple_of(32) {
            queue.check_invariants()?;
            let bytes = serde_json::to_vec(&queue)
                .map_err(|error| infer_core::Error::invalid(error.to_string()))?;
            queue = serde_json::from_slice(&bytes)
                .map_err(|error| infer_core::Error::invalid(error.to_string()))?;
            queue.check_invariants()?;
        }
    }
    queue.check_invariants()
}

#[test]
fn candidate_capacity_failure_preserves_lifecycle_and_retries_without_growth() -> Result<()> {
    let mut queue = RequestQueue::new(2)?;
    queue.enqueue(item(1, ExecutionRole::Decode)?)?;
    queue.enqueue(item(2, ExecutionRole::Prefill)?)?;
    let mut scratch = Vec::with_capacity(1);
    let pointer = scratch.as_ptr();
    assert_eq!(
        queue
            .candidates_into(&mut scratch, 2, 0, u64::MAX, 0)
            .err()
            .map(|error| error.code),
        Some(infer_core::ErrorCode::Capacity)
    );
    assert_eq!(scratch.as_ptr(), pointer);
    assert_eq!(scratch, []);
    queue.check_invariants()?;
    queue.candidates_into(&mut scratch, 1, 0, u64::MAX, 0)?;
    assert_eq!(scratch, [RequestId::ONE]);
    assert_eq!(scratch.as_ptr(), pointer);
    queue.check_invariants()
}
