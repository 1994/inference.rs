//! Checkpoints contain ownership and timing, never collection implementation details.
use super::{Entry, QueueState, ReadyKey, RequestQueue, Tenant};
use infer_core::{Error, RequestId, Result, map::BoundedMap};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::Arc;

#[derive(Deserialize)]
struct Checkpoint {
    capacity: usize,
    entries: BoundedMap<RequestId, Entry>,
    tenants: BoundedMap<Arc<str>, Tenant>,
    names: Vec<Option<Arc<str>>>,
    free_tenants: Vec<usize>,
    free_entries: Vec<usize>,
    scan_after: Option<RequestId>,
}
#[derive(Serialize)]
struct CheckpointRef<'a> {
    capacity: usize,
    entries: &'a BoundedMap<RequestId, Entry>,
    tenants: &'a BoundedMap<Arc<str>, Tenant>,
    names: &'a [Option<Arc<str>>],
    free_tenants: &'a [usize],
    free_entries: &'a [usize],
    scan_after: Option<RequestId>,
}
impl Serialize for RequestQueue {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.scratch
            .validate(self.capacity)
            .map_err(serde::ser::Error::custom)?;
        CheckpointRef {
            capacity: self.capacity,
            entries: &self.entries,
            tenants: &self.tenants,
            names: &self.names,
            free_tenants: &self.free_tenants,
            free_entries: &self.free_entries,
            scan_after: self.scan_after,
        }
        .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for RequestQueue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Self::restore(Checkpoint::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}
impl RequestQueue {
    fn restore(saved: Checkpoint) -> Result<Self> {
        let mut queue = Self::new(saved.capacity)?;
        queue.tenants = saved.tenants;
        queue.names = saved.names;
        queue.free_tenants = saved.free_tenants;
        queue.free_entries = saved.free_entries;
        queue.scan_after = saved.scan_after;
        if queue.names.len() != saved.capacity {
            return Err(Error::invalid("checkpoint tenant slots mismatch"));
        }
        for tenant in queue.tenants.values() {
            if queue
                .names
                .get(tenant.slot)
                .and_then(Option::as_ref)
                .is_none()
            {
                return Err(Error::invalid("checkpoint tenant slot out of bounds"));
            }
            queue.tenant_service.insert((tenant.finish, tenant.slot));
        }
        for (id, entry) in saved.entries.iter() {
            if *id != entry.request.request
                || entry.slot >= saved.capacity
                || entry.request.phase == infer_ir::ExecutionRole::Mixed
                || queue.names.get(entry.tenant).and_then(Option::as_ref)
                    != Some(&entry.request.tenant)
            {
                return Err(Error::invalid("checkpoint queue owner mismatch"));
            }
            if !matches!(entry.state, QueueState::CancelPending { .. })
                && let Some(deadline) = entry.request.hard_deadline_us
            {
                queue.hard_expiry.insert((deadline, *id));
            }
            match entry.state {
                QueueState::Ready => queue.index(ReadyKey::from(entry), true),
                QueueState::Blocked { reason, epoch } => {
                    queue.waiters.insert((reason, epoch, *id));
                }
                QueueState::Running { step } => {
                    queue.flights.insert((step, *id));
                    queue.running += 1;
                }
                QueueState::CancelPending { step } => {
                    queue.flights.insert((step, *id));
                    queue.cancelling += 1;
                }
            }
            if matches!(entry.state, QueueState::Ready | QueueState::Blocked { .. }) {
                queue
                    .wait_expiry
                    .insert((entry.request.wait_deadline_us, *id));
            }
        }
        queue.entries = saved.entries;
        queue.check_invariants()?;
        Ok(queue)
    }
}
