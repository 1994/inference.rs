//! One lifecycle table with fixed priority indexes and hierarchical tenant service.
mod checkpoint;
mod lifecycle;
mod ordered;
mod planning;
mod scratch;
use infer_core::{Error, ErrorCode, RequestId, Result, StepId, map::BoundedMap};
use infer_ir::ExecutionRole;
use ordered::OrderedSet;
use scratch::CandidateScratch;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU64;

/// Number of dispatchable execution phases (prefill, decode, forward) counted by the queue.
const PHASE_COUNT: usize = 3;
const fn seed_id(slot: usize) -> RequestId {
    RequestId::from_nonzero(NonZeroU64::MIN.saturating_add(slot as u64))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum BlockedOn {
    Memory,
    Transfer,
    Preparation,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueueState {
    Ready,
    Blocked { reason: BlockedOn, epoch: u64 },
    Running { step: StepId },
    CancelPending { step: StepId },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueRequest {
    pub request: RequestId,
    pub phase: ExecutionRole,
    pub tenant: std::sync::Arc<str>,
    pub virtual_finish: u64,
    pub last_service_us: u64,
    /// Earliest deadline minus its remaining service forecast, in monotonic microseconds.
    pub latest_start_us: Option<i128>,
    pub hard_deadline_us: Option<u64>,
    pub wait_deadline_us: u64,
}
/// Completion updates copy scalars; tenant strings stay in the admission table.
#[derive(Debug, Clone, Copy)]
pub struct QueueTiming {
    pub phase: ExecutionRole,
    pub last_service_us: u64,
    /// Earliest deadline minus its remaining service forecast, in monotonic microseconds.
    pub latest_start_us: Option<i128>,
    pub wait_deadline_us: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Entry {
    request: QueueRequest,
    state: QueueState,
    tenant: usize,
    slot: usize,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Tenant {
    slot: usize,
    finish: u64,
    owners: usize,
}
type FairKey = (u64, u64, RequestId, usize);
type TenantKey = (usize, u64, RequestId);
#[derive(Debug, Clone, Copy)]
struct ReadyKey {
    phase: ExecutionRole,
    tenant: usize,
    service: u64,
    id: RequestId,
    latest_start: Option<i128>,
}
impl From<&Entry> for ReadyKey {
    fn from(entry: &Entry) -> Self {
        Self {
            phase: entry.request.phase,
            tenant: entry.tenant,
            service: entry.request.last_service_us,
            id: entry.request.request,
            latest_start: entry.request.latest_start_us,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueInspection {
    pub capacity: usize,
    pub prefill: usize,
    pub decode: usize,
    pub forward: usize,
    pub blocked: usize,
    pub running: usize,
    pub cancel_pending: usize,
    pub oldest_service_us: Option<u64>,
}
/// Single-owner lifecycle table with library B-trees and reusable, preallocated nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestQueue {
    capacity: usize,
    entries: BoundedMap<RequestId, Entry>,
    tenants: BoundedMap<std::sync::Arc<str>, Tenant>,
    names: Vec<Option<std::sync::Arc<str>>>,
    free_tenants: Vec<usize>,
    free_entries: Vec<usize>,
    ready: OrderedSet<RequestId>,
    scan_after: Option<RequestId>,
    tenant_ready: OrderedSet<TenantKey>,
    fair: OrderedSet<FairKey>,
    tenant_service: OrderedSet<(u64, usize)>,
    scratch: CandidateScratch,
    aging: OrderedSet<(u64, RequestId)>,
    deadlines: OrderedSet<(i128, RequestId)>,
    hard_expiry: OrderedSet<(u64, RequestId)>,
    wait_expiry: OrderedSet<(u64, RequestId)>,
    waiters: OrderedSet<(BlockedOn, u64, RequestId)>,
    flights: OrderedSet<(StepId, RequestId)>,
    phases: [usize; PHASE_COUNT],
    running: usize,
    cancelling: usize,
}
impl RequestQueue {
    /// # Errors
    /// Rejects zero capacity or failure to allocate fixed indexes.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::invalid("request queue capacity must be positive"));
        }
        let entries = BoundedMap::new(capacity)?;
        let tenants = BoundedMap::new(capacity)?;

        Ok(Self {
            capacity,
            entries,
            tenants,
            flights: OrderedSet::new(capacity, |slot| (StepId::ONE, seed_id(slot))),
            names: vec![None; capacity],
            free_tenants: (0..capacity).rev().collect(),
            free_entries: (0..capacity).rev().collect(),
            ready: OrderedSet::new(capacity, seed_id),
            scan_after: None,
            tenant_ready: OrderedSet::new(capacity, |slot| (0, 0, seed_id(slot))),
            fair: OrderedSet::new(capacity, |slot| (0, 0, seed_id(slot), 0)),
            tenant_service: OrderedSet::new(capacity, |slot| (0, slot)),
            scratch: CandidateScratch::new(capacity)?,
            aging: OrderedSet::new(capacity, |slot| (0, seed_id(slot))),
            deadlines: OrderedSet::new(capacity, |slot| (0, seed_id(slot))),
            hard_expiry: OrderedSet::new(capacity, |slot| (0, seed_id(slot))),
            wait_expiry: OrderedSet::new(capacity, |slot| (0, seed_id(slot))),
            waiters: OrderedSet::new(capacity, |slot| (BlockedOn::Memory, 0, seed_id(slot))),
            phases: [0; PHASE_COUNT],
            running: 0,
            cancelling: 0,
        })
    }
    const fn phase_index(phase: ExecutionRole) -> usize {
        match phase {
            ExecutionRole::Prefill => 0,
            ExecutionRole::Decode => 1,
            _ => 2,
        }
    }
    fn head(&self, tenant: usize) -> Option<FairKey> {
        let (_, service, id) = self
            .tenant_ready
            .range((tenant, 0, RequestId::ONE)..)
            .next()
            .filter(|key| key.0 == tenant)?;
        let name = self.names[tenant].as_ref()?;
        let finish = self.tenants.get(name)?.finish;
        Some((finish, *service, *id, tenant))
    }
    fn index(&mut self, key: ReadyKey, insert: bool) {
        let head = self.head(key.tenant);
        let tenant_key = (key.tenant, key.service, key.id);
        let changes_head = head.is_none_or(|head| (key.service, key.id) <= (head.1, head.2));
        if changes_head && let Some(head) = head {
            self.fair.remove(&head);
        }
        let service_key = (key.service, key.id);
        if insert {
            self.ready.insert(key.id);
            self.tenant_ready.insert(tenant_key);
            self.aging.insert(service_key);
            if let Some(deadline) = key.latest_start {
                self.deadlines.insert((deadline, key.id));
            }
            self.phases[Self::phase_index(key.phase)] += 1;
        } else {
            self.ready.remove(&key.id);
            self.tenant_ready.remove(&tenant_key);
            self.aging.remove(&service_key);
            if let Some(deadline) = key.latest_start {
                self.deadlines.remove(&(deadline, key.id));
            }
            self.phases[Self::phase_index(key.phase)] -= 1;
        }
        if changes_head && let Some(head) = self.head(key.tenant) {
            self.fair.insert(head);
        }
    }
    /// # Errors
    /// Rejects duplicates, invalid phases/tenants or a full lifecycle queue.
    pub fn enqueue(&mut self, request: QueueRequest) -> Result<()> {
        if request.phase == ExecutionRole::Mixed
            || request.tenant.is_empty()
            || self.entries.contains_key(&request.request)
        {
            return Err(Error::invalid("invalid or duplicate queue admission"));
        }
        if self.entries.len() >= self.capacity {
            return Err(Error::new(
                ErrorCode::Capacity,
                "scheduler request queue full",
            ));
        }
        let tenant = if let Some(tenant) = self.tenants.get_mut(&request.tenant) {
            tenant.owners += 1;
            tenant.slot
        } else {
            let slot = self
                .free_tenants
                .pop()
                .ok_or_else(|| Error::invariant("tenant slots exhausted"))?;
            self.names[slot] = Some(request.tenant.clone());
            self.tenant_service.insert((request.virtual_finish, slot));
            self.tenants.insert(
                request.tenant.clone(),
                Tenant {
                    slot,
                    finish: request.virtual_finish,
                    owners: 1,
                },
            )?;
            slot
        };
        if let Some(deadline) = request.hard_deadline_us {
            self.hard_expiry.insert((deadline, request.request));
        }
        self.wait_expiry
            .insert((request.wait_deadline_us, request.request));
        let slot = self
            .free_entries
            .pop()
            .ok_or_else(|| Error::invariant("request slots exhausted"))?;
        let entry = Entry {
            slot,
            request,
            state: QueueState::Ready,
            tenant,
        };
        self.index(ReadyKey::from(&entry), true);
        self.entries.insert(entry.request.request, entry)?;
        Ok(())
    }
    /// # Errors
    /// Rejects unknown requests, tenant/deadline changes or a still-owned flight.
    pub fn update_ready(&mut self, request: &QueueRequest) -> Result<()> {
        let entry = self
            .entries
            .get(&request.request)
            .ok_or_else(|| Error::invalid("unknown queue update"))?;
        if entry.request.tenant != request.tenant
            || entry.request.hard_deadline_us != request.hard_deadline_us
        {
            return Err(Error::invariant(
                "queue update changed tenant or hard deadline",
            ));
        }
        self.refresh(
            request.request,
            QueueTiming {
                phase: request.phase,
                last_service_us: request.last_service_us,
                latest_start_us: request.latest_start_us,
                wait_deadline_us: request.wait_deadline_us,
            },
        )
    }
    /// # Errors
    /// Rejects updates before a device fence or invalid execution phases.
    pub fn refresh(&mut self, id: RequestId, timing: QueueTiming) -> Result<()> {
        let entry = self
            .entries
            .get(&id)
            .ok_or_else(|| Error::invalid("unknown queue update"))?;
        if matches!(
            entry.state,
            QueueState::Running { .. } | QueueState::CancelPending { .. }
        ) || timing.phase == ExecutionRole::Mixed
        {
            return Err(Error::invariant("queue update before flight completion"));
        }
        let key = ReadyKey::from(entry);
        let state = entry.state;
        let old_wait = entry.request.wait_deadline_us;
        if state == QueueState::Ready {
            self.index(key, false);
        }
        if let QueueState::Blocked { reason, epoch } = state {
            self.waiters.remove(&(reason, epoch, id));
        }
        self.wait_expiry.remove(&(old_wait, id));
        self.wait_expiry.insert((timing.wait_deadline_us, id));
        let entry = self
            .entries
            .get_mut(&id)
            .ok_or_else(|| Error::invariant("queue owner missing"))?;
        entry.request.phase = timing.phase;
        entry.request.last_service_us = timing.last_service_us;
        entry.request.latest_start_us = timing.latest_start_us;
        entry.request.wait_deadline_us = timing.wait_deadline_us;
        entry.state = QueueState::Ready;
        let key = ReadyKey::from(&*entry);
        self.index(key, true);
        Ok(())
    }
    /// Tenant service changes only one outer priority node, irrespective of its request count.
    pub fn update_tenant(&mut self, tenant: &str, finish: u64) {
        let Some((slot, previous)) = self.tenants.get(tenant).map(|t| (t.slot, t.finish)) else {
            return;
        };
        if previous == finish {
            return;
        }
        let head = self.head(slot);
        if let Some(head) = head {
            self.fair.remove(&head);
        }
        self.tenant_service.remove(&(previous, slot));
        if let Some(tenant) = self.tenants.get_mut(tenant) {
            tenant.finish = finish;
        }
        self.tenant_service.insert((finish, slot));
        if let Some((_, service, id, slot)) = head {
            self.fair.insert((finish, service, id, slot));
        }
    }
    #[must_use]
    pub fn virtual_time(&self) -> u64 {
        self.tenant_service.first().map_or(0, |key| key.0)
    }
    #[must_use]
    pub fn tenant_slot(&self, id: RequestId) -> Option<usize> {
        self.entries.get(&id).map(|e| e.tenant)
    }
    #[must_use]
    pub fn state(&self, request: RequestId) -> Option<QueueState> {
        self.entries.get(&request).map(|e| e.state)
    }
    #[must_use]
    pub fn inspect(&self) -> QueueInspection {
        QueueInspection {
            capacity: self.capacity,
            prefill: self.phases[0],
            decode: self.phases[1],
            forward: self.phases[2],
            blocked: self.waiters.len(),
            running: self.running,
            cancel_pending: self.cancelling,
            oldest_service_us: self.aging.first().map(|key| key.0),
        }
    }
}
