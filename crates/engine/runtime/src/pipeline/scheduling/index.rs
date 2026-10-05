//! Delta-updated request projections and a bounded, reusable candidate window.
use infer_core::{Error, ErrorCode, RequestId, Result};
use infer_ir::{CostEstimate, CostQuery, ExecutionRole, ReadyWork};

pub struct ReadyIndex {
    pub rows: Vec<Option<ReadyWork>>,
    pub versions: Vec<u64>,
    pub epoch: u64,
    pub candidates: Vec<RequestId>,
}
impl ReadyIndex {
    pub fn new(capacity: usize, limit: usize, query: CostQuery) -> Result<Self> {
        let mut rows = Vec::new();
        let mut candidates = Vec::new();
        rows.try_reserve_exact(capacity)
            .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
        candidates
            .try_reserve_exact(limit)
            .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
        rows.resize_with(capacity, || Some(placeholder(query)));
        Ok(Self {
            rows,
            versions: vec![u64::MAX; capacity],
            epoch: 0,
            candidates,
        })
    }
    pub fn invalidate(&mut self, slot: usize) -> Result<()> {
        self.versions[slot] = u64::MAX;
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| Error::invariant("ready epoch exhausted"))?;
        Ok(())
    }
}
pub struct ReadyWindow {
    storage: Vec<ReadyWork>,
    pub count: usize,
}
impl ReadyWindow {
    /// # Errors
    /// Reports inability to reserve the bounded candidate projection storage.
    pub fn new(capacity: usize, query: CostQuery) -> Result<Self> {
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(capacity)
            .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
        for _ in 0..capacity {
            storage.push(placeholder(query));
        }
        Ok(Self { storage, count: 0 })
    }
    /// # Errors
    /// Rejects writes beyond startup capacity without growing or overwriting the buffer.
    pub fn write(&mut self, row: &ReadyWork) -> Result<()> {
        let slot = self
            .storage
            .get_mut(self.count)
            .ok_or_else(|| Error::new(ErrorCode::Capacity, "ready projection buffer full"))?;
        slot.tenant.clone_from(&row.tenant);
        slot.request = row.request;
        slot.program = row.program;
        slot.state = row.state;
        slot.role = row.role;
        slot.remaining_tokens = row.remaining_tokens;
        slot.weight = row.weight;
        slot.deadline_us = row.deadline_us;
        slot.virtual_finish = row.virtual_finish;
        slot.cost_per_token = row.cost_per_token;
        slot.cost_query = row.cost_query;
        slot.latency_deadline_us = row.latency_deadline_us;
        slot.remaining_latency_us = row.remaining_latency_us;
        slot.remaining_completion_us = row.remaining_completion_us;
        slot.last_service_us = row.last_service_us;
        self.count += 1;
        Ok(())
    }
    pub fn as_mut_slice(&mut self) -> &mut [ReadyWork] {
        &mut self.storage[..self.count]
    }
    #[must_use]
    pub fn as_slice(&self) -> &[ReadyWork] {
        &self.storage[..self.count]
    }
}
impl std::ops::Deref for ReadyWindow {
    type Target = [ReadyWork];
    fn deref(&self) -> &[ReadyWork] {
        self.as_slice()
    }
}

fn placeholder(query: CostQuery) -> ReadyWork {
    ReadyWork {
        request: RequestId::ONE,
        program: query.program,
        state: infer_core::StateId::ONE,
        role: ExecutionRole::Prefill,
        remaining_tokens: 0,
        tenant: String::with_capacity(256),
        weight: 1,
        deadline_us: None,
        virtual_finish: 0,
        cost_per_token: CostEstimate::default(),
        cost_query: query,
        latency_deadline_us: None,
        remaining_latency_us: 0,
        remaining_completion_us: 0,
        last_service_us: 0,
    }
}
