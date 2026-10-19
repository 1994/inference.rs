//! Preallocated candidate membership marks and the transient tenant merge heap.
use super::FairKey;
use infer_core::{Error, ErrorCode, Result};
use std::collections::TryReserveError;
use std::mem::replace;
use std::{cmp::Reverse, collections::BinaryHeap};

#[derive(Debug)]
pub(super) struct CandidateScratch {
    pub seen: Vec<u64>,
    pub generation: u64,
    pub frontier: BinaryHeap<Reverse<FairKey>>,
}
impl CandidateScratch {
    pub fn new(capacity: usize) -> Result<Self> {
        let mut seen = Vec::new();
        let mut frontier = BinaryHeap::new();
        let error = |error: TryReserveError| Error::new(ErrorCode::Capacity, error.to_string());
        seen.try_reserve_exact(capacity).map_err(error)?;
        frontier.try_reserve_exact(capacity).map_err(error)?;
        seen.resize(capacity, 0);
        Ok(Self {
            seen,
            generation: 0,
            frontier,
        })
    }
    pub fn begin(&mut self) {
        self.frontier.clear();
        if let Some(next) = self.generation.checked_add(1) {
            self.generation = next;
        } else {
            self.seen.fill(0);
            self.generation = 1;
        }
    }
    pub fn visit(&mut self, slot: usize) -> bool {
        let previous = replace(&mut self.seen[slot], self.generation);
        previous != self.generation
    }
    pub fn validate(&self, capacity: usize) -> Result<()> {
        if self.seen.len() != capacity
            || self.frontier.capacity() < capacity
            || !self.frontier.is_empty()
        {
            return Err(Error::invariant(
                "candidate scratch capacity or ownership corrupt",
            ));
        }
        Ok(())
    }
}
impl Clone for CandidateScratch {
    fn clone(&self) -> Self {
        let mut frontier = BinaryHeap::with_capacity(self.seen.len());
        frontier.extend(self.frontier.iter().copied());
        Self {
            seen: self.seen.clone(),
            generation: self.generation,
            frontier,
        }
    }
}
impl PartialEq for CandidateScratch {
    fn eq(&self, other: &Self) -> bool {
        self.seen.len() == other.seen.len()
    }
}
impl Eq for CandidateScratch {}
