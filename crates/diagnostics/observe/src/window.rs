//! Bounded event pages shared with immutable readers; appending copies at most one page.
use crate::{EventQuery, ObservedEvent};
use infer_core::{Error, RequestId, Result};
use std::{collections::VecDeque, sync::Arc};

const PAGE_EVENTS: usize = 128;
#[derive(Clone, Default)]
pub struct EventPages {
    pub pages: VecDeque<Arc<Vec<ObservedEvent>>>,
    pub head: usize,
    pub len: usize,
    pub next_sequence: u64,
}
impl EventPages {
    pub fn push(&mut self, event: ObservedEvent) {
        if self
            .pages
            .back()
            .is_none_or(|page| page.len() == PAGE_EVENTS)
        {
            self.pages
                .push_back(Arc::new(Vec::with_capacity(PAGE_EVENTS)));
        }
        if let Some(page) = self.pages.back_mut() {
            Arc::make_mut(page).push(event);
            self.len += 1;
        }
    }
    pub fn pop(&mut self) -> Option<&ObservedEvent> {
        if self
            .pages
            .front()
            .is_some_and(|page| self.head == page.len())
        {
            self.pages.pop_front();
            self.head = 0;
        }
        let event = self.pages.front()?.get(self.head)?;
        self.head += 1;
        self.len -= 1;
        Some(event)
    }
    pub fn iter(&self) -> impl Iterator<Item = &ObservedEvent> {
        self.pages
            .iter()
            .flat_map(|page| page.iter())
            .skip(self.head)
            .take(self.len)
    }
    pub fn after(&self, cursor: u64) -> impl Iterator<Item = &ObservedEvent> {
        self.pages
            .iter()
            .enumerate()
            .flat_map(move |(index, page)| {
                let start = page.partition_point(|event| event.sequence <= cursor);
                let start = if index == 0 {
                    start.max(self.head)
                } else {
                    start
                };
                page[start..].iter()
            })
            .take_while(|event| event.sequence < self.next_sequence)
    }
    pub fn query(
        &self,
        after: u64,
        limit: usize,
        request: Option<RequestId>,
        ring_dropped: u64,
        history_evicted: u64,
    ) -> Result<EventQuery> {
        if limit == 0 || limit > 4096 {
            return Err(Error::invalid("event query limit must be 1 to 4096"));
        }
        let oldest = self
            .iter()
            .next()
            .map_or(self.next_sequence, |e| e.sequence);
        let events: Vec<_> = self
            .after(after)
            .filter(|e| request.is_none_or(|id| e.request == Some(id)))
            .take(limit)
            .cloned()
            .collect();
        let next_cursor = events
            .last()
            .map_or_else(|| self.next_sequence.saturating_sub(1), |e| e.sequence);
        Ok(EventQuery {
            session: None,
            events,
            next_cursor,
            oldest_cursor: oldest,
            cursor_gap: after.saturating_add(1) < oldest,
            ring_dropped,
            history_evicted,
        })
    }
}
/// Read-only page ownership survives live-ring eviction without copying the event history.
#[derive(Clone)]
pub struct ObservationWindow {
    pub(crate) events: EventPages,
    pub(crate) diagnostics: Arc<VecDeque<crate::Diagnostic>>,
    pub(crate) retained: usize,
    pub history_evicted: u64,
    pub diagnostics_evicted: u64,
}
impl ObservationWindow {
    /// Limit a collector snapshot to the successful ring publications captured by the owner.
    /// Later events share their pages but are excluded from iterators and query cursors.
    #[must_use]
    pub fn as_of(mut self, sequence: u64, include_events: bool) -> Self {
        self.events.next_sequence = self.events.next_sequence.min(sequence.saturating_add(1));
        while self
            .events
            .pages
            .back()
            .is_some_and(|page| page.first().is_some_and(|event| event.sequence > sequence))
        {
            self.events.pages.pop_back();
        }
        let count: usize = self
            .events
            .pages
            .iter()
            .map(|page| page.partition_point(|event| event.sequence <= sequence))
            .sum();
        self.events.len = count.saturating_sub(self.events.head);
        self.retained = self.events.len;
        if !include_events {
            self.events = EventPages::default();
        }
        self
    }
    #[must_use]
    pub const fn retained(&self) -> usize {
        self.retained
    }
    #[must_use]
    pub fn diagnostics(&self) -> Vec<crate::Diagnostic> {
        self.diagnostics.iter().cloned().collect()
    }
    #[must_use]
    pub fn timeline(&self) -> Vec<ObservedEvent> {
        self.events.iter().cloned().collect()
    }
    /// # Errors
    /// Rejects limits outside the bounded response budget of 1 to 4096 events.
    pub fn query(
        &self,
        after: u64,
        limit: usize,
        request: Option<RequestId>,
        ring_dropped: u64,
    ) -> Result<EventQuery> {
        self.events
            .query(after, limit, request, ring_dropped, self.history_evicted)
    }
}
