use crate::{Diagnostic, window::EventPages, window::ObservationWindow};
use infer_core::{
    Error, RequestId, Result, event::EventKind, event::ObjectKind, event::SemanticEvent,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, collections::VecDeque, sync::Arc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservedEvent {
    pub sequence: u64,
    pub request: Option<RequestId>,
    pub event: SemanticEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventQuery {
    pub session: Option<String>,
    pub events: Vec<ObservedEvent>,
    pub next_cursor: u64,
    pub oldest_cursor: u64,
    pub cursor_gap: bool,
    pub ring_dropped: u64,
    pub history_evicted: u64,
}

/// A collector is an explicit cold-path consumer; exporters never run in event emission.
pub struct ObservationStore {
    events: EventPages,
    diagnostics: Arc<VecDeque<Diagnostic>>,
    requests: BTreeMap<RequestId, usize>,
    capacity: usize,
    next_diagnostic: u64,
    pub history_evicted: u64,
    pub diagnostics_evicted: u64,
}
impl ObservationStore {
    /// # Errors
    /// Returns invalid input for an empty or excessively large retention capacity.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity > 1_048_576 {
            return Err(Error::invalid("observation capacity must be 1 to 1048576"));
        }
        Ok(Self {
            events: EventPages {
                next_sequence: 1,
                ..Default::default()
            },
            diagnostics: Arc::new(VecDeque::with_capacity(256)),
            requests: BTreeMap::new(),
            capacity,
            next_diagnostic: 1,
            history_evicted: 0,
            diagnostics_evicted: 0,
        })
    }
    pub fn record(&mut self, event: SemanticEvent) {
        let _ = self.record_retirement(event);
    }
    /// Report an identity whose last retained event was evicted, after inserting the new event.
    pub fn record_retirement(&mut self, event: SemanticEvent) -> Option<RequestId> {
        let mut retired = None;
        let request = match event.object_kind {
            ObjectKind::Request => RequestId::new(event.object_id).ok(),
            ObjectKind::State => RequestId::new(event.correlation_id).ok(),
            _ => None,
        };
        if self.events.len == self.capacity {
            if let Some(id) = self.events.pop().and_then(|event| event.request)
                && let Some(count) = self.requests.get_mut(&id)
            {
                *count -= 1;
                if *count == 0 {
                    self.requests.remove(&id);
                    retired = Some(id);
                }
            }
            self.history_evicted = self.history_evicted.saturating_add(1);
        }
        if let Some(id) = request {
            *self.requests.entry(id).or_default() += 1;
        }
        self.events.push(ObservedEvent {
            sequence: self.events.next_sequence,
            request,
            event,
        });
        self.events.next_sequence = self.events.next_sequence.saturating_add(1);
        retired.filter(|id| !self.requests.contains_key(id))
    }
    pub fn diagnostic(&mut self, mut diagnostic: Diagnostic) {
        diagnostic.sequence = self.next_diagnostic;
        self.next_diagnostic = self.next_diagnostic.saturating_add(1);
        if self.diagnostics.len() == 256 {
            Arc::make_mut(&mut self.diagnostics).pop_front();
            self.diagnostics_evicted = self.diagnostics_evicted.saturating_add(1);
        }
        Arc::make_mut(&mut self.diagnostics).push_back(diagnostic);
    }
    #[must_use]
    pub fn diagnostics(&self) -> Vec<Diagnostic> {
        self.diagnostics.iter().cloned().collect()
    }
    /// # Errors
    /// Returns invalid input for a query exceeding the per-response event budget.
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
    /// Share event pages and the bounded diagnostic window; no event payload is copied.
    #[must_use]
    pub fn snapshot(&self, include_events: bool) -> ObservationWindow {
        ObservationWindow {
            events: if include_events {
                self.events.clone()
            } else {
                EventPages::default()
            },
            diagnostics: self.diagnostics.clone(),
            retained: self.events.len,
            history_evicted: self.history_evicted,
            diagnostics_evicted: self.diagnostics_evicted,
        }
    }
    #[must_use]
    pub const fn retained(&self) -> usize {
        self.events.len
    }
    #[must_use]
    pub fn contains_request(&self, id: u64) -> bool {
        RequestId::new(id).is_ok_and(|id| self.requests.contains_key(&id))
    }
    #[must_use]
    pub fn timeline(&self) -> Vec<ObservedEvent> {
        self.events.iter().cloned().collect()
    }
    pub fn legacy_events(&self, cursor: &mut u64) -> Vec<SemanticEvent> {
        let events = self
            .events
            .after(*cursor)
            .map(|event| event.event)
            .collect();
        *cursor = self.events.next_sequence.saturating_sub(1);
        events
    }
}

#[must_use]
pub const fn finish_code(reason: &infer_core::FinishReason) -> u64 {
    match reason {
        infer_core::FinishReason::Length => 0,
        infer_core::FinishReason::Eos => 1,
        infer_core::FinishReason::Completed => 2,
        infer_core::FinishReason::Cancelled => 3,
        infer_core::FinishReason::Deadline => 4,
        infer_core::FinishReason::Failed(_) => 5,
    }
}

#[must_use]
pub const fn error_code(code: infer_core::ErrorCode) -> u64 {
    match code {
        infer_core::ErrorCode::InvalidInput => 1,
        infer_core::ErrorCode::Unsupported => 2,
        infer_core::ErrorCode::Capacity => 3,
        infer_core::ErrorCode::NotFound => 4,
        infer_core::ErrorCode::Conflict => 5,
        infer_core::ErrorCode::Invariant => 6,
        infer_core::ErrorCode::Backend => 7,
    }
}

#[must_use]
pub const fn diagnostic_for_event(kind: EventKind) -> Option<crate::DiagnosticCode> {
    match kind {
        EventKind::Rejected => Some(crate::DiagnosticCode::AdmissionRejected),
        EventKind::InvariantViolation => Some(crate::DiagnosticCode::InvariantViolation),
        EventKind::Backpressure => Some(crate::DiagnosticCode::OutputBackpressure),
        _ => None,
    }
}
