//! Diagnostics responsibilities.
use super::Engine;
use infer_core::{Error, Result, event::EventKind, event::ObjectKind, event::SemanticEvent};
use infer_observe::DiagnosticCode;
use infer_spi::{BackendProvider, SchedulingPolicy};

impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    pub fn drain_events(&mut self) -> Vec<SemanticEvent> {
        self.drain_legacy_events().unwrap_or_else(|error| {
            self.retain_diagnostic(DiagnosticCode::InvariantViolation, &error, None, None);
            Vec::new()
        })
    }
    pub(crate) fn event(
        &mut self,
        kind: EventKind,
        object_kind: ObjectKind,
        object_id: u64,
        correlation_id: u64,
        arg0: u64,
        arg1: u64,
    ) {
        let event = SemanticEvent {
            timestamp_us: self.now_us,
            kind,
            object_kind,
            reserved: 0,
            object_id,
            correlation_id,
            arg0,
            arg1,
        };
        self.emit_semantic(event);
    }
    pub(crate) fn emit_semantic(&mut self, event: SemanticEvent) {
        self.observations.metrics.record(&event);
        self.events.emit(event);
    }
    pub(crate) fn diagnostic<T>(&mut self, error: Error) -> Result<T> {
        self.event(
            EventKind::InvariantViolation,
            ObjectKind::Step,
            self.inflight.as_ref().map_or(0, |s| s.step.id.get()),
            0,
            self.global_progress_epoch,
            0,
        );
        self.isolate(DiagnosticCode::InvariantViolation, &error);
        Err(error)
    }
    pub const fn last_diagnostic_snapshot(&self) -> Option<&crate::RuntimeSnapshot> {
        self.diagnostic_snapshot.as_ref()
    }
}
