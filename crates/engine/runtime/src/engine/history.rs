//! History responsibilities.
use super::Engine;
use infer_spi::{BackendProvider, SchedulingPolicy};

impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    pub(crate) fn record_action(&mut self, action: crate::ReplayAction) {
        if self.actions.len() == self.config.history_capacity {
            self.evict_action();
        }
        let retained = action.retained_tokens();
        let bytes = action.retained_bytes();
        while self.host.history_tokens.saturating_add(retained) > self.config.max_history_tokens
            || self.host.history_bytes.saturating_add(bytes) > self.config.max_history_bytes
        {
            if self.actions.is_empty() {
                self.dropped_actions = self.dropped_actions.saturating_add(1);
                return;
            }
            self.evict_action();
        }
        self.host.history_tokens = self.host.history_tokens.saturating_add(retained);
        self.host.history_bytes = self.host.history_bytes.saturating_add(bytes);
        self.actions.push_back(action);
    }
    pub(super) fn evict_action(&mut self) {
        if let Some(old) = self.actions.pop_front() {
            self.host.history_bytes = self.host.history_bytes.saturating_sub(old.retained_bytes());
            self.host.history_tokens = self
                .host
                .history_tokens
                .saturating_sub(old.retained_tokens());
            self.dropped_actions = self.dropped_actions.saturating_add(1);
        }
    }
}
