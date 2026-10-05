use crate::Engine;
use infer_core::{Error, RequestStatus, Result};
use infer_ir::{DeferReason, SchedulingDecision};
use infer_spi::{BackendProvider, SchedulingPolicy};

impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    pub(crate) fn append_focus_deferrals(&self, decision: &mut SchedulingDecision) -> Result<()> {
        if let Some(focus) = self.preemption_focus {
            for id in &self.host.ready_index.candidates {
                if *id != focus && self.host.requests.known(*id)?.status == RequestStatus::Runnable
                {
                    decision.deferred.push(infer_ir::DeferredWork {
                        request: *id,
                        reason: DeferReason::PreemptionFocus { request: focus },
                        required: 1,
                        available: 0,
                    });
                }
            }
        }
        let runnable = self.host.ready_index.candidates.len();
        if decision.selected.len() + decision.deferred.len() != runnable {
            return Err(Error::invariant(
                "dispatch evidence omitted a runnable request",
            ));
        }
        let queues = self.host.queues.inspect();
        decision.window = Some(infer_ir::DecisionWindow {
            epoch: self.global_progress_epoch,
            cost_epoch: self.host.cost_epoch,
            resource_epoch: self.resource_epoch,
            inspected: runnable,
            ready: queues.prefill + queues.decode + queues.forward,
        });
        Ok(())
    }
}
