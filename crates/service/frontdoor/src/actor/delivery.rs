//! Delivery responsibilities.
use super::RuntimeActor;
use infer_runtime::EngineOutput;
use infer_spi::BackendProvider;

impl<B: BackendProvider> RuntimeActor<B> {
    pub(super) fn deliver(&mut self, generated: &mut Vec<EngineOutput>) {
        for event in generated.drain(..) {
            let (id, terminal) = match &event {
                EngineOutput::Token { request, .. } => (*request, false),
                EngineOutput::Finished(done) => (done.request, true),
            };
            if terminal {
                let _ = self.engine.take_completed(id);
            }
            let Some(subscriber) = self.subscribers.get_mut(&id) else {
                continue;
            };
            if terminal {
                subscriber.terminal = Some(Ok(event));
                continue;
            }
            if subscriber.channel.try_send(Ok(event)).is_err() {
                self.engine.record_backpressure(id);
                // Retain one terminal record until the bounded queue drains or disconnects.
                if let Ok(events) = self.engine.cancel(id)
                    && let Some(done) = events
                        .into_iter()
                        .find(|e| matches!(e, EngineOutput::Finished(_)))
                {
                    subscriber.terminal = Some(Ok(done));
                    let _ = self.engine.take_completed(id);
                }
            }
        }
    }
    pub(super) fn deliver_pending_terminals(&mut self) {
        for (id, subscriber) in &mut self.subscribers {
            if subscriber.terminal.is_none()
                && let Some(done) = self.engine.pending_terminal(*id)
            {
                subscriber.terminal = Some(Ok(EngineOutput::Finished(done)));
            }
        }
    }
    pub(super) fn reap_completed(&mut self) {
        let completed: Vec<_> = self
            .engine
            .request_records()
            .filter(|r| r.status.terminal())
            .map(|r| r.request.id)
            .collect();
        for id in completed {
            let _ = self.engine.take_completed(id);
        }
    }
}
