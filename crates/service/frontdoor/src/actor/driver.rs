//! Driver responsibilities.
use super::{
    Command, DEFERRED_QUEUE_CAPACITY, QUOTING_QUEUE_CAPACITY, RuntimeActor, STOP_WAITER_CAPACITY,
    mpsc,
};
use infer_runtime::{Engine, EngineOutput};
use infer_spi::BackendProvider;
use std::{
    collections::BTreeMap, collections::VecDeque, sync::Arc, sync::atomic::AtomicBool,
    sync::atomic::Ordering, time::Duration, time::Instant,
};

/// Microseconds the idle actor parks before re-checking engine and queue state.
const IDLE_PARK_TIMEOUT_MICROS: u64 = 250;

impl<B: BackendProvider> RuntimeActor<B> {
    pub(super) fn new(
        mut engine: Engine<B>,
        receiver: mpsc::Receiver<Command>,
        control: mpsc::Receiver<Command>,
        stop: Arc<AtomicBool>,
        clock_started: Instant,
    ) -> Self {
        let owner = std::thread::current();
        engine.set_waker(Arc::new(move || owner.unpark()));
        Self {
            clock_origin_us: engine.now_us(),
            engine,
            receiver,
            control,
            stop,
            deferred: VecDeque::with_capacity(DEFERRED_QUEUE_CAPACITY),
            quoting: VecDeque::with_capacity(QUOTING_QUEUE_CAPACITY),
            subscribers: BTreeMap::new(),
            start: clock_started,
            stopping: false,
            stop_replies: Vec::with_capacity(STOP_WAITER_CAPACITY),
        }
    }
    pub(super) fn run(mut self) {
        let mut generated = Vec::with_capacity(self.engine.config().max_requests.saturating_mul(2));
        loop {
            self.stopping |= self.stop.load(Ordering::Acquire);
            self.poll_engine(&mut generated);
            // Publish and retire fenced results before acknowledging control-plane queries.
            self.deliver(&mut generated);
            self.reap_completed();
            self.poll_admissions(&mut generated);
            self.receive_commands(&mut generated);
            self.cancel_closed_streams(&mut generated);
            self.drive_engine(&mut generated);
            self.engine.collect_observations();
            self.deliver(&mut generated);
            self.deliver_pending_terminals();
            self.subscribers.retain(|_, subscriber| subscriber.flush());
            self.reap_completed();
            if self.stopping && self.engine.is_idle() && self.quoting.is_empty() {
                break;
            }
            if self.engine.is_idle()
                && self.subscribers.is_empty()
                && self.deferred.is_empty()
                && self.quoting.is_empty()
                && !self.stopping
            {
                std::thread::park();
            } else {
                std::thread::park_timeout(Duration::from_micros(IDLE_PARK_TIMEOUT_MICROS));
            }
        }
        for reply in self.stop_replies {
            let _ = reply.send(Ok(()));
        }
    }
    pub(super) fn now_us(&self) -> u64 {
        self.clock_origin_us
            .saturating_add(u64::try_from(self.start.elapsed().as_micros()).unwrap_or(u64::MAX))
    }
    pub(super) fn poll_engine(&mut self, generated: &mut Vec<EngineOutput>) {
        match self.engine.poll_completed_into(self.now_us(), generated) {
            Ok(()) => {}
            Err(error) => {
                for s in self.subscribers.values_mut() {
                    s.terminal = Some(Err(error.clone()));
                }
            }
        }
    }
    pub(super) fn drive_engine(&mut self, generated: &mut Vec<EngineOutput>) {
        let now = self.now_us();
        match self.engine.tick_into(now, generated) {
            Ok(()) => {}
            Err(error) => {
                for s in self.subscribers.values_mut() {
                    s.terminal = Some(Err(error.clone()));
                }
            }
        }
    }
}
