//! Command admission, engine driving, and bounded output delivery.
mod admission;
mod commands;
mod delivery;
mod driver;
mod handle;
mod startup;
use infer_core::{Error, RequestId, Result};
use infer_ir::CanonicalRequest;
use infer_observe::{ObservationQuery, trace::TraceContext};
use infer_runtime::{Engine, EngineOutput, RuntimeInspection};
use infer_spi::BackendProvider;
use std::{
    collections::BTreeMap, collections::VecDeque, sync::Arc, sync::atomic::AtomicBool,
    sync::atomic::Ordering, sync::mpsc, sync::mpsc::SyncSender, time::Instant,
};
use tokio::{sync::mpsc as async_mpsc, sync::oneshot};

type OutputChannel = async_mpsc::Sender<Result<EngineOutput>>;
enum Command {
    Submit {
        request: Box<infer_runtime::PreparedRequest>,
        pending: Arc<crate::ingress::Pending>,
        trace: Option<TraceContext>,
        output: OutputChannel,
        reply: oneshot::Sender<Result<()>>,
    },
    Cancel {
        request: RequestId,
        preparing: bool,
        reply: oneshot::Sender<Result<()>>,
    },
    Inspect {
        reply: oneshot::Sender<RuntimeInspection>,
    },
    Observe {
        query: ObservationQuery,
        credit: crate::cpu::Credit,
        reply: oneshot::Sender<Result<(infer_runtime::ObservationSnapshot, crate::cpu::Credit)>>,
    },
    Stop {
        reply: oneshot::Sender<Result<()>>,
    },
}
struct Subscriber {
    channel: OutputChannel,
    terminal: Option<Result<EngineOutput>>,
}
#[derive(Clone)]
pub struct RuntimeHandle {
    commands: Arc<SyncSender<Command>>,
    control: Arc<SyncSender<Command>>,
    wake: std::thread::Thread,
    stop: Arc<AtomicBool>,
    preparer: infer_runtime::RequestPreparer,
    cpu: crate::cpu::CpuPool,
    delivery: crate::cpu::CpuPool,
    observation: crate::cpu::CpuPool,
    observation_bytes: usize,
    config: crate::cpu::CpuConfig,
    ingress: Arc<crate::ingress::Registry>,
    clock_started: Instant,
    clock_origin_us: u64,
}

struct RuntimeActor<B: BackendProvider> {
    engine: Engine<B>,
    receiver: mpsc::Receiver<Command>,
    control: mpsc::Receiver<Command>,
    stop: Arc<AtomicBool>,
    deferred: VecDeque<Command>,
    quoting: VecDeque<(Command, infer_spi::ResourceTicket)>,
    subscribers: BTreeMap<RequestId, Subscriber>,
    start: Instant,
    clock_origin_us: u64,
    stopping: bool,
    stop_replies: Vec<oneshot::Sender<Result<()>>>,
}
impl Subscriber {
    fn flush(&mut self) -> bool {
        if self.channel.is_closed() && self.terminal.is_some() {
            return false;
        }
        let Some(done) = self.terminal.take() else {
            return true;
        };
        match self.channel.try_send(done) {
            Ok(()) | Err(async_mpsc::error::TrySendError::Closed(_)) => false,
            Err(async_mpsc::error::TrySendError::Full(done)) => {
                self.terminal = Some(done);
                true
            }
        }
    }
}

fn preparation_bytes(request: &CanonicalRequest) -> Result<usize> {
    let tokens = match &request.input {
        infer_ir::RequestInput::Sequence { tokens, .. } => tokens.len(),
        infer_ir::RequestInput::Pairs { query, documents } => documents
            .iter()
            .try_fold(0usize, |n, doc| {
                n.checked_add(query.len())
                    .and_then(|n| n.checked_add(doc.len()))
            })
            .ok_or_else(|| Error::invalid("preparation token size overflow"))?,
    };
    let generated = if let infer_ir::Workload::Generate { max_new_tokens } = request.workload {
        max_new_tokens
    } else {
        0
    };
    tokens
        .checked_mul(8)
        .and_then(|n| {
            generated
                .checked_mul(4)
                .and_then(|tail| n.checked_add(tail))
        })
        .ok_or_else(|| Error::invalid("preparation byte size overflow"))
}

impl Drop for RuntimeHandle {
    fn drop(&mut self) {
        if Arc::strong_count(&self.commands) == 1 {
            self.stop.store(true, Ordering::Release);
            self.wake.unpark();
        }
    }
}

// Charge retained pages and conservative exporter expansion before the scheduler captures them.
fn observation_bytes(config: &infer_runtime::RuntimeConfig) -> Result<usize> {
    config
        .history_capacity
        .checked_add(256)
        .and_then(|events| events.checked_mul(size_of::<infer_observe::ObservedEvent>() * 32))
        .and_then(|bytes| {
            config
                .max_requests
                .checked_mul(1024)
                .and_then(|parents| bytes.checked_add(parents))
        })
        .and_then(|bytes| bytes.checked_add(64 << 10))
        .ok_or_else(|| Error::invalid("observation staging budget overflow"))
}

#[cfg(test)]
mod tests;
