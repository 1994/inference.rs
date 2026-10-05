//! Isolated history handoff cost; exporters and engine throughput are excluded.
use infer_core::{Result, event::EventKind, event::ObjectKind, event::SemanticEvent};
use infer_observe::ObservationStore;
use std::{hint::black_box, time::Instant};

fn run(capacity: usize) -> Result<()> {
    let mut store = ObservationStore::new(capacity)?;
    for sequence in 0..capacity {
        store.record(SemanticEvent {
            timestamp_us: sequence as u64,
            kind: EventKind::Accepted,
            object_kind: ObjectKind::Request,
            reserved: 0,
            object_id: sequence as u64 % 256 + 1,
            correlation_id: 0,
            arg0: 0,
            arg1: 0,
        });
    }
    let started = Instant::now();
    for _ in 0..200 {
        black_box(store.timeline());
    }
    let copy = started.elapsed().as_nanos() / 200;
    let started = Instant::now();
    for _ in 0..200 {
        black_box(store.snapshot(true));
    }
    let shared = started.elapsed().as_nanos() / 200;
    println!("{capacity},{copy},{shared}");
    Ok(())
}
fn main() -> Result<()> {
    println!("events,materialize_ns,shared_snapshot_ns");
    for capacity in [4096, 65_536, 1_048_576] {
        run(capacity)?;
    }
    Ok(())
}
