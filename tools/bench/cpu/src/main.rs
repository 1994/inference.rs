//! Release-mode CPU protocol gate; no synthetic number is labeled model inference throughput.
#[allow(
    unsafe_code,
    reason = "Standalone measurement shim delegates GlobalAlloc to mimalloc; never linked into production"
)]
mod allocator;
mod cycle;
mod engine;
mod measure;
mod primitives;
#[global_allocator]
static ALLOCATOR: allocator::CountingAllocator = allocator::CountingAllocator;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("INFER_ENGINE_ONLY").is_some() {
        println!(
            "{}",
            serde_json::to_string_pretty(&[
                engine::run(1, 1)?,
                engine::run(16, 16)?,
                engine::run(64, 64)?
            ])?
        );
        return Ok(());
    }
    let mut cycles = Vec::new();
    for requests in [1, 64, 1024, 8192] {
        for batch in [1, 16, 64] {
            for context in [1024, 16384] {
                cycles.push(cycle::run(requests, batch, context)?);
            }
        }
    }
    let engine = [
        engine::run(1, 1)?,
        engine::run(16, 16)?,
        engine::run(64, 64)?,
    ];
    let primitives = primitives::run()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "scope":"Single-thread CPU protocol cycle: queue -> bounded ready projection -> packing -> independent validation -> metadata seal -> SPSC handoff -> launch ack -> synthetic fence -> commit. No model execution, sampling, network, device driver, thread scheduling or whole-engine latency is measured.",
            "primitives_scope":"Fixed map/arena/credits/pooled resource acknowledgement, logical KV batch growth/reset, and physical KV append/fork/COW/pin/release. Prefix publication and actual GPU execution are excluded.",
            "engine_scope":"Actual Engine CPU tick: admission/resource acknowledgement, prefill/decode scheduling, state reservations, packing, cost feedback/history, sampling, buffer recycling, completion and live cancellation. Each case runs 10000 ticks including the first prefill. Uses a persistent backend contract double; excludes native GPU execution/driver allocation, threaded owners, request parsing and network/telemetry export.",
            "allocator":"mimalloc with thread-local allocation/reallocation/deallocation counters",
            "target":std::env::consts::ARCH,"os":std::env::consts::OS,"cycles":cycles,"primitives":primitives,"engine":engine
        }))?
    );
    Ok(())
}
