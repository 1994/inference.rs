//! Single-stream reference device; no production runtime state is shared with Candle.
use crate::Result;
use candle_core::Device;

#[expect(
    unsafe_code,
    reason = "Benchmark owns the sole execution stream and synchronizes before disabling cross-stream event tracking"
)]
pub fn reference_device() -> Result<Device> {
    let device = Device::new_cuda_with_stream(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    stream.context().synchronize()?;
    // SAFETY: this process creates one device/stream, executes sequentially, and all
    // Candle buffers are allocated only after this call. All copies, launches and frees
    // use that stream. Explicit synchronization precedes readback and graph destruction.
    // No buffer is shared with the engine or another stream. Tracking otherwise adds
    // waits on pre-capture events, which CUDA rejects as capture isolation violations.
    unsafe { stream.context().disable_event_tracking() };
    Ok(device)
}
