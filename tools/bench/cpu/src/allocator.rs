//! Measurement-only allocator boundary. Production crates retain their unsafe-code prohibition.
use serde::Serialize;
use std::{
    alloc::{GlobalAlloc, Layout},
    cell::Cell,
};
#[derive(Clone, Copy, Default, Serialize)]
pub struct Counts {
    pub allocations: u64,
    pub reallocations: u64,
    pub deallocations: u64,
    pub requested_bytes: u64,
}
thread_local! {
    static COUNTS: Cell<Option<Counts>> = const { Cell::new(None) };
}
fn record(update: impl FnOnce(&mut Counts)) {
    let _ = COUNTS.try_with(|counter| {
        if let Some(mut counts) = counter.get() {
            update(&mut counts);
            counter.set(Some(counts));
        }
    });
}
pub fn start() {
    COUNTS.with(|counts| counts.set(Some(Counts::default())));
}
pub fn stop() -> Counts {
    COUNTS.with(|counts| counts.replace(None).unwrap_or_default())
}
/// Diagnostic trap: capture the stack of the first allocation seen after [`arm_trap`].
static TRAP_ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static TRAP_STACK: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Arms the one-shot allocation trap when `INFER_CPU_BENCH_ALLOC_TRAP` is set.
///
/// The trap's own stack capture allocates, so it is off by default and must never run while
/// the gate counts are being interpreted.
pub fn arm_trap() {
    if std::env::var_os("INFER_CPU_BENCH_ALLOC_TRAP").is_some() {
        TRAP_ARMED.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Stack of the first allocation captured after arming, when one was seen.
#[must_use]
pub fn trapped_stack() -> Option<&'static str> {
    TRAP_STACK.get().map(String::as_str)
}

fn trap() {
    if !TRAP_ARMED.swap(false, std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    // The flag is already cleared, so the allocations this capture performs cannot re-enter it.
    if let Ok(text) =
        std::panic::catch_unwind(|| std::backtrace::Backtrace::force_capture().to_string())
    {
        let _ = TRAP_STACK.set(text);
    }
}

pub struct CountingAllocator;
// SAFETY: Every allocation operation delegates the exact layout/pointer to mimalloc. TLS bookkeeping
// uses only a Copy Cell and cannot allocate, reenter the allocator, or alter the returned allocation.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        trap();
        record(|counts| {
            counts.allocations += 1;
            counts.requested_bytes += layout.size() as u64;
        });
        // SAFETY: The caller's GlobalAlloc layout contract is forwarded without alteration.
        unsafe { mimalloc::MiMalloc.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        trap();
        record(|counts| {
            counts.allocations += 1;
            counts.requested_bytes += layout.size() as u64;
        });
        // SAFETY: The caller's valid layout and zero-initialization requirement are forwarded.
        unsafe { mimalloc::MiMalloc.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        trap();
        record(|counts| {
            counts.reallocations += 1;
            counts.requested_bytes += size as u64;
        });
        // SAFETY: Pointer ownership, original layout and the requested new size come from GlobalAlloc.
        unsafe { mimalloc::MiMalloc.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        record(|counts| counts.deallocations += 1);
        // SAFETY: The caller supplies a live allocation and its original matching layout.
        unsafe { mimalloc::MiMalloc.dealloc(pointer, layout) }
    }
}
