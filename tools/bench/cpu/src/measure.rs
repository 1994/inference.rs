use crate::allocator::{self, Counts};
use infer_core::{Error, Result};
use serde::Serialize;
use std::time::Instant;
#[derive(Serialize)]
pub struct Measurement {
    pub iterations: usize,
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub max_ns: u64,
    pub elapsed_ns: u64,
    pub allocator: Counts,
}
pub fn measure(
    iterations: usize,
    mut operation: impl FnMut() -> Result<()>,
) -> Result<Measurement> {
    for _ in 0..128 {
        operation()?;
    }
    let mut samples = Vec::with_capacity(iterations);
    let start = Instant::now();
    allocator::start();
    let result = (0..iterations).try_for_each(|_| {
        let start = Instant::now();
        operation()?;
        samples.push(u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX));
        Ok(())
    });
    let counts = allocator::stop();
    result?;
    let elapsed_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    samples.sort_unstable();
    let percentile = |p: usize| samples[(iterations * p / 100).min(iterations - 1)];
    if counts.allocations != 0 || counts.reallocations != 0 || counts.deallocations != 0 {
        return Err(Error::invariant(format!(
            "CPU steady-state allocated: {} alloc, {} realloc, {} dealloc",
            counts.allocations, counts.reallocations, counts.deallocations
        )));
    }
    Ok(Measurement {
        iterations,
        p50_ns: percentile(50),
        p95_ns: percentile(95),
        p99_ns: percentile(99),
        max_ns: percentile(100),
        elapsed_ns,
        allocator: counts,
    })
}
