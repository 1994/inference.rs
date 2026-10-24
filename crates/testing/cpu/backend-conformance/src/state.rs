//! State quote, lifecycle and prefix checks.
use crate::report::{Report, Status};
use infer_core::{ErrorCode, StateId};
use infer_ir::{OutputReadout, StateRecipe};
use infer_spi::BackendProvider;

/// Upper bound used for the pure prefix probes.
const PREFIX_PROBE: usize = 4;
/// Consecutive token ids used by the pure prefix probes.
const PREFIX_TOKENS: [u32; 3] = [1, 2, 3];

/// Check the state quote and the lifecycle a Tier-0 backend must satisfy.
///
/// `capacity` is a sequence length the backend accepts. The checks reserve and release one state,
/// so a device-gated caller should pass a capacity that is small but non-zero.
#[must_use]
pub fn backend_contract<B: BackendProvider>(backend: &mut B, capacity: usize) -> Report {
    let mut report = Report::new();
    if capacity == 0 {
        report.fail("state.capacity", "capacity must be nonzero");
        return report;
    }
    check_identity(backend, &mut report);
    check_quote(backend, capacity, &mut report);
    check_lifecycle(backend, capacity, &mut report);
    check_prefix(backend, &mut report);
    report
}

fn check_identity<B: BackendProvider>(backend: &B, report: &mut Report) {
    if backend.identity().is_empty() {
        report.fail("state.identity", "identity is empty");
    } else {
        report.pass("state.identity");
    }
    let capabilities = backend.capabilities();
    if capabilities.compute_dtypes.is_empty() {
        report.fail(
            "state.capabilities",
            "capabilities declare no compute dtype",
        );
    } else if backend.capabilities() != capabilities {
        report.fail("state.capabilities", "capabilities are not deterministic");
    } else {
        report.pass("state.capabilities");
    }
}

fn check_quote<B: BackendProvider>(backend: &B, capacity: usize, report: &mut Report) {
    let Some(recipe) = backend.state_recipe().cloned() else {
        report.skip("state.recipe", "backend does not publish a StateRecipe");
        return;
    };
    report.pass("state.recipe");
    if recipe.block_size == 0 || recipe.max_tokens < capacity {
        report.fail(
            "state.recipe_bounds",
            format!(
                "block size {} / max tokens {} cannot cover capacity {capacity}",
                recipe.block_size, recipe.max_tokens
            ),
        );
        return;
    }
    report.pass("state.recipe_bounds");
    let mut issues = Vec::new();
    let mut quotes = Vec::new();
    for sample in samples(capacity, recipe.block_size) {
        match backend.state_reservation_bytes_for(sample, OutputReadout::Full) {
            Ok(Some(quote)) => quotes.push((sample, quote)),
            Ok(None) => issues.push(format!("capacity {sample} has no quote")),
            Err(error) => issues.push(format!("capacity {sample} failed: {error}")),
        }
    }
    if !issues.is_empty() {
        report.fail("state.quote", issues.join("; "));
        return;
    }
    report.pass("state.quote");
    check_quote_floor(backend, &recipe, capacity, &quotes, report);
    check_quote_monotonic(&quotes, report);
    check_generic_quote(backend, capacity, report);
}

fn check_quote_floor<B: BackendProvider>(
    backend: &B,
    recipe: &StateRecipe,
    capacity: usize,
    priced: &[(usize, u64)],
    report: &mut Report,
) {
    let Ok(layout) = recipe.layout(capacity, OutputReadout::Full) else {
        report.fail("state.quote_floor", "recipe layout failed");
        return;
    };
    let pages = u64::try_from(layout.max_pages).unwrap_or(u64::MAX);
    let Some(kv) = layout.kv_block_bytes.checked_mul(pages) else {
        report.fail("state.quote_floor", "KV layout overflow");
        return;
    };
    let Some(minimum) = kv
        .checked_add(layout.private_bytes)
        .and_then(|total| total.checked_add(layout.host_bytes))
    else {
        report.fail("state.quote_floor", "state layout overflow");
        return;
    };
    let observed = priced
        .iter()
        .find(|(sample, _)| *sample == capacity)
        .map(|(_, value)| *value);
    match observed {
        Some(value) if value >= minimum => report.pass("state.quote_floor"),
        Some(value) => report.fail(
            "state.quote_floor",
            format!("quote {value} is below the recipe minimum {minimum}"),
        ),
        None => report.fail("state.quote_floor", "capacity was not quoted"),
    }
    // A quote must also be attainable from the backend's own accounting.
    if let Ok(Some(capacity_bytes)) = backend.state_capacity_bytes()
        && priced.iter().any(|(_, value)| *value > capacity_bytes)
    {
        report.fail(
            "state.quote_capacity",
            format!("a quote exceeds the declared capacity {capacity_bytes}"),
        );
    } else {
        report.pass("state.quote_capacity");
    }
}

fn check_quote_monotonic(quotes: &[(usize, u64)], report: &mut Report) {
    let mut sorted = quotes.to_vec();
    sorted.sort_unstable_by_key(|(capacity, _)| *capacity);
    if sorted.windows(2).all(|window| window[0].1 <= window[1].1) {
        report.pass("state.quote_monotonic");
    } else {
        report.fail(
            "state.quote_monotonic",
            "a larger capacity was quoted lower than a smaller one",
        );
    }
}

fn check_generic_quote<B: BackendProvider>(backend: &B, capacity: usize, report: &mut Report) {
    let generic = backend.state_reservation_bytes(capacity);
    let full = backend.state_reservation_bytes_for(capacity, OutputReadout::Full);
    match (generic, full) {
        (Ok(generic), Ok(full)) if generic == full => report.pass("state.quote_generic"),
        (Ok(generic), Ok(full)) => report.fail(
            "state.quote_generic",
            format!("generic quote {generic:?} differs from full-readout quote {full:?}"),
        ),
        (Err(error), _) | (_, Err(error)) => {
            report.fail("state.quote_generic", format!("quote failed: {error}"));
        }
    }
}

fn samples(capacity: usize, block_size: usize) -> Vec<usize> {
    let mut values = vec![1, block_size, capacity];
    values.retain(|value| *value > 0 && *value <= capacity);
    values.sort_unstable();
    values.dedup();
    values
}

fn check_lifecycle<B: BackendProvider>(backend: &mut B, capacity: usize, report: &mut Report) {
    let state = StateId::ONE;
    if let Err(error) = backend.reserve_state(state, capacity) {
        report.fail("state.reserve", error.to_string());
        return;
    }
    report.pass("state.reserve");
    check_duplicate_reserve(backend, state, capacity, report);
    match backend.release_state(state) {
        Ok(()) => report.pass("state.release"),
        Err(error) => report.fail("state.release", error.to_string()),
    }
    match backend.reserve_state(state, capacity) {
        Ok(()) => {
            report.pass("state.reuse");
            if let Err(error) = backend.release_state(state) {
                report.fail("state.reuse", error.to_string());
            }
        }
        Err(error) => report.fail("state.reuse", error.to_string()),
    }
}

fn check_duplicate_reserve<B: BackendProvider>(
    backend: &mut B,
    state: StateId,
    capacity: usize,
    report: &mut Report,
) {
    match backend.reserve_state(state, capacity) {
        Err(error) if error.code == ErrorCode::Conflict => report.pass("state.duplicate"),
        Err(error) => report.fail(
            "state.duplicate",
            format!("duplicate reserve returned {:?}: {error}", error.code),
        ),
        Ok(()) => report.fail("state.duplicate", "duplicate reserve was accepted"),
    }
}

fn check_prefix<B: BackendProvider>(backend: &B, report: &mut Report) {
    let mut issues = Vec::new();
    if backend.reusable_prefix(&[], PREFIX_PROBE) != 0 {
        issues.push("empty tokens matched a prefix".to_owned());
    }
    if backend.reusable_prefix(&PREFIX_TOKENS, 0) != 0 {
        issues.push("zero maximum matched a prefix".to_owned());
    }
    let limit = PREFIX_TOKENS.len() - 1;
    if backend.reusable_prefix(&PREFIX_TOKENS, limit) > limit {
        issues.push("prefix exceeded the requested maximum".to_owned());
    }
    let state = StateId::ONE;
    if backend.reusable_prefix_for(state, &[], PREFIX_PROBE) != 0 {
        issues.push("reserved-state empty tokens matched a prefix".to_owned());
    }
    if issues.is_empty() {
        report.pass("state.prefix");
    } else {
        report.fail("state.prefix", issues.join("; "));
    }
    check_kv_inspection(backend, report);
}

fn check_kv_inspection<B: BackendProvider>(backend: &B, report: &mut Report) {
    let Some(kv) = backend.kv_cache() else {
        report.optional("state.kv_cache", Status::Skipped("no KV inspection"));
        return;
    };
    let mut issues = Vec::new();
    if kv.block_size == 0 {
        issues.push("block size is zero".to_owned());
    }
    if kv.total_blocks == 0 {
        issues.push("no KV blocks".to_owned());
    }
    if kv.free_blocks > kv.total_blocks {
        issues.push("free blocks exceed total blocks".to_owned());
    }
    if issues.is_empty() {
        report.optional("state.kv_cache", Status::Passed);
    } else {
        report.optional("state.kv_cache", Status::Failed(issues.join("; ")));
    }
}

#[cfg(test)]
#[path = "../tests/unit/state.rs"]
mod tests;
