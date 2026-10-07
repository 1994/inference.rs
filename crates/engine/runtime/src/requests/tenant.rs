//! Constant-time admission accounting; independent cold audits verify lifecycle totals.
use crate::Engine;
use infer_core::{Error, Result};
use infer_spi::{BackendProvider, SchedulingPolicy};
use std::collections::BTreeMap;

#[derive(Default)]
struct Usage {
    owners: usize,
    active: usize,
    tokens: usize,
    pages: usize,
}
impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    pub(crate) fn check_tenant_accounting(&self) -> Result<()> {
        let mut expected: BTreeMap<&str, Usage> = BTreeMap::new();
        for record in self.host.requests.values() {
            let tenant = record.request.qos.tenant.as_str();
            let service = self
                .tenants
                .get(tenant)
                .ok_or_else(|| Error::invariant("request tenant absent"))?;
            if service.weight != record.request.qos.weight {
                return Err(Error::invariant("tenant weight differs from request"));
            }
            let usage = expected.entry(tenant).or_default();
            usage.owners += 1;
            if !record.status.terminal() {
                usage.active += 1;
                usage.tokens = usage
                    .tokens
                    .checked_add(record.plan.reserved_tokens)
                    .ok_or_else(|| Error::invariant("tenant token audit overflow"))?;
                usage.pages = usage
                    .pages
                    .checked_add(record.plan.reserved_tokens.div_ceil(self.config.block_size))
                    .ok_or_else(|| Error::invariant("tenant page audit overflow"))?;
            }
        }
        if expected.len() != self.tenants.len()
            || self.tenants.iter().any(|(name, actual)| {
                expected.get(name.as_ref()).is_none_or(|usage| {
                    (usage.owners, usage.active, usage.tokens, usage.pages)
                        != (actual.owners, actual.active, actual.tokens, actual.pages)
                })
            })
        {
            return Err(Error::invariant(
                "tenant accounting differs from lifecycle ownership",
            ));
        }
        Ok(())
    }
}
