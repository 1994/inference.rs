use infer_core::{Error, ErrorCode, ProgramId, Result};
use infer_ir::{CanonicalRequest, ModelIr, ModelOutput, Workload, WorkloadOutput, WorkloadPlan};
use infer_spi::{ProviderMetadata, WorkloadProvider};

/// Maximum number of providers one registry accepts.
const MAX_REGISTRY_PROVIDERS: usize = 64;
/// Maximum bytes in a provider's registered display name.
const MAX_PROVIDER_NAME_BYTES: usize = 256;
/// Maximum bytes in a provider's self-reported identity string.
const MAX_PROVIDER_IDENTITY_BYTES: usize = 1024;

struct Entry {
    metadata: ProviderMetadata,
    priority: i32,
    provider: Box<dyn WorkloadProvider + Send + Sync>,
}
pub struct WorkloadRegistry {
    entries: Vec<Entry>,
    identity: String,
}
impl Default for WorkloadRegistry {
    fn default() -> Self {
        Self {
            entries: vec![],
            identity: "workload-registry-v1[]".into(),
        }
    }
}
impl WorkloadRegistry {
    ///
    /// # Errors
    /// Returns an invalid-input or conflict error for incompatible provider metadata or duplicate registrations.
    pub fn register(
        &mut self,
        metadata: ProviderMetadata,
        priority: i32,
        provider: impl WorkloadProvider + Send + Sync + 'static,
    ) -> Result<()> {
        metadata.validate()?;
        if self.entries.len() >= MAX_REGISTRY_PROVIDERS {
            return Err(Error::new(
                ErrorCode::Capacity,
                "workload provider registry full",
            ));
        }
        if self
            .entries
            .iter()
            .any(|e| e.metadata.id == metadata.id || e.metadata.name == metadata.name)
        {
            return Err(Error::new(
                ErrorCode::Conflict,
                "duplicate workload provider identity",
            ));
        }
        if metadata.name.len() > MAX_PROVIDER_NAME_BYTES
            || provider.identity().len() > MAX_PROVIDER_IDENTITY_BYTES
        {
            return Err(Error::invalid("provider identity exceeds budget"));
        }
        self.entries.push(Entry {
            metadata,
            priority,
            provider: Box::new(provider),
        });
        self.entries
            .sort_by_key(|e| (std::cmp::Reverse(e.priority), e.metadata.id));
        self.identity = format!(
            "workload-registry-v1{:?}",
            self.entries
                .iter()
                .map(|e| (
                    e.metadata.id.get(),
                    e.metadata.name.as_str(),
                    e.priority,
                    e.provider.identity()
                ))
                .collect::<Vec<_>>()
        );
        Ok(())
    }
    fn select(&self, workload: &Workload) -> Result<&dyn WorkloadProvider> {
        self.entries
            .iter()
            .find(|e| e.provider.supports(workload))
            .map(|e| e.provider.as_ref() as &dyn WorkloadProvider)
            .ok_or_else(|| {
                Error::unsupported("no registered workload provider accepts this request")
            })
    }
}
impl WorkloadProvider for WorkloadRegistry {
    fn fork(&self) -> Option<Box<dyn WorkloadProvider + Send + Sync>> {
        let mut entries = vec![];
        for entry in &self.entries {
            entries.push(Entry {
                metadata: entry.metadata.clone(),
                priority: entry.priority,
                provider: entry.provider.fork()?,
            });
        }
        Some(Box::new(Self {
            entries,
            identity: self.identity.clone(),
        }))
    }
    fn identity(&self) -> &str {
        &self.identity
    }
    fn supports(&self, workload: &Workload) -> bool {
        self.entries.iter().any(|e| e.provider.supports(workload))
    }
    fn plan(
        &self,
        request: &CanonicalRequest,
        model: &ModelIr,
        program: ProgramId,
    ) -> Result<WorkloadPlan> {
        self.select(&request.workload)?
            .plan(request, model, program)
    }
    fn postprocess(
        &self,
        request: &CanonicalRequest,
        outputs: &[ModelOutput],
    ) -> Result<WorkloadOutput> {
        self.select(&request.workload)?
            .postprocess(request, outputs)
    }
}
