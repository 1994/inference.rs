//! Immutable request preparation runs outside the scheduler owner.
mod memory;
use crate::RuntimeConfig;
use infer_core::{Error, ProgramId, Result};
use infer_ir::{CanonicalRequest, ModelIr, RequestInput, WorkloadPlan};
use infer_spi::WorkloadProvider;
use std::sync::Arc;

pub use memory::request_bytes;

pub struct PreparedRequest {
    pub(crate) request: Arc<CanonicalRequest>,
    pub(crate) plan: WorkloadPlan,
    pub(crate) tenant: Arc<str>,
    pub(crate) generated: infer_ir::TokenBuffer,
    pub(crate) model: Arc<ModelIr>,
    pub(crate) provider: String,
}
impl PreparedRequest {
    #[must_use]
    pub fn id(&self) -> infer_core::RequestId {
        self.request.id
    }
}
#[derive(Clone)]
pub struct RequestPreparer {
    model: Arc<ModelIr>,
    program: ProgramId,
    workloads: Arc<dyn WorkloadProvider + Send + Sync>,
    config: RuntimeConfig,
}
impl RequestPreparer {
    pub(crate) fn new(
        model: ModelIr,
        program: ProgramId,
        workloads: Box<dyn WorkloadProvider + Send + Sync>,
        config: RuntimeConfig,
    ) -> Self {
        Self {
            model: Arc::new(model),
            program,
            workloads: workloads.into(),
            config,
        }
    }
    /// # Errors
    /// Rejects invalid schema, input/plan budgets, unsupported heads or invalid tokens.
    pub fn prepare(&self, request: CanonicalRequest) -> Result<PreparedRequest> {
        request.validate()?;
        validate_input(&request.input, &self.config)?;
        let plan = self.workloads.plan(&request, &self.model, self.program)?;
        validate_plan(&request, &plan, &self.model, self.program, &self.config)?;
        let generated = generation_storage(&request)?;
        let tenant = Arc::from(request.qos.tenant.as_str());
        Ok(PreparedRequest {
            generated,
            tenant,
            request: request.into(),
            plan,
            model: self.model.clone(),
            provider: self.workloads.identity().into(),
        })
    }
}
pub fn validate_input(input: &RequestInput, config: &RuntimeConfig) -> Result<()> {
    let (units, tokens) = match input {
        RequestInput::Sequence { tokens, .. } => (1, tokens.len()),
        RequestInput::Pairs { query, documents } => (
            documents.len(),
            documents
                .iter()
                .try_fold(0usize, |sum, doc| {
                    sum.checked_add(query.len())
                        .and_then(|n| n.checked_add(doc.len()))
                })
                .ok_or_else(|| Error::invalid("pair token count overflow"))?,
        ),
    };
    if units > config.max_request_units || tokens > config.max_input_tokens {
        return Err(Error::new(
            infer_core::ErrorCode::Capacity,
            "request input exceeds admission budget",
        ));
    }
    Ok(())
}
pub fn validate_plan(
    request: &CanonicalRequest,
    plan: &WorkloadPlan,
    model: &ModelIr,
    program: ProgramId,
    config: &RuntimeConfig,
) -> Result<()> {
    let tokens = plan
        .units
        .iter()
        .try_fold(0usize, |sum, unit| sum.checked_add(unit.len()));
    if plan.request != request.id
        || plan.model != model.id
        || plan.program != program
        || plan.units.is_empty()
        || plan.units.len() > config.max_request_units
        || tokens.is_none_or(|count| count > config.max_input_tokens)
        || plan.reserved_tokens == 0
        || plan.reserved_tokens > model.max_sequence
        || plan.units.iter().any(|unit| {
            unit.is_empty()
                || unit.len() > plan.reserved_tokens
                || unit.iter().any(|token| *token as usize >= model.vocab_size)
        })
    {
        return Err(Error::invalid(
            "workload provider returned an invalid/budget-exceeding plan",
        ));
    }
    Ok(())
}

pub fn generation_storage(request: &CanonicalRequest) -> Result<infer_ir::TokenBuffer> {
    let capacity = if let infer_ir::Workload::Generate { max_new_tokens } = request.workload {
        max_new_tokens
    } else {
        0
    };
    infer_ir::TokenBuffer::with_capacity(capacity)
}

pub fn retained_tokens(plan: &WorkloadPlan, generated: usize) -> Result<usize> {
    plan.units
        .iter()
        .try_fold(generated, |sum, unit| {
            sum.checked_add(unit.len().saturating_mul(2))
        })
        .ok_or_else(|| Error::invalid("host token reservation overflow"))
}

pub const fn generation_capacity(request: &CanonicalRequest) -> usize {
    if let infer_ir::Workload::Generate { max_new_tokens } = request.workload {
        max_new_tokens
    } else {
        0
    }
}

/// Conservative peak reservation for request tokens, full readouts and retained projections.
/// # Errors
/// Rejects overflow before any backend reservation or native output allocation.
pub fn retained_bytes(
    request: &CanonicalRequest,
    plan: &WorkloadPlan,
    model: &ModelIr,
    page_tokens: usize,
) -> Result<usize> {
    let tokens = retained_tokens(plan, generation_capacity(request))?;
    let mut total = tokens
        .checked_mul(size_of::<u32>())
        .and_then(|bytes| bytes.checked_add(4096))
        .and_then(|bytes| bytes.checked_add(request_bytes(request)))
        .and_then(|bytes| bytes.checked_add(memory::projection_bytes(request)))
        .and_then(|bytes| {
            plan.reserved_tokens
                .div_ceil(page_tokens)
                .checked_mul(size_of::<infer_core::StatePageId>())
                .and_then(|pages| bytes.checked_add(pages))
        })
        .and_then(|bytes| {
            plan.units
                .capacity()
                .checked_mul(size_of::<infer_ir::TokenBuffer>())
                .and_then(|units| bytes.checked_add(units))
        })
        .ok_or_else(|| Error::invalid("host request bytes overflow"))?;
    // Forward readouts can retain all pair units. Include vector headers and both projection/input lifetimes.
    let generate = matches!(request.workload, infer_ir::Workload::Generate { .. });
    for unit in &plan.units {
        let rows = if generate { 0 } else { unit.len() };
        let row_bytes = model
            .hidden_size
            .checked_mul(size_of::<f32>())
            .and_then(|bytes| bytes.checked_add(size_of::<Vec<f32>>()))
            .ok_or_else(|| Error::invalid("host readout width overflow"))?;
        total = rows
            .checked_mul(row_bytes)
            .and_then(|bytes| {
                model
                    .vocab_size
                    .checked_mul(size_of::<f32>())
                    .and_then(|logits| bytes.checked_add(logits))
            })
            .and_then(|bytes| bytes.checked_mul(2))
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| Error::invalid("host readout bytes overflow"))?;
    }
    Ok(total)
}
