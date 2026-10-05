//! Backend reference implementation.
use super::{ReferenceKernels, ReferenceModel};
use infer_core::{Error, Result};
use infer_ir::{
    BackendKind, DeviceCapabilities, ExecutionProgram, ExecutionTask, ModelIr, PrecisionPlan,
    StepPlan, TaskOutput,
};
use infer_spi::{BackendProvider, KernelProvider};
use sha2::{Digest, Sha256};

pub struct ReferenceBackend {
    pub(super) model: ReferenceModel,
    pub(super) identity: String,
    pub(super) sequences: std::collections::BTreeMap<infer_core::StateId, Vec<u32>>,
}
pub struct ReferenceTicket {
    pub(super) result: Option<Vec<TaskOutput>>,
}
impl ReferenceBackend {
    ///
    /// # Errors
    /// Returns an invalid-input error for invalid configuration, or a capacity error if the requested resources cannot be reserved.
    pub fn new(model: ReferenceModel) -> Result<Self> {
        model.validate()?;
        let bytes = serde_json::to_vec(&model).map_err(|e| Error::invalid(e.to_string()))?;
        let identity = format!("reference-f32-v1:{:x}", Sha256::digest(bytes));
        Ok(Self {
            model,
            identity,
            sequences: std::collections::BTreeMap::new(),
        })
    }
    #[must_use]
    pub const fn model(&self) -> &ReferenceModel {
        &self.model
    }
}
impl BackendProvider for ReferenceBackend {
    type Ticket = ReferenceTicket;
    fn reserve_state(&mut self, state: infer_core::StateId, _capacity: usize) -> Result<()> {
        if self.sequences.insert(state, vec![]).is_some() {
            return Err(Error::invalid("duplicate reference state"));
        }
        Ok(())
    }
    fn reset_state(&mut self, state: infer_core::StateId) -> Result<()> {
        self.sequences
            .get_mut(&state)
            .ok_or_else(|| Error::invalid("unknown reference state"))?
            .clear();
        Ok(())
    }
    fn release_state(&mut self, state: infer_core::StateId) -> Result<()> {
        self.sequences
            .remove(&state)
            .ok_or_else(|| Error::invalid("unknown reference state"))?;
        Ok(())
    }
    fn identity(&self) -> &str {
        &self.identity
    }
    fn supports_control_checkpoint(&self) -> bool {
        true
    }
    fn supports_recompute_preemption(&self) -> bool {
        true
    }
    fn capture_execution_state(&self) -> Result<Option<Vec<u8>>> {
        serde_json::to_vec(&self.sequences)
            .map(Some)
            .map_err(|e| Error::invalid(e.to_string()))
    }
    fn restore_execution_state(&mut self, state: Option<&[u8]>) -> Result<()> {
        let sequences: std::collections::BTreeMap<infer_core::StateId, Vec<u32>> = state
            .map_or_else(
                || Ok(std::collections::BTreeMap::new()),
                |bytes| serde_json::from_slice(bytes).map_err(|e| Error::invalid(e.to_string())),
            )?;
        if sequences.values().any(|tokens| {
            tokens.len() > self.model.ir.max_sequence
                || tokens
                    .iter()
                    .any(|t| *t as usize >= self.model.ir.vocab_size)
        }) {
            return Err(Error::invalid("invalid reference checkpoint"));
        }
        self.sequences = sequences;
        Ok(())
    }
    fn validate_state_ownership(
        &self,
        states: &[(infer_core::StateId, usize, usize)],
    ) -> Result<()> {
        if states.len() != self.sequences.len()
            || states.iter().any(|(id, capacity, position)| {
                self.sequences
                    .get(id)
                    .is_none_or(|tokens| tokens.len() != *position || tokens.len() > *capacity)
            })
        {
            return Err(Error::invariant("reference state ownership mismatch"));
        }
        Ok(())
    }
    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities::reference()
    }
    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()> {
        if model != &self.model.ir
            || program.precision != PrecisionPlan::f32()
            || program.model != model.id
            || program.backend != BackendKind::TestCpu
        {
            return Err(Error::unsupported(
                "reference backend/model/precision mismatch",
            ));
        }
        let ir = infer_compiler::lower(model, program.precision.clone())?;
        if program
            .operations
            .iter()
            .map(|op| &op.op)
            .ne(ir.operations.iter())
        {
            return Err(Error::invalid("program does not match model lowering"));
        }
        let kernels = ReferenceKernels.kernels();
        if program.operations.iter().any(|op| {
            !kernels
                .iter()
                .any(|k| k.id == op.kernel && k.operation == op.op.operation)
        }) {
            return Err(Error::unsupported(
                "non-reference kernel in reference program",
            ));
        }
        Ok(())
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<Self::Ticket> {
        self.validate_program(&self.model.ir, program)?;
        if step.program != program.id
            || tasks.len() != step.work.len()
            || tasks
                .iter()
                .zip(&step.work)
                .any(|(t, w)| t.request != w.request || t.state != w.state)
        {
            return Err(Error::invalid("submission does not match step"));
        }
        let result = tasks
            .into_iter()
            .map(|task| {
                // Direct diagnostic submits may not reserve state; runtime submits do.
                let history = self.sequences.entry(task.state).or_default();
                task.tokens.commit(history)?;
                let mut output = self.model.forward(history)?;
                if task.tokens.readout() != infer_ir::OutputReadout::Full {
                    output.hidden.clear();
                }
                if task.tokens.readout() == infer_ir::OutputReadout::None {
                    output.logits.clear();
                }
                Ok(TaskOutput {
                    request: task.request,
                    output,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ReferenceTicket {
            result: Some(result),
        })
    }
    fn poll(&mut self, ticket: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>> {
        ticket
            .result
            .take()
            .map(Some)
            .ok_or_else(|| Error::invariant("ticket completed more than once"))
    }
}
