//! Execution lifecycle.
use super::{HostBackend, HostTicket};
use infer_core::{Error, Result};
use infer_ir::{
    ExecutionProgram, ExecutionTask, ExecutionTiming, StepPlan, TaskOutput, TimingSource,
};
use infer_spi::BackendProvider;
use std::time::Instant;

impl HostBackend {
    pub(super) fn provider_submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: &[ExecutionTask],
    ) -> Result<HostTicket> {
        self.validate_program(&self.model, program)?;
        if step.program != program.id
            || tasks.len() != step.work.len()
            || tasks.iter().zip(&step.work).any(|(t, w)| {
                t.request != w.request
                    || t.state != w.state
                    || !self.sequences.get(&t.state).is_some_and(|s| {
                        s.tokens.len().checked_add(w.token_count) == Some(t.tokens.len())
                    })
            })
        {
            return Err(Error::invalid("host submission/step mismatch"));
        }
        let started = Instant::now();
        let result = tasks
            .iter()
            .map(|task| {
                Ok(TaskOutput {
                    request: task.request,
                    output: self.forward_task(program, step, task)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(HostTicket {
            result: Some(result),
            timing: ExecutionTiming {
                elapsed_us: (u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX))
                    .max(1),
                source: TimingSource::CpuWall,
            },
        })
    }
}
