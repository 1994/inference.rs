//! Execution responsibilities.
use super::{MetalBackend, MetalTicket, TicketWork};
use infer_core::{Error, ErrorCode, Result};
use infer_ir::{ExecutionProgram, ExecutionTask, StepPlan};
use infer_spi::BackendProvider;
use metal::MTLCommandBufferStatus;
use std::{collections::BTreeSet, sync::Arc, time::Instant};

impl MetalBackend {
    pub(super) fn prepare_submission(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: &[ExecutionTask],
    ) -> Result<()> {
        self.validate_program(&self.model, program)?;
        if self
            .transfers
            .iter()
            .any(|c| c.status() == MTLCommandBufferStatus::Error)
        {
            return Err(Error::new(ErrorCode::Backend, "KV fork transfer failed"));
        }
        if self.busy.is_some() {
            return Err(Error::new(
                ErrorCode::Conflict,
                "Metal scratch is in flight",
            ));
        }
        if step.program != program.id
            || tasks.is_empty()
            || tasks.iter().map(|t| t.state).collect::<BTreeSet<_>>().len() != tasks.len()
            || tasks
                .iter()
                .map(|t| t.request)
                .collect::<BTreeSet<_>>()
                .len()
                != tasks.len()
            || tasks.len() != step.work.len()
            || tasks.iter().zip(&step.work).any(|(t, w)| {
                t.request != w.request
                    || t.state != w.state
                    || w.token_count == 0
                    || w.role == infer_ir::ExecutionRole::Decode && w.token_count != 1
                    || step.role != infer_ir::ExecutionRole::Mixed && step.role != w.role
                    || !self.sequences.get(&t.state).is_some_and(|s| {
                        s.tokens.len().checked_add(w.token_count) == Some(t.tokens.len())
                            && t.tokens.len() <= s.capacity
                            && (t.tokens.readout() != infer_ir::OutputReadout::Full
                                || s.readout == infer_ir::OutputReadout::Full)
                            && t.tokens
                                .validate(&s.tokens, s.capacity, self.model.vocab_size)
                                .is_ok()
                    })
            })
        {
            return Err(Error::invalid("Metal submission/cursor mismatch"));
        }
        let required_pages = tasks.iter().try_fold(0usize, |n, t| {
            let growth = self
                .state_page_growth(t.state)?
                .ok_or_else(|| Error::invariant("Metal page growth"))?;
            n.checked_add(
                growth
                    .required_pages(t.tokens.len())
                    .ok_or_else(|| Error::invalid("batch page growth overflow"))?,
            )
            .ok_or_else(|| Error::invalid("batch page growth overflow"))
        })?;
        self.reclaim(required_pages)
    }
}

impl MetalBackend {
    pub(super) fn provider_submit_borrowed(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: &[ExecutionTask],
    ) -> Result<MetalTicket> {
        if tasks.len() > crate::constants::MAX_TICKET_TASKS {
            return Err(Error::invalid(
                "Metal batch exceeds ticket metadata capacity",
            ));
        }
        self.prepare_submission(program, step, tasks)?;
        objc::rc::autoreleasepool(|| {
            let command = self.gpu.queue.new_command_buffer().to_owned();
            command.set_label(&format!("infer-step-{}", step.id));
            let undo = self.begin_encoding(tasks)?;
            let encoded = match self.encode_batch(&command, program, step, tasks) {
                Ok(encoded) => encoded,
                Err(error) => {
                    self.rollback_encoding(undo)?;
                    return Err(error);
                }
            };
            self.finish_encoding(undo);
            for (task, start) in tasks.iter().zip(&encoded.starts) {
                self.tokens_executed += (task.tokens.len() - start) as u64;
                let sequence = self
                    .sequences
                    .get_mut(&task.state)
                    .ok_or_else(|| Error::invariant("validated submission sequence missing"))?;
                task.tokens.commit(&mut sequence.tokens)?;
                sequence.pending_prefix = None;
            }
            let mut metadata = std::mem::take(&mut self.ticket_work);
            metadata.clear();
            metadata.extend(tasks.iter().map(|task| TicketWork {
                request: task.request,
                state: task.state,
                readout: task.tokens.readout(),
            }));
            command.commit();
            self.busy = Some(step.id);
            self.inflight_states = tasks.iter().map(|t| t.state).collect();
            Ok(MetalTicket {
                owner: Arc::clone(&self.owner),
                command,
                step: step.id,
                tasks: metadata,
                starts: encoded.starts,
                submitted: Instant::now(),
                dispatches: encoded.dispatches,
                done: false,
                pending_cache: encoded.pending_cache,
            })
        })
    }
}
