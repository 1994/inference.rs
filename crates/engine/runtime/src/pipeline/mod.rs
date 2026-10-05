//! The inference loop: settle termination, collect execution, plan, dispatch.
pub mod admission;
pub mod completion;
pub mod dispatch;
pub mod lifecycle;
pub mod scheduling;
use crate::{Engine, EngineOutput};
use infer_core::{Error, Result};
use infer_observe::DiagnosticCode;
use infer_spi::{BackendProvider, SchedulingPolicy};

#[derive(Clone, Copy)]
enum DriveMode {
    Schedule,
    Quiesce,
    Poll,
}

impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    /// Collect completions/deadlines without launching a new batch.
    /// # Errors
    /// Rejects a backwards clock or invalid completion while retaining pending tickets.
    pub fn poll_completed(&mut self, now_us: u64) -> Result<Vec<EngineOutput>> {
        let mut emitted = Vec::new();
        self.poll_completed_into(now_us, &mut emitted)?;
        Ok(emitted)
    }

    /// Advance termination, execution completion and the next scheduling decision.
    ///
    /// # Errors
    /// Returns invalid input for a backwards clock, or an execution/planning error.
    pub fn tick(&mut self, now_us: u64) -> Result<Vec<EngineOutput>> {
        let mut emitted = Vec::new();
        self.tick_into(now_us, &mut emitted)?;
        Ok(emitted)
    }
    /// Reuse caller-owned delivery storage. Appends outputs without clearing pending caller work.
    /// # Errors
    /// Returns clock, lifecycle, planner or backend errors while retaining published ownership.
    pub fn tick_into(&mut self, now_us: u64, emitted: &mut Vec<EngineOutput>) -> Result<()> {
        self.validate_clock(now_us)?;
        let mode = if self.fault.is_none() {
            DriveMode::Schedule
        } else {
            DriveMode::Quiesce
        };
        self.drive_into(now_us, mode, emitted)
            .inspect_err(|error| self.isolate(DiagnosticCode::InvariantViolation, error))
    }
    /// Poll without dispatch using reusable output storage.
    /// # Errors
    /// Rejects backwards time or invalid completion while retaining pending tickets.
    pub fn poll_completed_into(
        &mut self,
        now_us: u64,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<()> {
        self.validate_clock(now_us)?;
        self.drive_into(now_us, DriveMode::Poll, emitted)
            .inspect_err(|error| self.isolate(DiagnosticCode::InvariantViolation, error))
    }

    /// Drain execution before capturing a checkpoint; native tickets are never saved.
    ///
    /// # Errors
    /// Returns an execution error, a backwards-clock error or a checkpoint error.
    pub fn quiesce(
        &mut self,
        now_us: u64,
    ) -> Result<(Vec<EngineOutput>, Option<crate::RuntimeSnapshot>)> {
        self.validate_clock(now_us)?;
        let mut events = Vec::new();
        self.drive_into(now_us, DriveMode::Quiesce, &mut events)
            .inspect_err(|error| self.isolate(DiagnosticCode::InvariantViolation, error))?;
        let checkpoint = if self.fault.is_none()
            && self.inflight.is_none()
            && self.resources_pending.is_empty()
            && self.output_pending.is_empty()
        {
            let checkpoint = if self.backend.requires_async_checkpoint() {
                self.poll_checkpoint()?
            } else {
                Some(self.snapshot()?)
            };
            if checkpoint.is_none() {
                return Ok((events, None));
            }
            self.event(
                infer_core::event::EventKind::CheckpointCaptured,
                infer_core::event::ObjectKind::Program,
                self.program.id.get(),
                0,
                self.global_progress_epoch,
                0,
            );
            checkpoint
        } else {
            None
        };
        Ok((events, checkpoint))
    }

    fn validate_clock(&self, now_us: u64) -> Result<()> {
        if now_us < self.now_us {
            return Err(Error::invalid("runtime clock must be monotonic"));
        }
        Ok(())
    }

    fn drive_into(
        &mut self,
        now_us: u64,
        mode: DriveMode,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<()> {
        let start = std::time::Instant::now();
        let initial = emitted.len();
        let result = self.drive_owner(now_us, mode, emitted);
        self.cpu_stage(
            infer_core::event::CpuStage::Owner,
            0,
            start,
            emitted.len() - initial,
        );
        result
    }
    fn drive_owner(
        &mut self,
        now_us: u64,
        mode: DriveMode,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<()> {
        self.now_us = now_us;
        if matches!(mode, DriveMode::Schedule) {
            self.host.checkpoint = None;
        }
        if !matches!(mode, DriveMode::Schedule) {
            self.discard_prepared();
        }
        self.backend.maintenance()?;
        if self.backend_resource_epoch != self.backend.resource_epoch() {
            self.backend_resource_epoch = self.backend.resource_epoch();
            self.resources_changed()?;
        }
        self.wake_memory_waiters()?;
        if self.fault.is_none() {
            self.apply_cost_feedback()?;
        }
        self.record_action(match mode {
            DriveMode::Schedule => crate::ReplayAction::Tick { now_us },
            DriveMode::Quiesce => crate::ReplayAction::Quiesce { now_us },
            DriveMode::Poll => crate::ReplayAction::Poll { now_us },
        });

        self.poll_output_stages(emitted)?;
        self.poll_resources(emitted)?;
        self.drain_quarantined_requests(emitted)?;
        self.expire_requests(emitted)?;
        if self.poll_execution(emitted)?.is_break() {
            if matches!(mode, DriveMode::Schedule) && self.fault.is_none() {
                self.prepare_next()?;
            }
            return Ok(());
        }
        if matches!(mode, DriveMode::Schedule) && self.fault.is_none() {
            self.schedule_next(emitted)?;
        }
        if cfg!(debug_assertions) {
            self.check_invariants()?;
        }
        Ok(())
    }
}
