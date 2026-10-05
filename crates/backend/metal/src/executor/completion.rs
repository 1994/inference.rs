//! Completion lifecycle.
use super::{CommandTrace, MetalBackend, MetalDevice, MetalTicket};
use infer_core::{Error, ErrorCode, Result};
mod probes;
use infer_ir::{ModelOutput, TaskOutput};
use metal::MTLCommandBufferStatus;
use std::sync::Arc;

impl MetalBackend {
    pub(super) fn provider_poll(&mut self, t: &mut MetalTicket) -> Result<Option<Vec<TaskOutput>>> {
        if !Arc::ptr_eq(&self.owner, &t.owner) || t.done || self.busy != Some(t.step) {
            return Err(Error::invariant("Metal ticket ownership mismatch"));
        }
        match t.command.status() {
            MTLCommandBufferStatus::Completed => {}
            MTLCommandBufferStatus::Error => {
                self.release_encoding_pins()?;
                self.busy = None;
                self.inflight_states.clear();
                t.done = true;
                t.tasks.clear();
                self.ticket_work = std::mem::take(&mut t.tasks);
                return Err(Error::new(ErrorCode::Backend, "Metal GPU command failed"));
            }
            _ => return Ok(None),
        }
        self.release_encoding_pins()?;
        self.busy = None;
        self.inflight_states.clear();
        t.done = true;
        self.transfers.clear();
        self.record_command(t);
        let mut outputs = self
            .completion_pool
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(64));
        for (task, start) in t.tasks.iter().zip(&t.starts) {
            let s = self
                .sequences
                .get_mut(&task.state)
                .ok_or_else(|| Error::invariant("readback sequence missing"))?;
            let mut output = s.readback.take().unwrap_or_else(|| ModelOutput {
                logits: Vec::with_capacity(self.model.vocab_size),
                hidden: if task.readout == infer_ir::OutputReadout::Full {
                    (0..s.capacity)
                        .map(|_| vec![0.0; self.model.hidden_size])
                        .collect()
                } else {
                    Vec::new()
                },
            });
            let rows = if task.readout == infer_ir::OutputReadout::Full {
                s.tokens.len()
            } else {
                0
            };
            while output.hidden.len() > rows {
                if let Some(row) = output.hidden.pop() {
                    s.hidden_spares.push(row);
                }
            }
            for (index, row) in output.hidden.iter_mut().enumerate() {
                MetalDevice::read_into_idle(&s.hidden, index * self.model.hidden_size, row)?;
            }
            output.logits.resize(
                if task.readout == infer_ir::OutputReadout::None {
                    0
                } else {
                    self.model.vocab_size
                },
                0.0,
            );
            MetalDevice::read_into_idle(&s.logits, 0, &mut output.logits)?;
            probes::ProbeCapture {
                config: &self.config,
                model: &self.model,
                layers: &self.layer_outputs,
                samples: &mut self.probes,
                dropped: &mut self.probe_dropped,
            }
            .capture(t, task, s, *start)?;
            outputs.push(TaskOutput {
                request: task.request,
                output,
            });
        }

        for cached in t.pending_cache.drain(..) {
            self.kv.publish_prefix(cached)?;
        }
        t.tasks.clear();
        self.ticket_work = std::mem::take(&mut t.tasks);
        Ok(Some(outputs))
    }
    fn record_command(&mut self, t: &MetalTicket) {
        let (gpu_start_ns, gpu_duration_ns) = crate::device::command_timing(&t.command);
        if self.commands.len() == self.config.trace_capacity {
            self.commands.pop_front();
        }
        self.commands.push_back(CommandTrace {
            step: t.step,
            requests: t.tasks.iter().map(|task| task.request).collect(),
            gpu_start_ns,
            gpu_duration_ns,
            encoded_dispatches: t.dispatches,
            elapsed_wall_ns: u64::try_from(t.submitted.elapsed().as_nanos()).unwrap_or(u64::MAX),
        });
    }
}
