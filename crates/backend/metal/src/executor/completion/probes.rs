use super::super::{MetalConfig, MetalDevice, MetalTicket, Sequence, TicketWork};
use infer_core::Result;
use infer_ir::{LayerProbe, ModelIr};
use std::collections::{BTreeMap, VecDeque};
pub(super) struct ProbeCapture<'a> {
    pub config: &'a MetalConfig,
    pub model: &'a ModelIr,
    pub layers: &'a BTreeMap<infer_core::OpId, usize>,
    pub samples: &'a mut VecDeque<LayerProbe>,
    pub dropped: &'a mut u64,
}
impl ProbeCapture<'_> {
    pub fn capture(
        &mut self,
        ticket: &MetalTicket,
        task: &TicketWork,
        sequence: &Sequence,
        start: usize,
    ) -> Result<()> {
        if let Some(buffer) = &sequence.probes {
            let all = MetalDevice::read_idle(
                buffer,
                sequence.tokens.len() * self.model.mixers.len() * self.model.hidden_size,
            )?;
            let sample_bytes = self.model.hidden_size as u64 * 4 + size_of::<LayerProbe>() as u64;
            let capacity = self.config.probe_bytes / sample_bytes;
            if capacity > 0 {
                for position in start..sequence.tokens.len() {
                    for (op, layer) in self.layers {
                        let offset =
                            (position * self.model.mixers.len() + layer) * self.model.hidden_size;
                        if self.samples.len() as u64 == capacity {
                            self.samples.pop_front();
                            *self.dropped += 1;
                        }
                        self.samples.push_back(LayerProbe {
                            request: task.request,
                            step: ticket.step,
                            state: task.state,
                            op: *op,
                            layer: *layer,
                            position,
                            hidden: all[offset..offset + self.model.hidden_size].to_vec(),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}
