//! Batched external-hidden execution and attention state transfer.
use super::{BatchOutput, SlotCopy, SlotLane, SlotPool};
use crate::device::CudaDevice;
use infer_core::{Error, Result};

impl SlotPool {
    /// Discard an unaccepted attention-only draft suffix without copying KV data.
    pub(crate) fn rewind_external(&mut self, slot: usize, position: usize) -> Result<()> {
        if self
            .plan
            .iter()
            .any(|(_, copy)| matches!(copy, SlotCopy::Whole))
            || self
                .cursors()
                .get(slot)
                .is_none_or(|&cursor| position > cursor)
        {
            return Err(Error::invalid("draft rewind range or recurrent state"));
        }
        let graph = self
            .decode
            .as_mut()
            .filter(|graph| graph.external_hidden)
            .ok_or_else(|| Error::invalid("draft rewind requires external decode"))?;
        graph.set_cursor(slot, position)
    }

    /// Supply one external hidden row per active lane, retaining upload owners for every
    /// graph update (including inactive slots on later replays).
    pub(crate) fn run_external(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotLane],
        hidden: &[&[f32]],
    ) -> Result<BatchOutput> {
        self.run_external_readout(device, lanes, hidden, true)
    }

    pub(crate) fn run_external_readout(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotLane],
        hidden: &[&[f32]],
        read_logits: bool,
    ) -> Result<BatchOutput> {
        let graph = self
            .decode
            .as_ref()
            .ok_or_else(|| Error::invalid("external hidden requires a decode pool"))?;
        if !graph.external_hidden || lanes.len() != hidden.len() {
            return Err(Error::invalid("external hidden lane count or graph"));
        }
        for (&(slot, ..), values) in lanes.iter().zip(hidden) {
            let buffer = self
                .lane_external
                .get_mut(slot)
                .ok_or_else(|| Error::invalid("external hidden slot"))?;
            if values.len() != buffer.size() {
                return Err(Error::invalid("external hidden shape"));
            }
            let uploaded = device.upload(values.to_vec(), &[values.len()])?;
            graph.bind_external(buffer, &uploaded)?;
            self.external_uploads[slot] = Some(uploaded);
        }
        self.decode
            .as_mut()
            .ok_or_else(|| Error::invariant("external decode graph"))?
            .run_lanes(device, lanes, self.capacity, self.vocabulary, read_logits)
    }

    /// Copy a draft slot back to its private program before acceptance/rejection replay.
    /// Recurrent state requires checkpoints and is deliberately rejected here.
    pub(crate) fn export_attention(
        &self,
        slot: usize,
        target: &mut super::super::DeviceProgram,
        device: &CudaDevice,
    ) -> Result<()> {
        let rows = *self
            .cursors()
            .get(slot)
            .ok_or_else(|| Error::invalid("export slot"))?;
        let capacity = target.external_attention_capacity(rows)?;
        for (id, copy) in &self.plan {
            match copy {
                SlotCopy::Kv => Self::copy_kv_states(
                    target.states_mut(),
                    &self.states[slot],
                    *id,
                    rows,
                    capacity,
                    device,
                )?,
                SlotCopy::KvFp8 => Self::copy_kv_fp8(
                    target.fp8_states_mut(),
                    &self.fp8[slot],
                    *id,
                    rows,
                    capacity,
                    device,
                )?,
                SlotCopy::Whole => {
                    return Err(Error::unsupported(
                        "recurrent slot export requires checkpoints",
                    ));
                }
            }
        }
        target.adopt_attention_prefix(rows)
    }
}
