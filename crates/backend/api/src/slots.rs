//! Slots contract.
use infer_core::{Error, Result, StepId};

/// Buffer slots cannot be recycled until their matching step completes.
pub struct SubmissionSlots {
    slots: Vec<Option<StepId>>,
    cursor: usize,
}
impl SubmissionSlots {
    ///
    /// # Errors
    /// Returns an invalid-input error for a zero ring capacity or a capacity exceeding the device ABI.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity > u32::MAX as usize {
            return Err(Error::invalid("invalid metadata ring capacity"));
        }
        Ok(Self {
            slots: vec![None; capacity],
            cursor: 0,
        })
    }
    ///
    /// # Errors
    /// Returns an invalid-input or capacity error for duplicate step identities or an exhausted metadata ring.
    pub fn acquire(&mut self, step: StepId) -> Result<u32> {
        if self.slots.contains(&Some(step)) {
            return Err(Error::invariant("step already owns a metadata slot"));
        }
        for offset in 0..self.slots.len() {
            let index = (self.cursor + offset) % self.slots.len();
            if self.slots[index].is_none() {
                let slot = u32::try_from(index)
                    .map_err(|_| Error::invariant("metadata slot exceeds device ABI"))?;
                self.slots[index] = Some(step);
                self.cursor = (index + 1) % self.slots.len();
                return Ok(slot);
            }
        }
        Err(Error::new(
            infer_core::ErrorCode::Capacity,
            "all metadata slots are in flight",
        ))
    }
    ///
    /// # Errors
    /// Returns a not-found or invariant error for unknown identities or inconsistent ownership.
    pub fn release(&mut self, slot: u32, step: StepId) -> Result<()> {
        let owner = self
            .slots
            .get_mut(slot as usize)
            .ok_or_else(|| Error::invalid("invalid metadata slot"))?;
        if *owner != Some(step) {
            return Err(Error::invariant("completion does not own metadata slot"));
        }
        *owner = None;
        Ok(())
    }
}
