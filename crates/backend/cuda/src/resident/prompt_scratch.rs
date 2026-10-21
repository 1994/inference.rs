//! Prompt-graph scratch shared by every sequence of one program.
//!
//! A sequence's prompt graphs need a full-width activation arena, NVFP4 split partials, the
//! chunked recurrent workspace and the draft fusion workspace: about 1 GB at 2048 lanes for the
//! target and as much again for the draft. Sequences replay their prompt graphs one at a time on
//! the device stream and read nothing from this scratch after their own replay, so one
//! allocation per prompt width serves every admitted sequence instead of one per sequence.
use super::ActivationArena;
use crate::device::CudaDevice;
use cutile::prelude::*;
use infer_core::{Error, Result};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// The scratch one prompt width's graphs capture against.
pub(super) struct PromptScratch {
    /// Master arena; each program captures through [`ActivationArena::shared_view`].
    pub(super) arena: ActivationArena,
    /// Final prompt row the one-row logits projection reads, when the vocabulary projection
    /// supports it.
    pub(super) row: Option<Tensor<f32>>,
    pub(super) partials: BTreeMap<usize, Tensor<f32>>,
    pub(super) recurrent: super::recurrent_prefill::Workspace,
    pub(super) fusion: Option<super::fusion_batch::Workspace>,
}

/// Shared prompt scratch of one program, keyed by prompt width.
#[derive(Default)]
pub struct Registry(Mutex<BTreeMap<usize, Arc<Mutex<PromptScratch>>>>);

impl Registry {
    /// Whether a width's storage is already physically allocated, including idle cache entries.
    pub(crate) fn is_allocated(&self, width: usize) -> Result<bool> {
        let entries = self
            .0
            .lock()
            .map_err(|_| Error::invariant("prompt scratch registry poisoned"))?;
        Ok(entries.contains_key(&width))
    }

    /// Release only entries no captured program owns, after draining detached replays.
    /// # Errors
    /// Reports a poisoned registry or failure to synchronize deferred device work.
    pub(crate) fn reclaim_unused(&self, device: &CudaDevice) -> Result<usize> {
        device.reclaim_barrier()?;
        let mut entries = self
            .0
            .lock()
            .map_err(|_| Error::invariant("prompt scratch registry poisoned"))?;
        let before = entries.len();
        entries.retain(|_, entry| Arc::strong_count(entry) > 1);
        let released = before - entries.len();
        drop(entries);
        if released != 0 {
            device.reclaim_barrier()?;
        }
        Ok(released)
    }

    /// The scratch for `width`, allocated by `allocate` on first use.
    /// # Errors
    /// Propagates allocation failures; a failed allocation is retried by the next caller.
    pub(super) fn get_or_allocate(
        &self,
        width: usize,
        allocate: impl FnOnce() -> Result<PromptScratch>,
    ) -> Result<Arc<Mutex<PromptScratch>>> {
        let mut entries = self
            .0
            .lock()
            .map_err(|_| Error::invariant("prompt scratch registry poisoned"))?;
        let scratch = if let Some(scratch) = entries.get(&width) {
            Arc::clone(scratch)
        } else {
            let scratch = Arc::new(Mutex::new(allocate()?));
            entries.insert(width, Arc::clone(&scratch));
            scratch
        };
        drop(entries);
        Ok(scratch)
    }
}
