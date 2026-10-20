//! Prompt-graph scratch shared by every sequence of one program.
//!
//! A sequence's prompt graphs need a full-width activation arena, NVFP4 split partials, the
//! chunked recurrent workspace and the draft fusion workspace: about 1 GB at 2048 lanes for the
//! target and as much again for the draft. Sequences replay their prompt graphs one at a time on
//! the device stream and read nothing from this scratch after their own replay, so one
//! allocation per prompt width serves every admitted sequence instead of one per sequence.
use super::ActivationArena;
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
