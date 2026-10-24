//! Request-local, device-independent generation constraints.
use infer_core::{Error, Result};
use serde::de::Error as DeserializeError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex};

/// A matcher is advanced only after a token is selected from its current mask.
pub trait GenerationConstraint: Send {
    /// # Errors
    /// Reports grammar exhaustion or resource limits. Bits use the original vocabulary IDs.
    fn mask(&mut self) -> Result<&[u32]>;
    /// # Errors
    /// Rejects an illegal token or a failed matcher transition.
    fn commit(&mut self, token: u32) -> Result<()>;
}

struct State {
    matcher: Box<dyn GenerationConstraint>,
    position: usize,
}

/// Ownership moves between the request and its one outstanding CPU output job.
///
/// Unique ownership avoids locking; diagnostic clones use the guarded fallback.
/// Diagnostic serialization preserves a marker; restoring a live matcher is unsupported.
#[derive(Clone)]
pub struct ConstraintHandle {
    state: Arc<Mutex<State>>,
    vocabulary: usize,
    bytes: usize,
}
impl ConstraintHandle {
    #[must_use]
    pub fn new(matcher: Box<dyn GenerationConstraint>, vocabulary: usize, bytes: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                matcher,
                position: 0,
            })),
            vocabulary,
            bytes,
        }
    }
    #[must_use]
    pub const fn vocabulary(&self) -> usize {
        self.vocabulary
    }
    #[must_use]
    pub const fn retained_bytes(&self) -> usize {
        self.bytes
    }
    /// Compute a mask and atomically advance it with the selected token on the CPU worker.
    /// # Errors
    /// Rejects stale/replayed jobs, poisoned state, and matcher/sampler failures.
    pub fn select(
        &mut self,
        position: usize,
        sample: impl FnOnce(&[u32]) -> Result<u32>,
    ) -> Result<u32> {
        if let Some(state) = Arc::get_mut(&mut self.state) {
            let state = state
                .get_mut()
                .map_err(|_| Error::invariant("constraint state poisoned"))?;
            return state.select(position, sample);
        }
        self.state
            .lock()
            .map_err(|_| Error::invariant("constraint state poisoned"))?
            .select(position, sample)
    }
}
impl State {
    fn select(
        &mut self,
        position: usize,
        sample: impl FnOnce(&[u32]) -> Result<u32>,
    ) -> Result<u32> {
        let state = self;
        if position != state.position {
            return Err(Error::invariant("constraint token position mismatch"));
        }
        let token = sample(state.matcher.mask()?)?;
        state.matcher.commit(token)?;
        state.position += 1;
        Ok(token)
    }
}
impl fmt::Debug for ConstraintHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConstraintHandle")
            .field("vocabulary", &self.vocabulary)
            .finish_non_exhaustive()
    }
}
impl Serialize for ConstraintHandle {
    fn serialize<S: Serializer>(&self, serializer: S) -> StdResult<S::Ok, S::Error> {
        serializer.serialize_str("live_generation_constraint")
    }
}
impl<'de> Deserialize<'de> for ConstraintHandle {
    fn deserialize<D: Deserializer<'de>>(_deserializer: D) -> StdResult<Self, D::Error> {
        Err(DeserializeError::custom(
            "live generation constraints cannot be restored",
        ))
    }
}
