use crate::{Error, Result, StepId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaitReason {
    MediaReadiness,
    StateCapacity { required: usize, available: usize },
    BackendBusy,
    Transfer,
    Budget,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    Length,
    Eos,
    Completed,
    Cancelled,
    Deadline,
    Failed(String),
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestStatus {
    Runnable,
    Running { step: StepId },
    Waiting(WaitReason),
    Finished(FinishReason),
}
impl RequestStatus {
    #[must_use]
    pub const fn terminal(&self) -> bool {
        matches!(self, Self::Finished(_))
    }
    ///
    /// # Errors
    /// Returns an invariant error for an illegal request lifecycle transition.
    pub fn transition(&mut self, next: Self) -> Result<()> {
        if self.terminal() {
            return Err(Error::invariant("terminal request cannot transition"));
        }
        match (&*self, &next) {
            (Self::Running { .. }, Self::Running { .. }) => {
                Err(Error::invariant("request already running"))
            }
            (Self::Waiting(_), Self::Running { .. }) => {
                Err(Error::invariant("waiting request must become runnable"))
            }
            _ => {
                *self = next;
                Ok(())
            }
        }
    }
}
