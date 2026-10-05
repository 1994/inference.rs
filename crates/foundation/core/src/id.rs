use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{fmt, num::NonZeroU64, sync::atomic::AtomicU64, sync::atomic::Ordering};

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

pub trait StableId: Copy + Eq {
    /// # Errors
    /// Returns an invalid-input error when `raw` is zero.
    fn from_raw(raw: u64) -> Result<Self>;
    fn raw(self) -> u64;
}

macro_rules! ids {
    ($($name:ident),+ $(,)?) => {$ (
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(NonZeroU64);
        impl $name {
            pub const ONE: Self = Self(NonZeroU64::MIN);
            /// # Errors
            /// Returns an invalid-input error when `raw` is zero.
            pub fn new(raw: u64) -> Result<Self> { Self::from_raw(raw) }
            #[must_use]
            pub const fn from_nonzero(raw: NonZeroU64) -> Self { Self(raw) }
            pub const fn get(self) -> u64 { self.0.get() }
        }
        impl StableId for $name {
            fn from_raw(raw: u64) -> Result<Self> {
                NonZeroU64::new(raw).map(Self).ok_or_else(|| Error::invalid("IDs must be nonzero"))
            }
            fn raw(self) -> u64 { self.get() }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { self.0.fmt(f) }
        }
    )+};
}
ids!(
    RequestId,
    OwnerId,
    RequestHandle,
    BatchHandle,
    SessionId,
    MediaId,
    ModelId,
    WorkloadId,
    DecisionId,
    StepId,
    StateId,
    StatePageId,
    ProgramId,
    OpId,
    KernelId,
    DeviceId,
    AllocationId,
    SnapshotId,
    ExperimentId,
    ProviderId,
    TensorId
);

/// Assign an ephemeral process-wide owner namespace; never reuse it after pool teardown.
/// # Errors
/// Rejects owner identity exhaustion without wrapping the shared counter.
pub fn new_owner_id() -> Result<OwnerId> {
    let mut owner = NEXT_OWNER.load(Ordering::Relaxed);
    loop {
        let next = owner
            .checked_add(1)
            .ok_or_else(|| Error::invariant("owner identities exhausted"))?;
        match NEXT_OWNER.compare_exchange_weak(owner, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return OwnerId::new(owner),
            Err(current) => owner = current,
        }
    }
}

/// Allocate IDs within an explicit namespace. Serialize the counter in checkpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdAllocator {
    next: u64,
}
impl Default for IdAllocator {
    fn default() -> Self {
        Self { next: 1 }
    }
}
impl IdAllocator {
    ///
    /// # Errors
    /// Returns an invariant error if the next allocated identity could collide with a live identity.
    pub fn validate_after(&self, largest_live_id: u64) -> Result<()> {
        if self.next <= largest_live_id {
            return Err(Error::invariant(
                "ID allocator would reuse an existing identity",
            ));
        }
        Ok(())
    }
    /// Reserve a contiguous identity transaction without constructing an intermediate vector.
    /// # Errors
    /// Rejects zero allocator cursors or exhaustion without advancing the counter.
    pub fn reserve_range(&mut self, count: usize) -> Result<std::ops::Range<u64>> {
        let count =
            u64::try_from(count).map_err(|_| Error::invariant("ID range exceeds namespace"))?;
        if self.next == 0 {
            return Err(Error::invariant("ID allocator cursor is zero"));
        }
        let end = self
            .next
            .checked_add(count)
            .ok_or_else(|| Error::invariant("ID space exhausted"))?;
        let range = self.next..end;
        self.next = end;
        Ok(range)
    }
    ///
    /// # Errors
    /// Returns an invariant error when the identity namespace is exhausted.
    pub fn allocate<I: StableId>(&mut self) -> Result<I> {
        let next = self
            .next
            .checked_add(1)
            .ok_or_else(|| Error::invariant("ID space exhausted"))?;
        let id = I::from_raw(self.next)?;
        self.next = next;
        Ok(id)
    }
}
