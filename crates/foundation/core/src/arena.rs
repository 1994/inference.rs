use crate::{Error, Result, StableId};
use std::marker::PhantomData;

struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

/// Slot generation prevents a stale handle from resolving to a newly inserted object.
/// Arena handles are local; externally visible request/model IDs use `IdAllocator`.
pub struct Arena<I, T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
    len: usize,
    bounded: bool,
    id: PhantomData<I>,
}
impl<I: StableId, T> Default for Arena<I, T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            len: 0,
            bounded: false,
            id: PhantomData,
        }
    }
}
impl<I: StableId, T> Arena<I, T> {
    /// Preallocate every slot and free-list entry. Insert/remove never grow this arena.
    /// # Errors
    /// Rejects invalid capacity or inability to allocate the fixed storage.
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity >= u32::MAX as usize {
            return Err(Error::invalid("invalid fixed arena capacity"));
        }
        let mut slots = Vec::new();
        let mut free = Vec::new();
        slots
            .try_reserve_exact(capacity)
            .map_err(|e| Error::new(crate::ErrorCode::Capacity, e.to_string()))?;
        free.try_reserve_exact(capacity)
            .map_err(|e| Error::new(crate::ErrorCode::Capacity, e.to_string()))?;
        slots.resize_with(capacity, || Slot {
            generation: 1,
            value: None,
        });
        for index in (0..capacity).rev() {
            free.push(u32::try_from(index).map_err(|_| Error::invalid("arena index overflow"))?);
        }
        Ok(Self {
            slots,
            free,
            len: 0,
            bounded: true,
            id: PhantomData,
        })
    }
    #[must_use]
    pub fn index(&self, id: I) -> Option<usize> {
        self.decode(id)
    }
    ///
    /// # Errors
    /// Returns an invariant error when the identity namespace is exhausted.
    pub fn insert(&mut self, value: T) -> Result<I> {
        let index = if let Some(index) = self.free.pop() {
            self.slots[index as usize].value = Some(value);
            index
        } else {
            if self.bounded {
                return Err(Error::new(
                    crate::ErrorCode::Capacity,
                    "fixed arena exhausted",
                ));
            }
            let index =
                u32::try_from(self.slots.len()).map_err(|_| Error::invariant("arena exhausted"))?;
            if index == u32::MAX {
                return Err(Error::invariant("arena exhausted"));
            }
            self.slots.push(Slot {
                generation: 1,
                value: Some(value),
            });
            index
        };
        self.len += 1;
        I::from_raw(
            (u64::from(self.slots[index as usize].generation) << 32) | (u64::from(index) + 1),
        )
    }
    fn decode(&self, id: I) -> Option<usize> {
        let raw = id.raw();
        let index = u32::try_from(raw & u64::from(u32::MAX))
            .ok()?
            .checked_sub(1)? as usize;
        let slot = self.slots.get(index)?;
        (u64::from(slot.generation) == raw >> 32 && slot.value.is_some()).then_some(index)
    }
    pub fn get(&self, id: I) -> Option<&T> {
        self.slots
            .get(self.decode(id)?)
            .and_then(|s| s.value.as_ref())
    }
    pub fn get_mut(&mut self, id: I) -> Option<&mut T> {
        let index = self.decode(id)?;
        self.slots[index].value.as_mut()
    }
    pub fn remove(&mut self, id: I) -> Option<T> {
        let index = self.decode(id)?;
        let free_index = u32::try_from(index).ok()?;
        let slot = &mut self.slots[index];
        let value = slot.value.take();
        // Retire exhausted slots rather than allowing an ABA collision.
        if let Some(generation) = slot.generation.checked_add(1) {
            slot.generation = generation;
            self.free.push(free_index);
        }
        self.len -= 1;
        value
    }
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequestId;
    #[test]
    fn stale_handle_cannot_alias_reused_slot() {
        let mut arena = Arena::<RequestId, _>::default();
        let old = arena.insert(12).unwrap();
        assert_eq!(arena.remove(old), Some(12));
        let new = arena.insert(24).unwrap();
        assert_ne!(old, new);
        assert_eq!(arena.get(old), None);
        assert_eq!(arena.get(new), Some(&24));
        assert_eq!(arena.get(RequestId::new(1).unwrap()), None);
    }
    #[test]
    fn fixed_slots_never_move_and_exhausted_generations_retire() -> Result<()> {
        let mut arena = Arena::<RequestId, _>::with_capacity(2)?;
        let storage = arena.slots.as_ptr();
        let free = arena.free.as_ptr();
        let first = arena.insert(1)?;
        let second = arena.insert(2)?;
        assert!(arena.insert(3).is_err());
        assert_eq!(arena.remove(first), Some(1));
        arena.slots[0].generation = u32::MAX;
        let last = arena.insert(3)?;
        assert_eq!(arena.remove(last), Some(3));
        assert!(arena.insert(4).is_err());
        assert_eq!(arena.get(first), None);
        assert_eq!(arena.get(second), Some(&2));
        assert_eq!(arena.slots.as_ptr(), storage);
        assert_eq!(arena.free.as_ptr(), free);
        Ok(())
    }
}
