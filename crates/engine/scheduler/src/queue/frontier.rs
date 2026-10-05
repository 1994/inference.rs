//! Transient tenant merge heap. Persistent lifecycle indexes remain ordered trees.
use super::{FairKey, index::Index};
use infer_core::{Error, ErrorCode, RequestId, Result};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{cmp::Reverse, collections::BinaryHeap};

#[derive(Debug)]
pub(super) struct Frontier {
    items: BinaryHeap<Reverse<(FairKey, usize)>>,
    capacity: usize,
}
impl Clone for Frontier {
    fn clone(&self) -> Self {
        let mut items = BinaryHeap::with_capacity(self.capacity);
        items.extend(self.items.iter().copied());
        Self {
            items,
            capacity: self.capacity,
        }
    }
}
impl Frontier {
    pub fn new(capacity: usize) -> Result<Self> {
        let mut items = BinaryHeap::new();
        items
            .try_reserve_exact(capacity)
            .map_err(|error| Error::new(ErrorCode::Capacity, error.to_string()))?;
        Ok(Self { items, capacity })
    }
    pub fn first(&self) -> Option<(FairKey, usize)> {
        self.items.peek().map(|Reverse(key)| *key)
    }
    pub fn insert(&mut self, key: (FairKey, usize)) -> Result<()> {
        if self.items.len() == self.capacity {
            return Err(Error::invariant("tenant merge scratch exhausted"));
        }
        self.items.push(Reverse(key));
        Ok(())
    }
    pub fn pop(&mut self) -> Option<(FairKey, usize)> {
        self.items.pop().map(|Reverse(key)| key)
    }
    pub fn clear(&mut self) {
        self.items.clear();
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    pub fn validate(&self) -> Result<()> {
        if !self.items.is_empty() || self.items.capacity() < self.capacity {
            return Err(Error::invariant("tenant merge scratch is not reusable"));
        }
        Ok(())
    }
}
impl PartialEq for Frontier {
    fn eq(&self, other: &Self) -> bool {
        self.capacity == other.capacity && self.items.as_slice() == other.items.as_slice()
    }
}
impl Eq for Frontier {}
// Preserve schema 6's empty Index representation. Scratch never crosses an owner boundary.
impl Serialize for Frontier {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.validate().map_err(serde::ser::Error::custom)?;
        Index::new(self.capacity, (0, 0, RequestId::ONE, 0))
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for Frontier {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let index = Index::<FairKey>::deserialize(deserializer)?;
        index.validate().map_err(serde::de::Error::custom)?;
        if !index.is_empty() {
            return Err(serde::de::Error::custom(
                "checkpoint contains active merge scratch",
            ));
        }
        Self::new(index.capacity()).map_err(serde::de::Error::custom)
    }
}
