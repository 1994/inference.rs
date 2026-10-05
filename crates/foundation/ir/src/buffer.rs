//! Shared result payloads keep host credits until the last consumer releases the storage.
use infer_core::{Error, ErrorCode, Result, credits::CreditLease};
use serde::{Deserialize, Serialize};
use std::{ops::Deref, sync::Arc};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SharedOutput<T>(Arc<Vec<T>>, #[serde(skip)] Option<CreditLease>);
impl<T> From<Vec<T>> for SharedOutput<T> {
    fn from(values: Vec<T>) -> Self {
        Self(Arc::new(values), None)
    }
}
impl<T> Deref for SharedOutput<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T: PartialEq> PartialEq for SharedOutput<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T: Eq> Eq for SharedOutput<T> {}
impl<T> SharedOutput<T> {
    /// # Errors
    /// Reports inability to allocate a fixed payload slot at startup.
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(capacity)
            .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
        Ok(values.into())
    }
    /// A published reader prevents overwrite; this never invokes copy-on-write allocation.
    pub fn replace_if_unique(&mut self, values: &[T]) -> bool
    where
        T: Clone,
    {
        let Some(storage) = Arc::get_mut(&mut self.0) else {
            return false;
        };
        if values.len() > storage.capacity() {
            return false;
        }
        storage.clear();
        storage.extend_from_slice(values);
        true
    }
    pub fn attach_credit(&mut self, credit: CreditLease) {
        self.1 = Some(credit);
    }
    #[must_use]
    pub fn shares_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<'a, T> IntoIterator for &'a SharedOutput<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
