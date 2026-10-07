//! Prompt-prefix cache: device-resident KV snapshots keyed by the token prefix they cover.
//!
//! Matching policy is pure so it can be unit-tested without a device; the snapshots themselves
//! hold resident state tensors that later sequences restore with a device-to-device copy.
use super::prefix_state::{DraftSnapshot, ProgramSnapshot};
use std::collections::VecDeque;

/// One cached prompt prefix: the covered tokens plus copies of the sequence's state tensors.
pub struct CachedPrefix {
    /// Prompt tokens this snapshot covers, in order.
    pub tokens: Vec<u32>,
    /// Device copies of the state tensors at that boundary.
    pub target: ProgramSnapshot,
    /// Draft state and last hidden row are restored together with the target.
    pub draft: Option<DraftSnapshot>,
    /// Admission bytes of the snapshot, used for eviction.
    pub bytes: u64,
}

/// Bounded, deduplicated, least-recently-used prefix store.
pub struct PrefixCache {
    entries: VecDeque<CachedPrefix>,
    bytes: u64,
    capacity_bytes: u64,
    /// Successful restores, for observation.
    pub hits: u64,
    /// Match attempts, for observation.
    pub lookups: u64,
}

impl PrefixCache {
    #[must_use]
    pub const fn new(capacity_bytes: u64) -> Self {
        Self {
            entries: VecDeque::new(),
            bytes: 0,
            capacity_bytes,
            hits: 0,
            lookups: 0,
        }
    }

    /// Longest prefix of `tokens` already covered by a snapshot, capped at `maximum`.
    #[must_use]
    pub fn match_len(&self, tokens: &[u32], maximum: usize) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.tokens.len() <= maximum && tokens.starts_with(&entry.tokens))
            .map(|entry| entry.tokens.len())
            .max()
            .unwrap_or(0)
    }

    /// Take the best matching snapshot, keeping the rest in place.
    pub fn take_match(&mut self, tokens: &[u32], maximum: usize) -> Option<CachedPrefix> {
        self.lookups = self.lookups.saturating_add(1);
        let wanted = self.match_len(tokens, maximum);
        if wanted == 0 {
            return None;
        }
        let index = self
            .entries
            .iter()
            .position(|entry| entry.tokens.len() == wanted && tokens.starts_with(&entry.tokens))?;
        let entry = self.entries.remove(index)?;
        self.bytes -= entry.bytes;
        self.hits = self.hits.saturating_add(1);
        Some(entry)
    }

    /// Insert a snapshot, replacing an equal prefix and evicting least-recently-used entries
    /// until the store fits its capacity again.
    pub fn insert(&mut self, entry: CachedPrefix) {
        if self.capacity_bytes == 0 || entry.bytes > self.capacity_bytes {
            return;
        }
        if let Some(index) = self
            .entries
            .iter()
            .position(|cached| cached.tokens == entry.tokens)
            && let Some(previous) = self.entries.remove(index)
        {
            self.bytes -= previous.bytes;
        }
        self.entries.push_back(entry);
        self.bytes += self.entries.back().map_or(0, |entry| entry.bytes);
        while self.bytes > self.capacity_bytes {
            let Some(evicted) = self.entries.pop_front() else {
                break;
            };
            self.bytes -= evicted.bytes;
        }
    }

    /// Release the least recently used snapshot under sequence admission pressure.
    pub fn evict(&mut self) -> bool {
        let Some(entry) = self.entries.pop_front() else {
            return false;
        };
        self.bytes -= entry.bytes;
        true
    }

    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(tokens: &[u32], bytes: u64) -> CachedPrefix {
        CachedPrefix {
            tokens: tokens.to_vec(),
            target: ProgramSnapshot::default(),
            draft: None,
            bytes,
        }
    }

    #[test]
    fn restores_only_complete_snapshots_within_the_cap() {
        let mut cache = PrefixCache::new(1024);
        cache.insert(entry(&[1, 2, 3, 4], 10));
        cache.insert(entry(&[1, 2, 9], 10));
        // Recurrent state cannot be rolled back to an arbitrary partial match.
        assert_eq!(cache.match_len(&[1, 2, 3, 5], 4), 0);
        assert_eq!(cache.match_len(&[1, 2, 3, 4, 5], 4), 4);
        assert_eq!(cache.match_len(&[1, 2, 3, 4, 5], 2), 0);
        assert!(cache.take_match(&[1, 2, 3, 5], 4).is_none());
        assert_eq!(cache.match_len(&[7, 7], 4), 0);
    }

    #[test]
    fn take_match_reports_hits_and_keeps_the_store_usable() {
        let mut cache = PrefixCache::new(1024);
        cache.insert(entry(&[4, 5, 6], 10));
        let taken = cache.take_match(&[4, 5, 6, 7], 4).expect("prefix match");
        assert_eq!(taken.tokens, vec![4, 5, 6]);
        assert_eq!(cache.hits, 1);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.match_len(&[4, 5, 6, 7], 4), 0);
    }

    #[test]
    fn insert_deduplicates_and_evicts_by_capacity() {
        let mut cache = PrefixCache::new(20);
        cache.insert(entry(&[1], 10));
        cache.insert(entry(&[1], 10));
        assert_eq!(cache.len(), 1, "equal prefixes are stored once");
        cache.insert(entry(&[2], 10));
        assert_eq!(cache.len(), 2);
        cache.insert(entry(&[3], 10));
        assert_eq!(cache.len(), 2, "least recently used entry is evicted");
        assert_eq!(cache.bytes(), 20);
        // A snapshot larger than the whole store is not retained.
        cache.insert(entry(&[9], 21));
        assert_eq!(cache.len(), 2);
    }
}
