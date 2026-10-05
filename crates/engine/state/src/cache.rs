//! Hash-chain block index and deterministic LRU. Values own device/host state;
//! callers release their page leases when entries are evicted.
use infer_core::{Error, Result};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, collections::BTreeSet};

type Hash = [u8; 32];
struct Entry<V> {
    tokens: Vec<u32>,
    value: V,
    bytes: u64,
    touched: u64,
}
pub struct PrefixCache<V> {
    root: Hash,
    page_tokens: usize,
    budget: u64,
    max_entries: usize,
    entries: BTreeMap<Hash, Entry<V>>,
    lru: BTreeSet<(u64, Hash)>,
    clock: u64,
    bytes: u64,
    pub hits: u64,
    pub reused_tokens: u64,
    pub evictions: u64,
}
impl<V> PrefixCache<V> {
    ///
    /// # Errors
    /// Returns an invalid-input error for an empty namespace, zero page size, or zero entry limit.
    pub fn new(
        binding: &[u8],
        page_tokens: usize,
        budget: u64,
        max_entries: usize,
    ) -> Result<Self> {
        if binding.is_empty() || page_tokens == 0 || max_entries == 0 {
            return Err(Error::invalid("invalid prefix cache namespace/budget"));
        }
        let mut digest = Sha256::new();
        digest.update(b"kv-block-chain-v1");
        digest.update(binding);
        digest.update((page_tokens as u64).to_le_bytes());
        Ok(Self {
            root: digest.finalize().into(),
            page_tokens,
            budget,
            max_entries,
            entries: BTreeMap::new(),
            lru: BTreeSet::new(),
            clock: 0,
            bytes: 0,
            hits: 0,
            reused_tokens: 0,
            evictions: 0,
        })
    }
    fn key(&self, tokens: &[u32]) -> Hash {
        let mut parent = self.root;
        for block in tokens.chunks(self.page_tokens) {
            let mut hash = Sha256::new();
            hash.update(parent);
            for token in block {
                hash.update(token.to_le_bytes());
            }
            parent = hash.finalize().into();
        }
        parent
    }
    fn tick(&mut self) -> u64 {
        if self.clock == u64::MAX {
            self.lru.clear();
            let mut order: Vec<_> = self.entries.iter().map(|(k, e)| (e.touched, *k)).collect();
            order.sort_unstable();
            for (i, (_, key)) in order.into_iter().enumerate() {
                let t = i as u64;
                if let Some(entry) = self.entries.get_mut(&key) {
                    entry.touched = t;
                    self.lru.insert((t, key));
                }
            }
            self.clock = self.entries.len() as u64;
        }
        self.clock += 1;
        self.clock
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.values().map(|e| &e.value)
    }
    #[must_use]
    pub fn contains(&self, tokens: &[u32]) -> bool {
        self.entries
            .get(&self.key(tokens))
            .is_some_and(|e| e.tokens == tokens)
    }
    #[must_use]
    pub fn contains_where(&self, tokens: &[u32], compatible: impl Fn(&V) -> bool) -> bool {
        self.entries
            .get(&self.key(tokens))
            .is_some_and(|entry| entry.tokens == tokens && compatible(&entry.value))
    }
    #[must_use]
    pub fn matched_tokens(&self, tokens: &[u32], maximum: usize) -> usize {
        self.matched_tokens_where(tokens, maximum, |_| true)
    }
    pub fn matched_tokens_where(
        &self,
        tokens: &[u32],
        maximum: usize,
        compatible: impl Fn(&V) -> bool,
    ) -> usize {
        let mut parent = self.root;
        let mut matched = 0;
        for (i, block) in tokens.chunks_exact(self.page_tokens).enumerate() {
            let length = (i + 1) * self.page_tokens;
            if length > maximum {
                break;
            }
            let mut hash = Sha256::new();
            hash.update(parent);
            for t in block {
                hash.update(t.to_le_bytes());
            }
            parent = hash.finalize().into();
            if self
                .entries
                .get(&parent)
                .is_some_and(|e| e.tokens == tokens[..length] && compatible(&e.value))
            {
                matched = length;
            }
        }
        matched
    }
    pub fn lookup(&mut self, tokens: &[u32], maximum: usize) -> Option<V>
    where
        V: Clone,
    {
        self.lookup_where(tokens, maximum, |_| true)
    }
    pub fn lookup_where(
        &mut self,
        tokens: &[u32],
        maximum: usize,
        compatible: impl Fn(&V) -> bool,
    ) -> Option<V>
    where
        V: Clone,
    {
        let length = self.matched_tokens_where(tokens, maximum, compatible);
        if length == 0 {
            return None;
        }
        let key = self.key(&tokens[..length]);
        let tick = self.tick();
        let e = self.entries.get_mut(&key)?;
        self.lru.remove(&(e.touched, key));
        e.touched = tick;
        self.lru.insert((tick, key));
        self.hits = self.hits.saturating_add(1);
        self.reused_tokens = self.reused_tokens.saturating_add(length as u64);
        Some(e.value.clone())
    }
    pub fn evict_oldest(&mut self) -> Option<V> {
        let (_, key) = self.lru.pop_first()?;
        let e = self.entries.remove(&key)?;
        self.bytes -= e.bytes;
        self.evictions = self.evictions.saturating_add(1);
        Some(e.value)
    }
    ///
    /// # Errors
    /// Returns an invalid-input error for empty or partial token blocks, or an invariant error if eviction metadata is inconsistent.
    pub fn insert(&mut self, tokens: Vec<u32>, value: V, bytes: u64) -> Result<Vec<V>> {
        if tokens.is_empty() || !tokens.len().is_multiple_of(self.page_tokens) {
            return Err(Error::invalid(
                "only complete token blocks may be published",
            ));
        }
        let mut removed = vec![];
        if bytes > self.budget || self.budget == 0 {
            removed.push(value);
            return Ok(removed);
        }
        let key = self.key(&tokens);
        if let Some(old) = self.entries.remove(&key) {
            self.lru.remove(&(old.touched, key));
            self.bytes -= old.bytes;
            removed.push(old.value);
        }
        while self.entries.len() >= self.max_entries || self.bytes > self.budget - bytes {
            removed.push(
                self.evict_oldest().ok_or_else(|| {
                    Error::invariant("prefix cache eviction index is inconsistent")
                })?,
            );
        }
        let touched = self.tick();
        self.bytes += bytes;
        self.entries.insert(
            key,
            Entry {
                tokens,
                value,
                bytes,
                touched,
            },
        );
        self.lru.insert((touched, key));
        Ok(removed)
    }
}
