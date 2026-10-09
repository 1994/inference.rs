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
