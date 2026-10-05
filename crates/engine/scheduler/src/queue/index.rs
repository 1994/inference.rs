//! Fixed node storage with ordered cursors. Rotations never allocate or leave tombstones.
use infer_core::{Error, ErrorCode, Result};
use serde::{Deserialize, Serialize};
use std::{hash::Hash, hash::Hasher, num::NonZeroUsize};

mod link;
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Node<K> {
    key: K,
    priority: u64,
    #[serde(with = "link")]
    parent: Option<NonZeroUsize>,
    #[serde(with = "link")]
    left: Option<NonZeroUsize>,
    #[serde(with = "link")]
    right: Option<NonZeroUsize>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Index<K> {
    nodes: Vec<Node<K>>,
    free: Vec<usize>,
    root: Option<usize>,
    len: usize,
}
impl<K: Copy + Ord + Hash> Index<K> {
    pub fn new(capacity: usize, seed: K) -> Result<Self> {
        let mut nodes = Vec::new();
        let mut free = Vec::new();
        nodes
            .try_reserve_exact(capacity)
            .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
        free.try_reserve_exact(capacity)
            .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
        nodes.resize_with(capacity, || Node {
            key: seed,
            priority: 0,
            parent: None,
            left: None,
            right: None,
        });
        free.extend((0..capacity).rev());
        Ok(Self {
            nodes,
            free,
            root: None,
            len: 0,
        })
    }
    fn tag(slot: Option<usize>) -> Option<NonZeroUsize> {
        slot.and_then(|slot| slot.checked_add(1))
            .and_then(NonZeroUsize::new)
    }
    fn untag(link: Option<NonZeroUsize>) -> Option<usize> {
        link.map(NonZeroUsize::get)
            .and_then(|index| index.checked_sub(1))
    }
    fn node(&self, slot: usize) -> &Node<K> {
        &self.nodes[slot]
    }
    fn node_mut(&mut self, slot: usize) -> &mut Node<K> {
        &mut self.nodes[slot]
    }
    pub const fn capacity(&self) -> usize {
        self.nodes.len()
    }
    pub const fn len(&self) -> usize {
        self.len
    }
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
    fn locate(&self, key: K) -> Option<usize> {
        let mut cursor = self.root;
        while let Some(slot) = cursor {
            let node = self.node(slot);
            match key.cmp(&node.key) {
                std::cmp::Ordering::Equal => return Some(slot),
                std::cmp::Ordering::Less => cursor = Self::untag(node.left),
                std::cmp::Ordering::Greater => cursor = Self::untag(node.right),
            }
        }
        None
    }
    pub fn position(&self, key: K) -> Option<usize> {
        self.locate(key)
    }
    /// The owner keeps the index immutable while a merge cursor is active.
    pub fn next_at(&self, slot: usize) -> Option<(usize, K)> {
        self.successor(slot).map(|next| (next, self.node(next).key))
    }
    fn replace_child(&mut self, parent: Option<usize>, old: usize, new: Option<usize>) {
        if let Some(parent) = parent {
            let node = self.node_mut(parent);
            if node.left == Self::tag(Some(old)) {
                node.left = Self::tag(new);
            } else {
                node.right = Self::tag(new);
            }
        } else {
            self.root = new;
        }
        if let Some(new) = new {
            self.node_mut(new).parent = Self::tag(parent);
        }
    }
    fn rotate_up(&mut self, child: usize) {
        let Some(parent) = Self::untag(self.node(child).parent) else {
            return;
        };
        let grand = Self::untag(self.node(parent).parent);
        let left = self.node(parent).left == Self::tag(Some(child));
        let middle = if left {
            self.node(child).right
        } else {
            self.node(child).left
        };
        self.replace_child(grand, parent, Some(child));
        if left {
            self.node_mut(parent).left = middle;
            self.node_mut(child).right = Self::tag(Some(parent));
        } else {
            self.node_mut(parent).right = middle;
            self.node_mut(child).left = Self::tag(Some(parent));
        }
        self.node_mut(parent).parent = Self::tag(Some(child));
        if let Some(middle) = Self::untag(middle) {
            self.node_mut(middle).parent = Self::tag(Some(parent));
        }
    }
    pub fn insert(&mut self, key: K) -> bool {
        let mut parent = None;
        let mut cursor = self.root;
        while let Some(slot) = cursor {
            let node = self.node(slot);
            if node.key == key {
                return false;
            }
            parent = cursor;
            cursor = if key < node.key {
                Self::untag(node.left)
            } else {
                Self::untag(node.right)
            };
        }
        let Some(slot) = self.free.pop() else {
            return false;
        };
        let mut hasher = ahash::AHasher::default();
        key.hash(&mut hasher);
        self.nodes[slot] = Node {
            key,
            priority: hasher.finish(),
            parent: Self::tag(parent),
            left: None,
            right: None,
        };
        if let Some(parent) = parent {
            if key < self.node(parent).key {
                self.node_mut(parent).left = Self::tag(Some(slot));
            } else {
                self.node_mut(parent).right = Self::tag(Some(slot));
            }
        } else {
            self.root = Some(slot);
        }
        while let Some(parent) = Self::untag(self.node(slot).parent) {
            if self.node(parent).priority <= self.node(slot).priority {
                break;
            }
            self.rotate_up(slot);
        }
        self.len += 1;
        true
    }
    pub fn remove(&mut self, key: &K) -> bool {
        let Some(slot) = self.locate(*key) else {
            return false;
        };
        loop {
            let node = self.node(slot);
            let child = match (Self::untag(node.left), Self::untag(node.right)) {
                (Some(left), Some(right)) => {
                    Some(if self.node(left).priority <= self.node(right).priority {
                        left
                    } else {
                        right
                    })
                }
                (left, right) => left.or(right),
            };
            let Some(child) = child else {
                break;
            };
            self.rotate_up(child);
        }
        self.replace_child(Self::untag(self.node(slot).parent), slot, None);
        self.node_mut(slot).parent = None;
        self.free.push(slot);
        self.len -= 1;
        true
    }
    fn leftmost(&self, mut slot: usize) -> usize {
        while let Some(left) = Self::untag(self.node(slot).left) {
            slot = left;
        }
        slot
    }
    fn successor(&self, slot: usize) -> Option<usize> {
        if let Some(right) = Self::untag(self.node(slot).right) {
            return Some(self.leftmost(right));
        }
        let mut child = slot;
        while let Some(parent) = Self::untag(self.node(child).parent) {
            if self.node(parent).left == Self::tag(Some(child)) {
                return Some(parent);
            }
            child = parent;
        }
        None
    }
    pub fn validate(&self) -> Result<()> {
        if self.len > self.nodes.len() || self.free.len() != self.nodes.len() - self.len {
            return Err(Error::invariant("ordered index size corrupt"));
        }
        let mut seen = vec![false; self.nodes.len()];
        let mut pending = Vec::with_capacity(self.len);
        if let Some(root) = self.root {
            if self
                .nodes
                .get(root)
                .is_none_or(|node| node.parent.is_some())
            {
                return Err(Error::invariant("ordered index root corrupt"));
            }
            pending.push((root, None, None));
        }
        let mut visited = 0;
        while let Some((slot, lower, upper)) = pending.pop() {
            let node = self
                .nodes
                .get(slot)
                .ok_or_else(|| Error::invariant("ordered index slot out of bounds"))?;
            if seen[slot]
                || lower.is_some_and(|key| node.key <= key)
                || upper.is_some_and(|key| node.key >= key)
            {
                return Err(Error::invariant("ordered index cycle or key order corrupt"));
            }
            seen[slot] = true;
            visited += 1;
            for (child, lo, hi) in [
                (Self::untag(node.left), lower, Some(node.key)),
                (Self::untag(node.right), Some(node.key), upper),
            ] {
                if let Some(child) = child {
                    if self.nodes.get(child).is_none_or(|c| {
                        c.parent != Self::tag(Some(slot)) || c.priority < node.priority
                    }) {
                        return Err(Error::invariant("ordered index child corrupt"));
                    }
                    pending.push((child, lo, hi));
                }
            }
        }
        for slot in &self.free {
            if seen.get(*slot).is_none_or(|seen| *seen) {
                return Err(Error::invariant("ordered index free list corrupt"));
            }
            seen[*slot] = true;
        }
        if visited != self.len || seen.contains(&false) {
            return Err(Error::invariant("ordered index unreachable node"));
        }
        Ok(())
    }
    pub fn first(&self) -> Option<K> {
        self.root.map(|slot| self.node(self.leftmost(slot)).key)
    }
    pub fn iter(&self) -> Cursor<'_, K> {
        Cursor {
            index: self,
            slot: self.root.map(|r| self.leftmost(r)),
        }
    }
    pub fn from(&self, key: K) -> Cursor<'_, K> {
        let mut cursor = self.root;
        let mut found = None;
        while let Some(slot) = cursor {
            let node = self.node(slot);
            if node.key >= key {
                found = cursor;
                cursor = Self::untag(node.left);
            } else {
                cursor = Self::untag(node.right);
            }
        }
        Cursor {
            index: self,
            slot: found,
        }
    }
}
pub(super) struct Cursor<'a, K> {
    index: &'a Index<K>,
    slot: Option<usize>,
}
impl<K: Copy + Ord + Hash> Iterator for Cursor<'_, K> {
    type Item = K;
    fn next(&mut self) -> Option<K> {
        let slot = self.slot?;
        self.slot = self.index.successor(slot);
        Some(self.index.node(slot).key)
    }
}
impl<K: Copy + Ord + Hash> PartialEq for Index<K> {
    fn eq(&self, other: &Self) -> bool {
        self.nodes.len() == other.nodes.len()
            && self.len == other.len
            && self.iter().eq(other.iter())
    }
}
impl<K: Copy + Ord + Hash> Eq for Index<K> {}
