//! Device-neutral physical page ownership. A lease identifies an allocation,
//! not just a reusable device address; page tables expose the index to kernels.
use infer_core::{Error, ErrorCode, OwnerId, Result, new_owner_id};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BlockLease {
    pub owner: OwnerId,
    pub index: u32,
    pub generation: u64,
}
#[derive(Clone)]
struct Slot {
    generation: u64,
    references: usize,
}
pub struct BlockPool {
    owner: OwnerId,
    slots: Vec<Slot>,
    free: VecDeque<u32>,
    pub allocations: u64,
    pub releases: u64,
    pub cow_copies: u64,
    validation: Vec<u64>,
    validation_epoch: u64,
}
impl BlockPool {
    ///
    /// # Errors
    /// Returns an invalid-input error if the block count exceeds the device ABI.
    pub fn new(blocks: usize) -> Result<Self> {
        let block_count =
            u32::try_from(blocks).map_err(|_| Error::invalid("page pool exceeds device ABI"))?;
        if blocks >= u32::MAX as usize {
            return Err(Error::invalid("page pool exceeds device ABI"));
        }
        let owner = new_owner_id()?;
        Ok(Self {
            owner,
            slots: vec![
                Slot {
                    generation: 0,
                    references: 0
                };
                blocks
            ],
            free: (0..block_count).collect(),
            allocations: 0,
            releases: 0,
            cow_copies: 0,
            validation: vec![0; blocks],
            validation_epoch: 0,
        })
    }
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.slots.len()
    }
    #[must_use]
    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }
    #[must_use]
    pub fn reference_count(&self, index: u32) -> Option<usize> {
        self.slots.get(index as usize).map(|slot| slot.references)
    }
    ///
    /// # Errors
    /// Returns an invalid-input error for a stale, out-of-range, or unowned block lease.
    pub fn references(&self, lease: BlockLease) -> Result<usize> {
        if lease.owner != self.owner {
            return Err(Error::invalid("foreign physical page owner"));
        }
        let s = self
            .slots
            .get(lease.index as usize)
            .ok_or_else(|| Error::invalid("page address out of bounds"))?;
        if s.generation != lease.generation || s.references == 0 {
            return Err(Error::invalid("stale physical page lease"));
        }
        Ok(s.references)
    }
    ///
    /// # Errors
    /// Returns a capacity error for insufficient free blocks, or an invariant error for exhausted generations.
    pub fn allocate(&mut self, count: usize) -> Result<Vec<BlockLease>> {
        let mut output = Vec::new();
        output
            .try_reserve_exact(count)
            .map_err(|error| Error::new(ErrorCode::Capacity, error.to_string()))?;
        self.allocate_into(count, &mut output)?;
        Ok(output)
    }
    /// Append page leases into caller-owned storage without allocating or partially committing.
    /// # Errors
    /// Rejects insufficient scratch/free pages or exhausted generations before changing references.
    pub fn allocate_into(&mut self, count: usize, output: &mut Vec<BlockLease>) -> Result<()> {
        if count > output.capacity().saturating_sub(output.len()) {
            return Err(Error::new(
                ErrorCode::Capacity,
                "physical page scratch exhausted",
            ));
        }
        self.validate_allocation(count)?;
        for _ in 0..count {
            output.push(self.allocate_checked()?);
        }
        self.allocations = self.allocations.saturating_add(count as u64);
        Ok(())
    }
    fn validate_allocation(&self, count: usize) -> Result<()> {
        if count > self.free.len() {
            return Err(Error::new(
                ErrorCode::Capacity,
                "physical KV page pool exhausted",
            ));
        }
        if self
            .free
            .iter()
            .take(count)
            .any(|index| self.slots[*index as usize].generation == u64::MAX)
        {
            return Err(Error::new(
                ErrorCode::Capacity,
                "physical page generation exhausted",
            ));
        }
        Ok(())
    }
    fn allocate_checked(&mut self) -> Result<BlockLease> {
        let index = self
            .free
            .pop_front()
            .ok_or_else(|| Error::invariant("free-block count disagrees with queue"))?;
        let slot = &mut self.slots[index as usize];
        slot.generation += 1;
        slot.references = 1;
        Ok(BlockLease {
            owner: self.owner,
            index,
            generation: slot.generation,
        })
    }
    fn allocate_one(&mut self) -> Result<BlockLease> {
        self.validate_allocation(1)?;
        let lease = self.allocate_checked()?;
        self.allocations = self.allocations.saturating_add(1);
        Ok(lease)
    }
    ///
    /// # Errors
    /// Returns an invariant error for stale leases, duplicate ownership, or overflowing reference counts.
    pub fn retain(&mut self, leases: &[BlockLease]) -> Result<()> {
        self.validate_unique(leases)?;
        for l in leases {
            if self.references(*l)? == usize::MAX {
                return Err(Error::invalid("page reference overflow"));
            }
        }
        for l in leases {
            self.slots[l.index as usize].references += 1;
        }
        Ok(())
    }
    ///
    /// # Errors
    /// Returns an invariant error for stale leases or reference-count underflow.
    pub fn release(&mut self, leases: &[BlockLease]) -> Result<()> {
        self.validate_unique(leases)?;
        for l in leases {
            self.references(*l)?;
        }
        for l in leases {
            let s = &mut self.slots[l.index as usize];
            s.references -= 1;
            if s.references == 0 {
                self.free.push_back(l.index);
                self.releases = self.releases.saturating_add(1);
            }
        }
        Ok(())
    }
    ///
    /// # Errors
    /// Returns an invalid-input error for an invalid page size or table, or a capacity error if a shared tail cannot be copied.
    pub fn writable_tail(
        &mut self,
        table: &mut [BlockLease],
        rows: usize,
        page_rows: usize,
    ) -> Result<Option<(BlockLease, BlockLease)>> {
        if page_rows == 0 {
            return Err(Error::invalid("zero page size"));
        }
        if rows.is_multiple_of(page_rows) {
            return Ok(None);
        }
        let at = rows / page_rows;
        let old = *table
            .get(at)
            .ok_or_else(|| Error::invalid("tail page missing"))?;
        if self.references(old)? == 1 {
            return Ok(None);
        }
        let new = self.allocate_one()?;
        self.release(&[old])?;
        table[at] = new;
        self.cow_copies = self.cow_copies.saturating_add(1);
        Ok(Some((old, new)))
    }
    fn validate_unique(&mut self, leases: &[BlockLease]) -> Result<()> {
        if self.validation_epoch == u64::MAX {
            self.validation.fill(0);
            self.validation_epoch = 0;
        }
        self.validation_epoch += 1;
        for lease in leases {
            let mark = self
                .validation
                .get_mut(lease.index as usize)
                .ok_or_else(|| Error::invalid("page address out of bounds"))?;
            if *mark == self.validation_epoch {
                return Err(Error::invalid("aliased block table"));
            }
            *mark = self.validation_epoch;
        }
        Ok(())
    }
    ///
    /// # Errors
    /// Returns an invariant error if lease generations or recorded reference counts disagree with the supplied owners.
    pub fn check_owners<'a>(&self, owners: impl Iterator<Item = &'a [BlockLease]>) -> Result<()> {
        let mut counts = vec![0usize; self.capacity()];
        for table in owners {
            let mut unique = std::collections::BTreeSet::new();
            if table.iter().any(|lease| !unique.insert(lease.index)) {
                return Err(Error::invalid("aliased block owner table"));
            }
            for l in table {
                self.references(*l)?;
                counts[l.index as usize] += 1;
            }
        }
        let mut free = std::collections::BTreeSet::new();
        for i in &self.free {
            if *i as usize >= self.capacity() || !free.insert(*i) {
                return Err(Error::invariant("corrupt page free list"));
            }
        }
        for (i, s) in self.slots.iter().enumerate() {
            if s.references != counts[i]
                || (s.references == 0)
                    != free.contains(
                        &u32::try_from(i)
                            .map_err(|_| Error::invariant("pool index exceeds device ABI"))?,
                    )
            {
                return Err(Error::invariant(
                    "page references differ from active/cache owners",
                ));
            }
        }
        Ok(())
    }
}
