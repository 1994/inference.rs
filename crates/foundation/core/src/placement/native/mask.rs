//! Dynamically sized Linux bitmaps; never truncate a large host to `cpu_set_t`'s 1024 bits.
use crate::{Error, Result};
#[derive(Clone)]
pub(super) struct Mask(Vec<libc::c_ulong>);
impl Mask {
    pub fn new(bits: usize) -> Result<Self> {
        if bits == 0 || bits > crate::constants::MAX_TOPOLOGY_CPUS {
            return Err(Error::invalid("OS bitmap size outside topology limits"));
        }
        Ok(Self(vec![0; bits.div_ceil(libc::c_ulong::BITS as usize)]))
    }
    pub const fn bits(&self) -> usize {
        self.0.len() * libc::c_ulong::BITS as usize
    }
    pub const fn bytes(&self) -> usize {
        self.0.len() * size_of::<libc::c_ulong>()
    }
    pub const fn pointer(&self) -> *const libc::c_ulong {
        self.0.as_ptr()
    }
    pub const fn pointer_mut(&mut self) -> *mut libc::c_ulong {
        self.0.as_mut_ptr()
    }
    pub fn contains(&self, bit: usize) -> bool {
        let width = libc::c_ulong::BITS as usize;
        self.0
            .get(bit / width)
            .is_some_and(|word| word & (1 << (bit % width)) != 0)
    }
    pub fn insert(&mut self, bit: usize) -> Result<()> {
        let width = libc::c_ulong::BITS as usize;
        let word = self
            .0
            .get_mut(bit / width)
            .ok_or_else(|| Error::invalid("index outside OS bitmap"))?;
        *word |= 1 << (bit % width);
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn large_masks_preserve_word_boundaries_and_reject_overflow() -> Result<()> {
        let mut mask = Mask::new(8193)?;
        for bit in [0, 63, 64, 1023, 1024, 8192] {
            mask.insert(bit)?;
            assert!(mask.contains(bit));
        }
        assert!(!mask.contains(8191));
        assert!(!mask.contains(mask.bits()));
        assert!(mask.insert(mask.bits()).is_err());
        assert_eq!(mask.bytes(), mask.bits() / 8);
        Ok(())
    }
}
