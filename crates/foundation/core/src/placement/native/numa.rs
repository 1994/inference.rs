use super::mask::Mask;
use crate::{
    Error, Result,
    placement::{NumaPolicy, parse_cpu_list},
};
pub(super) struct Policy {
    mode: i32,
    mask: Mask,
}
impl Policy {
    pub fn current() -> Result<Self> {
        let possible = std::fs::read_to_string("/sys/devices/system/node/possible")
            .map_err(|error| Error::invalid(format!("read possible NUMA nodes: {error}")))?;
        let nodes = parse_cpu_list(&possible)?;
        let bits = nodes
            .last()
            .ok_or_else(|| Error::invalid("no possible NUMA node"))?
            + 1;
        let mut mask = Mask::new(bits)?;
        let mut mode = 0i32;
        // SAFETY: mode is a writable int and mask spans bits() aligned bits; null address/zero flags queries this thread.
        if unsafe {
            libc::syscall(
                libc::SYS_get_mempolicy,
                &raw mut mode,
                mask.pointer_mut(),
                // Linux's nodemask ABI subtracts one from maxnode before copying bits.
                // Include the highest bit of an exactly word-sized bitmap without requesting an extra word.
                mask.bits() + 1,
                std::ptr::null::<u8>(),
                0usize,
            )
        } != 0
        {
            return Err(super::linux::os_error("read NUMA policy"));
        }
        Ok(Self { mode, mask })
    }
    pub fn selected(&self, policy: NumaPolicy) -> Result<Self> {
        let (mode, node) = match policy {
            NumaPolicy::Bind(node) => (2, node),
            NumaPolicy::Prefer(node) => (1, node),
        };
        if !crate::placement::topology::allowed_nodes()?.contains(&node) {
            return Err(Error::invalid("NUMA node outside allowed memory set"));
        }
        let mut mask = Mask::new(self.mask.bits())?;
        mask.insert(node)?;
        Ok(Self { mode, mask })
    }
    pub fn apply(&self) -> Result<()> {
        let pointer = if self.mode == 0 {
            std::ptr::null()
        } else {
            self.mask.pointer()
        };
        // SAFETY: set_mempolicy reads the supplied aligned bitmap; MPOL_DEFAULT requires a null pointer.
        if unsafe {
            libc::syscall(
                libc::SYS_set_mempolicy,
                self.mode,
                pointer,
                self.mask.bits() + 1,
            )
        } == 0
        {
            Ok(())
        } else {
            Err(super::linux::os_error("set NUMA memory policy"))
        }
    }
}
#[cfg(test)]
#[path = "numa_tests.rs"]
mod memory_tests;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::ThreadPlacement;
    fn identity(policy: &Policy) -> (i32, Vec<usize>) {
        (
            policy.mode,
            (0..policy.mask.bits())
                .filter(|node| policy.mask.contains(*node))
                .collect(),
        )
    }
    #[test]
    #[ignore = "requires a Linux production host with get/set_mempolicy permitted; make check-linux-numa"]
    fn binding_preference_scope_failure_and_worker_restore() -> Result<()> {
        let node = *crate::placement::CpuTopology::discover()?
            .allowed_nodes
            .first()
            .ok_or_else(|| Error::invalid("no allowed NUMA node"))?;
        let before = identity(&Policy::current()?);
        for (mode, numa) in [(2, NumaPolicy::Bind(node)), (1, NumaPolicy::Prefer(node))] {
            let placement = ThreadPlacement {
                numa: Some(numa),
                ..ThreadPlacement::default()
            };
            placement.scope(|| {
                assert_eq!(identity(&Policy::current()?), (mode, vec![node]));
                Ok(())
            })?;
            assert_eq!(identity(&Policy::current()?), before);
            assert!(
                placement
                    .scope::<()>(|| Err(Error::invalid("initializer failed")))
                    .is_err()
            );
            assert_eq!(identity(&Policy::current()?), before);
            let (tx, rx) = std::sync::mpsc::channel();
            let thread = placement.spawn("infer-numa-test".into(), move |report| {
                let _ = tx.send((
                    report.numa,
                    Policy::current().map(|policy| identity(&policy)),
                ));
            })?;
            let (reported, actual) = rx
                .recv()
                .map_err(|error| Error::invalid(error.to_string()))?;
            assert_eq!(reported, Some(numa));
            assert_eq!(actual?, (mode, vec![node]));
            thread
                .join()
                .map_err(|_| Error::invariant("NUMA owner failed"))?;
        }
        assert_eq!(identity(&Policy::current()?), before);
        Ok(())
    }
}
