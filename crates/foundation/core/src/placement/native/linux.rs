use super::super::{PlacementReport, ThreadPlacement, ThreadQos};
pub(in crate::placement) use super::affinity::allowed as allowed_cpus;
use super::{affinity, mask::Mask, numa::Policy};
use crate::{Error, Result};
pub(in crate::placement) struct Guard {
    old_cpus: Option<Mask>,
    old_numa: Option<Policy>,
    retained: bool,
    placement: ThreadPlacement,
}
impl Guard {
    pub fn enter(placement: &ThreadPlacement) -> Result<Self> {
        if placement.qos != ThreadQos::Inherit || placement.affinity_tag.is_some() {
            return Err(Error::unsupported(
                "Darwin QoS/affinity tags are unavailable on Linux",
            ));
        }
        let mut guard = Self {
            old_cpus: None,
            old_numa: None,
            retained: false,
            placement: placement.clone(),
        };
        if !placement.cpus.is_empty() {
            let old = affinity::current()?;
            let mut selected = Mask::new(old.bits())?;
            for cpu in &placement.cpus {
                if !old.contains(*cpu) {
                    return Err(Error::invalid("CPU outside inherited cpuset"));
                }
                selected.insert(*cpu)?;
            }
            guard.old_cpus = Some(old);
            affinity::set(&selected)?;
        }
        if let Some(policy) = placement.numa {
            let old = Policy::current()?;
            let selected = old.selected(policy)?;
            guard.old_numa = Some(old);
            selected.apply()?;
        }
        Ok(guard)
    }
    pub fn retain(mut self) -> Result<PlacementReport> {
        let report = self.placement.report()?;
        self.retained = true;
        Ok(report)
    }
    pub fn restore(&mut self) -> Result<()> {
        let mut failure = None;
        if let Some(policy) = &self.old_numa {
            match policy.apply() {
                Ok(()) => self.old_numa = None,
                Err(error) => failure = Some(error),
            }
        }
        if let Some(mask) = &self.old_cpus {
            match affinity::set(mask) {
                Ok(()) => self.old_cpus = None,
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
        }
        failure.map_or(Ok(()), Err)
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        if !self.retained {
            let _ = self.restore();
        }
    }
}
pub(super) fn os_error(operation: &str) -> Error {
    Error::invalid(format!("{operation}: {}", std::io::Error::last_os_error()))
}
