use super::super::{PlacementReport, ThreadPlacement, ThreadQos};
use crate::{Error, Result};

/// Darwin `QOS_CLASS_UTILITY` class passed to `pthread_set_qos_class_self_np`.
const QOS_CLASS_UTILITY: u32 = 0x11;
/// Darwin `QOS_CLASS_USER_INITIATED` class passed to `pthread_set_qos_class_self_np`.
const QOS_CLASS_USER_INITIATED: u32 = 0x19;
/// Darwin `QOS_CLASS_USER_INTERACTIVE` class passed to `pthread_set_qos_class_self_np`.
const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
/// Darwin thread-policy flavor selecting the thread affinity policy.
const THREAD_AFFINITY_POLICY_FLAVOR: libc::thread_policy_flavor_t = 4;

unsafe extern "C" {
    fn mach_thread_self() -> libc::mach_port_t;
    static mach_task_self_: libc::mach_port_t;
    fn mach_port_deallocate(
        task: libc::mach_port_t,
        port: libc::mach_port_t,
    ) -> libc::kern_return_t;
    fn pthread_set_qos_class_self_np(qos: u32, relative_priority: i32) -> i32;
    fn pthread_get_qos_class_np(thread: libc::pthread_t, qos: *mut u32, priority: *mut i32) -> i32;
}
pub(in crate::placement) fn allowed_cpus() -> Result<Vec<usize>> {
    Ok((0..std::thread::available_parallelism()
        .map_err(|error| Error::invalid(error.to_string()))?
        .get())
        .collect())
}
pub(in crate::placement) struct Guard {
    old_qos: Option<(u32, i32)>,
    old_tag: Option<i32>,
    retained: bool,
    placement: ThreadPlacement,
}
impl Guard {
    pub fn enter(placement: &ThreadPlacement) -> Result<Self> {
        if !placement.cpus.is_empty() || placement.numa.is_some() {
            return Err(Error::unsupported(
                "macOS supports QoS and affinity hints, not CPU/NUMA binding",
            ));
        }
        let mut guard = Self {
            old_qos: None,
            old_tag: None,
            retained: false,
            placement: placement.clone(),
        };
        if placement.qos != ThreadQos::Inherit {
            let (mut qos, mut priority) = (0, 0);
            // SAFETY: pthread_self returns the current valid pthread handle.
            let thread = unsafe { libc::pthread_self() };
            // SAFETY: Output pointers are initialized stack values with the ABI's exact types.
            if unsafe { pthread_get_qos_class_np(thread, &raw mut qos, &raw mut priority) } != 0 {
                return Err(Error::invalid("cannot read macOS thread QoS"));
            }
            guard.old_qos = Some((qos, priority));
            set_qos(
                match placement.qos {
                    ThreadQos::Inherit => qos,
                    ThreadQos::Utility => QOS_CLASS_UTILITY,
                    ThreadQos::UserInitiated => QOS_CLASS_USER_INITIATED,
                    ThreadQos::UserInteractive => QOS_CLASS_USER_INTERACTIVE,
                },
                0,
            )?;
        }
        if let Some(tag) = placement.affinity_tag {
            if tag < 0 {
                return Err(Error::invalid("negative macOS affinity tag"));
            }
            guard.old_tag = Some(affinity(Some(tag))?);
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
        if let Some(tag) = self.old_tag {
            match affinity(Some(tag)) {
                Ok(_) => self.old_tag = None,
                Err(error) => failure = Some(error),
            }
        }
        if let Some((qos, priority)) = self.old_qos {
            match set_qos(qos, priority) {
                Ok(()) => self.old_qos = None,
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
fn set_qos(qos: u32, priority: i32) -> Result<()> {
    // SAFETY: These are documented Darwin QoS classes and bounded relative priority from get_qos.
    if unsafe { pthread_set_qos_class_self_np(qos, priority) } == 0 {
        Ok(())
    } else {
        Err(Error::invalid("macOS thread QoS rejected"))
    }
}
fn affinity(tag: Option<i32>) -> Result<i32> {
    // SAFETY: Mach returns a send right for the current live thread; released below on every branch.
    let thread = unsafe { mach_thread_self() };
    let (mut old, mut count, mut default_policy) = (0i32, libc::THREAD_AFFINITY_POLICY_COUNT, 0);
    // SAFETY: THREAD_AFFINITY_POLICY is one integer; count and boolean outputs are valid stack cells.
    let read = unsafe {
        libc::thread_policy_get(
            thread,
            THREAD_AFFINITY_POLICY_FLAVOR,
            &raw mut old,
            &raw mut count,
            &raw mut default_policy,
        )
    };
    let written = if read == 0
        && let Some(mut tag) = tag
    {
        // SAFETY: Valid thread send right and exactly one integer for the affinity policy ABI.
        unsafe {
            libc::thread_policy_set(
                thread,
                THREAD_AFFINITY_POLICY_FLAVOR,
                &raw mut tag,
                libc::THREAD_AFFINITY_POLICY_COUNT,
            )
        }
    } else {
        read
    };
    // SAFETY: The process owns the send right returned by mach_thread_self; release exactly once.
    let task = unsafe { mach_task_self_ };
    // SAFETY: thread is the retained send right in the current task's port namespace.
    let released = unsafe { mach_port_deallocate(task, thread) };
    if read == 0 && written == 0 && released == 0 {
        Ok(old)
    } else {
        Err(Error::invalid("macOS affinity hint rejected"))
    }
}
