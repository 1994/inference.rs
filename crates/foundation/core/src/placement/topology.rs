#[cfg(target_os = "linux")]
use super::NumaPolicy;
use super::ThreadPlacement;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuTopology {
    pub allowed_cpus: Vec<usize>,
    pub allowed_nodes: Vec<usize>,
    /// One allowed logical CPU per physical core; avoids selecting both SMT siblings.
    pub physical_cores: Vec<usize>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementPair {
    pub scheduler: ThreadPlacement,
    pub device: ThreadPlacement,
}
/// Parse Linux cpulist syntax with bounded expansion and strict validation.
/// # Errors
/// Rejects malformed, reversed or excessive CPU/node ranges.
pub fn parse_cpu_list(input: &str) -> Result<Vec<usize>> {
    let mut cpus = Vec::new();
    if input.trim().is_empty() {
        return Ok(cpus);
    }
    for part in input.trim().split(',') {
        let (first, last) = part.split_once('-').unwrap_or((part, part));
        if !first.bytes().all(|byte| byte.is_ascii_digit())
            || !last.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(Error::invalid("invalid CPU list"));
        }
        let first = first
            .parse::<usize>()
            .map_err(|_| Error::invalid("invalid CPU list"))?;
        let last = last
            .parse::<usize>()
            .map_err(|_| Error::invalid("invalid CPU list"))?;
        if first > last
            || last >= 1_048_576
            || cpus.len().saturating_add(last - first + 1) > 1_048_576
        {
            return Err(Error::invalid("CPU list exceeds topology limits"));
        }
        cpus.extend(first..=last);
    }
    cpus.sort_unstable();
    cpus.dedup();
    Ok(cpus)
}
impl CpuTopology {
    /// # Errors
    /// Returns unavailable affinity/cpuset or malformed OS topology data.
    pub fn discover() -> Result<Self> {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        let allowed_cpus = super::native::allowed_cpus()?;
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        let allowed_cpus = (0..std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get))
            .collect();
        #[cfg(target_os = "linux")]
        let (physical_cores, allowed_nodes) = (physical_cores(&allowed_cpus)?, allowed_nodes()?);
        #[cfg(not(target_os = "linux"))]
        let (physical_cores, allowed_nodes) = (Vec::new(), Vec::new());
        Ok(Self {
            allowed_cpus,
            allowed_nodes,
            physical_cores,
        })
    }
}
impl PlacementPair {
    /// Resolve the GPU PCI node against process cpusets; never guess an unknown (-1) NUMA node.
    /// # Errors
    /// Rejects invalid PCI identities, unknown nodes, or fewer than two available physical cores.
    pub fn for_gpu(pci_address: &str) -> Result<Self> {
        validate_pci(pci_address)?;
        #[cfg(target_os = "linux")]
        {
            let topology = CpuTopology::discover()?;
            let node =
                std::fs::read_to_string(format!("/sys/bus/pci/devices/{pci_address}/numa_node"))
                    .map_err(|error| Error::invalid(error.to_string()))?
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| Error::invalid("GPU NUMA node unavailable"))?;
            if !topology.allowed_nodes.contains(&node) {
                return Err(Error::invalid("GPU NUMA node outside memory cpuset"));
            }
            let node_cpus = read_list(&format!("/sys/devices/system/node/node{node}/cpulist"))?;
            let mut cores = topology
                .physical_cores
                .into_iter()
                .filter(|cpu| node_cpus.contains(cpu));
            let scheduler = cores
                .next()
                .ok_or_else(|| Error::invalid("GPU node has no allowed core"))?;
            let device = cores.next().ok_or_else(|| {
                Error::invalid("GPU node needs two distinct allowed physical cores")
            })?;
            Ok(Self {
                scheduler: ThreadPlacement {
                    cpus: vec![scheduler],
                    numa: Some(NumaPolicy::Bind(node)),
                    ..ThreadPlacement::default()
                },
                device: ThreadPlacement {
                    cpus: vec![device],
                    numa: Some(NumaPolicy::Bind(node)),
                    ..ThreadPlacement::default()
                },
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = pci_address;
            Err(Error::unsupported("GPU PCI NUMA topology requires Linux"))
        }
    }
}
fn validate_pci(address: &str) -> Result<()> {
    let bytes = address.as_bytes();
    if bytes.len() == 12
        && bytes[4] == b':'
        && bytes[7] == b':'
        && bytes[10] == b'.'
        && bytes
            .iter()
            .enumerate()
            .all(|(i, b)| matches!(i, 4 | 7 | 10) || b.is_ascii_hexdigit())
        && u8::from_str_radix(&address[8..10], 16).is_ok_and(|slot| slot <= 31)
        && matches!(bytes[11], b'0'..=b'7')
    {
        Ok(())
    } else {
        Err(Error::invalid(
            "expected PCI address domain:bus:slot.function",
        ))
    }
}
#[cfg(target_os = "linux")]
pub(super) fn allowed_nodes() -> Result<Vec<usize>> {
    let status = std::fs::read_to_string("/proc/self/status")
        .map_err(|error| Error::invalid(error.to_string()))?;
    let list = status
        .lines()
        .find_map(|line| line.strip_prefix("Mems_allowed_list:"))
        .ok_or_else(|| Error::invalid("memory cpuset unavailable"))?;
    parse_cpu_list(list)
}
#[cfg(target_os = "linux")]
fn read_list(path: &str) -> Result<Vec<usize>> {
    parse_cpu_list(
        &std::fs::read_to_string(path).map_err(|error| Error::invalid(error.to_string()))?,
    )
}
#[cfg(target_os = "linux")]
fn physical_cores(cpus: &[usize]) -> Result<Vec<usize>> {
    let mut cores = Vec::new();
    for cpu in cpus {
        let siblings = read_list(&format!(
            "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        ))?;
        if !cores.iter().any(|core| siblings.contains(core)) {
            cores.push(*cpu);
        }
    }
    Ok(cores)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pci_identity_rejects_bad_separators_and_out_of_range_slot_function() {
        assert!(validate_pci("0000:01:00.0").is_ok());
        for address in [
            "0000.01:00.0",
            "0000:01:20.0",
            "0000:01:00.8",
            "../01:00.0",
            "0000:01:é.0",
        ] {
            assert!(validate_pci(address).is_err());
        }
    }
}
