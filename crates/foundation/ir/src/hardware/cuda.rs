use serde::{Deserialize, Serialize};

/// Preserve NVIDIA architecture information instead of reducing it to generic GPU flags.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NvidiaArchitecture {
    pub compute_major: u32,
    pub compute_minor: u32,
    pub multiprocessors: u32,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Independent NVIDIA hardware feature bits describe device capabilities rather than mutually exclusive runtime states"
)]
pub struct NvidiaCapabilities {
    pub architecture: NvidiaArchitecture,
    pub tensor_core_generation: Option<u16>,
    pub warp_size: u32,
    pub graphs: bool,
    pub tma: bool,
    pub clusters: bool,
    pub pinned_transfer: bool,
    pub cuda_ipc: bool,
    pub nvlink: bool,
    pub gpu_direct: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudaRequirements {
    pub graphs: bool,
    pub tma: bool,
    pub clusters: bool,
}
impl NvidiaCapabilities {
    #[must_use]
    pub const fn satisfies(&self, r: &CudaRequirements) -> bool {
        (!r.graphs || self.graphs) && (!r.tma || self.tma) && (!r.clusters || self.clusters)
    }
}
