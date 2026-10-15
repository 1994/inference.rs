//! Actual native IR/compiler identity and embedded Rust device artifact identity.
use super::super::{SHAPE_RANK, reference::Fixture};
use super::capture::Captured;
use crate::device::CudaDevice;
use infer_core::{Error, Result};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
};
type IdentityCache = Mutex<BTreeMap<([usize; SHAPE_RANK], String), String>>;
pub(super) fn fingerprint(device: &CudaDevice, shape: [usize; SHAPE_RANK]) -> Result<String> {
    use cutile::cutile_compiler::{cuda_tile_runtime_utils as compiler, hints::CompileOptions};
    static CACHE: OnceLock<IdentityCache> = OnceLock::new();
    let context = format!(
        "{}:{}:{:?}:{:?}:{:?}:{:?}",
        device.profile()?.architecture,
        compiler::tileiras_fingerprint(),
        compiler::bytecode_override(),
        CompileOptions::default(),
        std::env::var_os("CUTILE_DISABLE_CHECK_HOISTING"),
        std::env::var_os("CUTILE_FORCE_DEVICE_CHECKS")
    );
    let key = (shape, context.clone());
    let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let cached = cache
        .lock()
        .map_err(|_| Error::invariant("exact GDN identity memo poisoned"))?
        .get(&key)
        .cloned();
    if let Some(value) = cached {
        return Ok(value);
    }
    let fixture = Fixture::new_case(shape, false, [0, 0, 0, 0])?;
    let mut data = Captured::new(device, &fixture)?;
    let identities = data.identities(device)?;
    let mut hash = Sha256::new();
    for bytes in [
        context.as_bytes(),
        identities.join(":").as_bytes(),
        include_bytes!("../../../../kernels/oxide-gdn/gdn-sm120.ptx"),
        include_bytes!("../../../../kernels/oxide-gdn/src/main.rs"),
        include_bytes!("../../../../kernels/oxide-gdn/provenance.json"),
    ] {
        hash.update(bytes);
        hash.update([0]);
    }
    let value = format!("{:x}", hash.finalize());
    cache
        .lock()
        .map_err(|_| Error::invariant("exact GDN identity memo poisoned"))?
        .insert(key, value.clone());
    Ok(value)
}
