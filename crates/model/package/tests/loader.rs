use infer_core::{Error, ErrorCode, ModelId, Result};
use std::{cell::Cell, path::Path, rc::Rc};

use infer_models::*;

struct Tensor {
    data: Vec<u8>,
    live: Rc<Cell<usize>>,
}
impl Drop for Tensor {
    fn drop(&mut self) {
        self.live.set(self.live.get() - 1);
    }
}
#[derive(Default)]
struct Target {
    live: Rc<Cell<usize>>,
    allocations: usize,
    uploads: usize,
    largest_chunk: usize,
    fail_after: Option<usize>,
}
impl WeightTarget for Target {
    type Tensor = Tensor;
    fn allocate(&mut self, plan: &TensorLoadPlan) -> Result<Tensor> {
        self.allocations += 1;
        self.live.set(self.live.get() + 1);
        Ok(Tensor {
            data: vec![0; usize::try_from(plan.bytes).map_err(|_| Error::invalid("fixture size"))?],
            live: self.live.clone(),
        })
    }
    fn upload(&mut self, tensor: &mut Tensor, offset: u64, input: &[u8]) -> Result<()> {
        if self.fail_after == Some(self.uploads) {
            return Err(Error::new(ErrorCode::Backend, "injected upload failure"));
        }
        self.uploads += 1;
        self.largest_chunk = self.largest_chunk.max(input.len());
        let start = usize::try_from(offset).map_err(|_| Error::invalid("fixture offset"))?;
        tensor.data[start..start + input.len()].copy_from_slice(input);
        Ok(())
    }
}
fn package() -> Result<ModelPackage> {
    ModelPackage::open(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny"),
        ModelId::ONE,
    )
}
const fn options(bytes: u64, chunk: usize) -> LoadOptions {
    LoadOptions {
        storage: WeightStorage::Native,
        resident_budget_bytes: bytes,
        staging_budget_bytes: chunk,
        chunk_bytes: chunk,
    }
}
#[test]
fn bounded_upload_reconstructs_every_weight_and_hash_is_chunk_independent() -> Result<()> {
    let mut p = package()?;
    let expected = p.load_host_weights(1 << 20)?;
    let mut target = Target::default();
    let budget = p.manifest.weights_bytes;
    let loaded = load_weights(&mut p, &mut target, options(budget, 12))?;
    assert_eq!(loaded.plan.resident_bytes, budget);
    assert_eq!(loaded.plan.staging_bytes, 12);
    assert_eq!(target.largest_chunk, 12);
    assert!(target.uploads > target.allocations);
    for (slot, tensor) in &loaded.tensors {
        let expected: Vec<_> = expected[slot]
            .data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(tensor.data, expected);
    }
    let other = load_weights(&mut p, &mut Target::default(), options(budget, 4096))?;
    assert_eq!(loaded.payload_fingerprint, other.payload_fingerprint);
    drop(loaded);
    assert_eq!(target.live.get(), 0);
    Ok(())
}
#[test]
fn insufficient_budget_allocates_nothing_and_late_failure_releases_every_tensor() -> Result<()> {
    let mut p = package()?;
    let mut target = Target::default();
    let error = load_weights(&mut p, &mut target, options(1, 12))
        .err()
        .ok_or_else(|| Error::invariant("capacity failure expected"))?;
    assert_eq!(error.code, ErrorCode::Capacity);
    assert_eq!(target.allocations, 0);
    let mut options = options(p.manifest.weights_bytes, 12);
    options.staging_budget_bytes = 1;
    assert!(load_weights(&mut p, &mut target, options).is_err());
    assert_eq!(target.allocations, 0);
    target.fail_after = Some(100);
    let options = LoadOptions {
        staging_budget_bytes: 12,
        ..options
    };
    let error = load_weights(&mut p, &mut target, options)
        .err()
        .ok_or_else(|| Error::invariant("upload failure expected"))?;
    assert_eq!(error.code, ErrorCode::Backend);
    assert!(target.allocations > 1);
    assert_eq!(target.live.get(), 0);
    let recovered = load_weights(&mut p, &mut Target::default(), options)?;
    assert_eq!(recovered.tensors.len(), p.manifest.bindings.len());
    Ok(())
}
#[test]
fn aligned_native_and_converted_float_chunks_have_exact_values() -> Result<()> {
    for (dtype, input) in [
        (TensorDtype::BF16, vec![0x80, 0x3f, 0, 0xc0, 0, 0x3f, 0, 0]),
        (TensorDtype::F16, vec![0, 0x3c, 0, 0xc0, 0, 0x38, 0, 0]),
    ] {
        let output = convert_float_bytes(&input, dtype)?;
        let expected: Vec<_> = [1.0f32, -2.0, 0.5, 0.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        assert_eq!(output, expected);
        assert!(convert_float_bytes(&input[..3], dtype).is_err());
    }
    for (dtype, input) in [
        (TensorDtype::BF16, [0x80, 0x7f]),
        (TensorDtype::F16, [0, 0x7c]),
    ] {
        assert!(convert_float_bytes(&input, dtype).is_err());
    }
    Ok(())
}

#[test]
fn changed_binding_metadata_is_rejected_before_target_allocation() -> Result<()> {
    let mut p = package()?;
    let mut target = Target::default();
    let budget = p.manifest.weights_bytes;
    p.manifest.bindings[0].dtype = TensorDtype::BF16;
    let error = load_weights(&mut p, &mut target, options(budget, 12))
        .err()
        .ok_or_else(|| Error::invariant("binding mismatch expected"))?;
    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(target.allocations, 0);
    Ok(())
}

fn mixed_package() -> std::result::Result<ModelPackage, Box<dyn std::error::Error>> {
    use std::collections::BTreeMap;
    let root = std::env::temp_dir().join(format!("infer-mixed-loader-{}", std::process::id()));
    std::fs::create_dir_all(&root)?;
    let original = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    std::fs::copy(original.join("config.json"), root.join("config.json"))?;
    let mut file = SafetensorsFile::open(original.join("model.safetensors"))?;
    let mut headers = BTreeMap::new();
    let mut payload = Vec::new();
    for (i, (name, mut header)) in file.tensors.clone().into_iter().enumerate() {
        let tensor = file.read_f32(&name, 1 << 20)?;
        let start = payload.len() as u64;
        if i == 0 {
            header.dtype = TensorDtype::BF16;
            for value in tensor.data {
                payload.extend_from_slice(&u16::try_from(value.to_bits() >> 16)?.to_le_bytes());
            }
        } else {
            for value in tensor.data {
                payload.extend_from_slice(&value.to_le_bytes());
            }
        }
        header.data_offsets = [start, payload.len() as u64];
        headers.insert(name, header);
    }
    let mut header = serde_json::to_vec(&headers)?;
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(payload);
    std::fs::write(root.join("model.safetensors"), bytes)?;
    Ok(ModelPackage::open(root, ModelId::ONE)?)
}
#[test]
fn f32_upload_conversion_accounts_for_both_live_chunks()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut p = mixed_package()?;
    let expected = p.load_host_weights(1 << 20)?;
    let options = LoadOptions {
        storage: WeightStorage::F32,
        resident_budget_bytes: p.manifest.host_f32_bytes,
        staging_budget_bytes: 36,
        chunk_bytes: 12,
    };
    let mut target = Target::default();
    assert!(
        load_weights(
            &mut p,
            &mut target,
            LoadOptions {
                staging_budget_bytes: 35,
                ..options
            }
        )
        .is_err()
    );
    assert_eq!(target.allocations, 0);
    let loaded = load_weights(&mut p, &mut target, options)?;
    assert_eq!(loaded.plan.staging_bytes, 36);
    assert_eq!(loaded.plan.resident_bytes, p.manifest.host_f32_bytes);
    assert_eq!(target.largest_chunk, 24);
    for (slot, tensor) in &loaded.tensors {
        let expected: Vec<_> = expected[slot]
            .data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(tensor.data, expected);
    }
    std::fs::remove_dir_all(p.root)?;
    Ok(())
}

#[test]
fn mapped_chunks_borrow_the_payload_and_enforce_bounds() -> Result<()> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/qwen-hybrid-tiny/model.safetensors");
    let mut file = SafetensorsFile::open(path)?;
    let name = file
        .tensors
        .keys()
        .next()
        .ok_or_else(|| Error::invariant("fixture tensor missing"))?
        .clone();
    let expected = file.read_bytes(&name, u64::MAX)?;
    let base = file.bytes(&name, u64::MAX)?.as_ptr();
    let mut visited = 0usize;
    file.visit_float_chunks(&name, 13, |offset, chunk| {
        let offset = usize::try_from(offset).map_err(|_| Error::invalid("test offset"))?;
        assert_eq!(chunk.as_ptr(), base.wrapping_add(offset));
        assert_eq!(chunk, &expected[offset..offset + chunk.len()]);
        visited += chunk.len();
        Ok(())
    })?;
    assert_eq!(visited, expected.len());
    assert_eq!(
        file.bytes(&name, 0).err().map(|e| e.code),
        Some(ErrorCode::Capacity)
    );
    assert!(file.bytes("missing tensor", u64::MAX).is_err());
    file.tensors
        .get_mut(&name)
        .ok_or_else(|| Error::invariant("fixture tensor missing"))?
        .data_offsets = [u64::MAX - 4, u64::MAX];
    assert!(file.bytes(&name, u64::MAX).is_err());
    Ok(())
}
