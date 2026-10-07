use infer_core::{Error, ErrorCode, Result};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs::File, io::Read, path::Path, path::PathBuf};

const MAX_HEADER: u64 = 16 * 1024 * 1024;

/// Width of the little-endian header-length prefix in a safetensors file.
const HEADER_LEN_BYTES: usize = 8;
/// `HEADER_LEN_BYTES` as `u64`, for file-offset arithmetic.
const HEADER_LEN_BYTES_U64: u64 = HEADER_LEN_BYTES as u64;
/// Left shift that moves a 16-bit pattern into the high half of a `u32`.
const HIGH_HALF_SHIFT: u32 = 16;
/// Fraction bits in an IEEE 754 binary16 value.
const F16_MANTISSA_BITS: u32 = 10;
/// Exponent bias of an IEEE 754 binary16 value.
const F16_EXPONENT_BIAS: u32 = 15;
/// Maximum biased binary16 exponent, marking infinity or NaN.
const F16_MAX_EXPONENT: u16 = 31;
/// In-place exponent field of a binary16 bit pattern.
const F16_EXPONENT_FIELD_MASK: u16 = 0x7c00;
/// Sign bit of a binary16 bit pattern.
const F16_SIGN_MASK: u16 = 0x8000;
/// Fraction field of a binary16 bit pattern.
const F16_MANTISSA_MASK: u16 = 0x03ff;
/// Left shift aligning a binary16 subnormal fraction with its implicit one.
const F16_SUBNORMAL_ALIGN_SHIFT: u32 = u32::BITS - 1 - F16_MANTISSA_BITS;
/// Exponent bias of an IEEE 754 binary32 value.
const F32_EXPONENT_BIAS: u32 = 127;
/// Fraction bits in an IEEE 754 binary32 value.
const F32_MANTISSA_BITS: u32 = 23;
/// In-place exponent field of a binary32 bit pattern.
const F32_EXPONENT_FIELD_MASK: u32 = 0x7f80_0000;
/// Exponent-field delta mapping a binary16 exponent to a binary32 exponent.
const F16_TO_F32_EXPONENT_DELTA: u32 = F32_EXPONENT_BIAS - F16_EXPONENT_BIAS;
/// Binary32 exponent field of the largest binary16 subnormal.
const F16_SUBNORMAL_F32_EXPONENT: u32 = F16_TO_F32_EXPONENT_DELTA + 1;
/// Left shift moving a binary16 fraction into the binary32 fraction field.
const F16_MANTISSA_TO_F32_SHIFT: u32 = F32_MANTISSA_BITS - F16_MANTISSA_BITS;
/// Exponent field of a BF16 bit pattern.
const BF16_EXPONENT_FIELD_MASK: u16 = 0x7f80;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorDtype {
    F32,
    BF16,
    F16,
    F64,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    BOOL,
    #[serde(rename = "F8_E4M3")]
    F8E4m3,
    #[serde(rename = "F8_E5M2")]
    F8E5m2,
}
/// Bytes per 64-bit tensor element.
const ELEMENT_BYTES_64: u64 = 8;
/// Bytes per 32-bit tensor element.
const ELEMENT_BYTES_32: u64 = 4;
/// Bytes per 16-bit tensor element.
const ELEMENT_BYTES_16: u64 = 2;
/// Bytes per 8-bit or boolean tensor element.
const ELEMENT_BYTES_8: u64 = 1;

impl TensorDtype {
    #[must_use]
    pub const fn bytes(self) -> u64 {
        match self {
            Self::F64 | Self::I64 | Self::U64 => ELEMENT_BYTES_64,
            Self::F32 | Self::I32 | Self::U32 => ELEMENT_BYTES_32,
            Self::BF16 | Self::F16 | Self::I16 | Self::U16 => ELEMENT_BYTES_16,
            _ => ELEMENT_BYTES_8,
        }
    }
    #[must_use]
    pub const fn is_host_float(self) -> bool {
        matches!(self, Self::F32 | Self::BF16 | Self::F16)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorHeader {
    pub dtype: TensorDtype,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
}
impl TensorHeader {
    ///
    /// # Errors
    /// Returns an invalid-input error if multiplying the dimensions overflows.
    pub fn elements(&self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |n, d| {
            n.checked_mul(*d)
                .ok_or_else(|| Error::invalid("tensor shape overflow"))
        })
    }
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.data_offsets[1] - self.data_offsets[0]
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}
impl HostTensor {
    ///
    /// # Errors
    /// Returns an invalid-input or invariant error if dimensions, identities, ranges, or ownership are inconsistent.
    pub fn validate(&self) -> Result<()> {
        let n = self
            .shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| Error::invalid("tensor shape overflow"))?;
        if n != self.data.len() || self.data.iter().any(|v| !v.is_finite()) {
            return Err(Error::invalid("tensor payload shape/numerics mismatch"));
        }
        Ok(())
    }
}
/// Validates only bounded headers at open. Payload pages are mapped on demand.
///
/// The weight file must not be modified or truncated while this object is alive.
/// Replace model files by renaming a new file instead of overwriting them in place.
pub struct SafetensorsFile {
    mapping: memmap2::Mmap,
    pub path: PathBuf,
    pub tensors: BTreeMap<String, TensorHeader>,
    data_start: u64,
}
impl SafetensorsFile {
    /// Bytes preceding the validated, contiguous tensor payload.
    #[must_use]
    pub const fn header_bytes(&self) -> u64 {
        self.data_start
    }
    /// Read validated raw tensor storage without expanding a quantized representation.
    /// # Errors
    /// Rejects missing tensors, address overflow, short reads or an exceeded staging budget.
    pub fn read_bytes(&mut self, name: &str, budget_bytes: u64) -> Result<Vec<u8>> {
        Ok(self.bytes(name, budget_bytes)?.to_vec())
    }
    /// Borrow stored tensor bytes without allocating a host staging buffer.
    /// # Errors
    /// Rejects missing tensors, invalid bounds or an exceeded byte budget.
    pub fn bytes(&self, name: &str, budget_bytes: u64) -> Result<&[u8]> {
        let header = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::invalid(format!("missing tensor {name}")))?;
        if header.byte_len() > budget_bytes {
            return Err(Error::new(
                ErrorCode::Capacity,
                "raw tensor exceeds staging budget",
            ));
        }
        self.payload(header.data_offsets[0], header.byte_len())
    }
    fn payload(&self, offset: u64, length: u64) -> Result<&[u8]> {
        let start = self
            .data_start
            .checked_add(offset)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| Error::invalid("tensor offset exceeds address space"))?;
        let end = usize::try_from(length)
            .ok()
            .and_then(|n| start.checked_add(n))
            .ok_or_else(|| Error::invalid("tensor size exceeds address space"))?;
        self.mapping
            .get(start..end)
            .ok_or_else(|| Error::invalid("tensor outside mapped file"))
    }
    /// Read native floating-point payload in aligned bounded chunks, validating every value.
    /// # Errors
    /// Returns I/O, format, numeric, or callback errors. Earlier chunks may already be consumed.
    pub fn visit_float_chunks(
        &mut self,
        name: &str,
        chunk_bytes: usize,
        mut consume: impl FnMut(u64, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let header = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::invalid(format!("missing tensor {name}")))?;
        let width = usize::try_from(header.dtype.bytes())
            .map_err(|_| Error::invalid("weight byte width"))?;
        if !header.dtype.is_host_float() || chunk_bytes < width {
            return Err(Error::unsupported(
                "invalid floating-point streaming format/chunk",
            ));
        }
        let chunk = chunk_bytes / width * width;
        let bytes = self.bytes(name, u64::MAX)?;
        let mut offset = 0;
        for input in bytes.chunks(chunk) {
            validate_float_bytes(input, header.dtype)?;
            consume(offset, input)?;
            offset += input.len() as u64;
        }
        Ok(())
    }
    ///
    /// # Errors
    /// Returns an I/O or invalid-input error for missing, malformed, unsupported, or oversized assets.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        // Deserialize as entries to reject duplicate names instead of silently overwriting.
        struct Entries(BTreeMap<String, serde_json::Value>);
        impl<'de> Deserialize<'de> for Entries {
            fn deserialize<D: serde::Deserializer<'de>>(
                d: D,
            ) -> std::result::Result<Self, D::Error> {
                struct Visitor;
                impl<'de> serde::de::Visitor<'de> for Visitor {
                    type Value = Entries;
                    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        f.write_str("a unique tensor map")
                    }
                    fn visit_map<M: serde::de::MapAccess<'de>>(
                        self,
                        mut m: M,
                    ) -> std::result::Result<Entries, M::Error> {
                        let mut entries = BTreeMap::new();
                        while let Some((k, v)) = m.next_entry::<String, serde_json::Value>()? {
                            if entries.insert(k, v).is_some() {
                                return Err(serde::de::Error::custom("duplicate tensor name"));
                            }
                        }
                        Ok(Entries(entries))
                    }
                }
                d.deserialize_map(Visitor)
            }
        }
        let path = path.as_ref();
        let mut file =
            File::open(path).map_err(|e| Error::invalid(format!("{}: {e}", path.display())))?;
        let file_len = file.metadata().map_err(io_error)?.len();
        let mut prefix = [0u8; HEADER_LEN_BYTES];
        file.read_exact(&mut prefix).map_err(io_error)?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len == 0
            || header_len > MAX_HEADER
            || header_len > file_len.saturating_sub(HEADER_LEN_BYTES_U64)
        {
            return Err(Error::invalid(
                "invalid/budget-exceeding safetensors header length",
            ));
        }
        let header_size = usize::try_from(header_len)
            .map_err(|_| Error::invalid("header size exceeds address space"))?;
        let mut bytes = vec![0; header_size];
        file.read_exact(&mut bytes).map_err(io_error)?;
        let Entries(entries) =
            serde_json::from_slice(&bytes).map_err(|e| Error::invalid(e.to_string()))?;
        let data_start = header_len + HEADER_LEN_BYTES_U64;
        let data_len = file_len - data_start;
        let mut tensors = BTreeMap::new();
        for (name, value) in entries {
            if name == "__metadata__" {
                let _: BTreeMap<String, String> =
                    serde_json::from_value(value).map_err(|e| Error::invalid(e.to_string()))?;
                continue;
            }
            if name.is_empty() {
                return Err(Error::invalid("empty tensor name"));
            }
            let header: TensorHeader = serde_json::from_value(value)
                .map_err(|e| Error::invalid(format!("{name}: {e}")))?;
            let expected = (header.elements()? as u64)
                .checked_mul(header.dtype.bytes())
                .ok_or_else(|| Error::invalid("tensor byte overflow"))?;
            let [start, end] = header.data_offsets;
            if end < start || end > data_len || end - start != expected {
                return Err(Error::invalid(format!(
                    "{name}: inconsistent tensor bounds/shape"
                )));
            }
            tensors.insert(name, header);
        }
        if tensors.is_empty() {
            return Err(Error::invalid("empty safetensors file"));
        }
        let mut intervals: Vec<_> = tensors.values().map(|h| h.data_offsets).collect();
        intervals.sort_unstable();
        let mut end = 0;
        for [start, next] in intervals {
            if start != end {
                return Err(Error::invalid("overlapping or gapped tensor payload"));
            }
            end = next;
        }
        if end != data_len {
            return Err(Error::invalid("unindexed trailing tensor payload"));
        }
        Ok(Self {
            mapping: map_weights(&file)?,
            path: path.to_owned(),
            tensors,
            data_start,
        })
    }
    ///
    /// # Errors
    /// Returns an I/O, not-found, unsupported, or capacity error if the tensor cannot be read and converted within the byte budget.
    pub fn read_f32(&mut self, name: &str, budget_bytes: u64) -> Result<HostTensor> {
        let header = self
            .tensors
            .get(name)
            .cloned()
            .ok_or_else(|| Error::invalid(format!("missing tensor {name}")))?;
        let data = self.read_elements(name, 0, header.elements()?, budget_bytes)?;
        Ok(HostTensor {
            shape: header.shape,
            data,
        })
    }
    ///
    /// # Errors
    /// Returns an I/O, invalid-input, unsupported, or capacity error for invalid rows, tensor formats, or an insufficient byte budget.
    pub fn read_row_f32(&mut self, name: &str, row: usize, budget_bytes: u64) -> Result<Vec<f32>> {
        let h = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::invalid(format!("missing tensor {name}")))?;
        if h.shape.len() != 2 || row >= h.shape[0] {
            return Err(Error::invalid("invalid tensor row"));
        }
        let width = h.shape[1];
        self.read_elements(name, row * width, width, budget_bytes)
    }
    fn read_elements(
        &self,
        name: &str,
        offset: usize,
        count: usize,
        budget_bytes: u64,
    ) -> Result<Vec<f32>> {
        let header = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::invalid("missing tensor"))?;
        if !header.dtype.is_host_float() {
            return Err(Error::unsupported(format!(
                "host conversion for {:?}",
                header.dtype
            )));
        }
        let output_bytes = (count as u64)
            .checked_mul(crate::constants::F32_BYTES_U64)
            .ok_or_else(|| Error::invalid("read size overflow"))?;
        let input_bytes = (count as u64)
            .checked_mul(header.dtype.bytes())
            .ok_or_else(|| Error::invalid("read size overflow"))?;
        if output_bytes
            .checked_add(input_bytes)
            .is_none_or(|n| n > budget_bytes)
        {
            return Err(Error::new(
                ErrorCode::Capacity,
                "tensor conversion exceeds host budget",
            ));
        }
        let offset = (offset as u64)
            .checked_mul(header.dtype.bytes())
            .and_then(|n| header.data_offsets[0].checked_add(n))
            .ok_or_else(|| Error::invalid("tensor offset overflow"))?;
        let bytes = self.payload(offset, input_bytes)?;
        let data: Vec<f32> = match header.dtype {
            TensorDtype::F32 => bytes
                .as_chunks::<{ crate::constants::F32_BYTES }>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            TensorDtype::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << HIGH_HALF_SHIFT))
                .collect(),
            TensorDtype::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half_to_f32(u16::from_le_bytes(*b)))
                .collect(),
            _ => unreachable!(),
        };
        if data.iter().any(|v| !v.is_finite()) {
            return Err(Error::invalid("non-finite weight tensor"));
        }
        Ok(data)
    }
}
#[expect(
    clippy::needless_pass_by_value,
    reason = "Result::map_err transfers its owned I/O error directly into this error adapter"
)]
fn io_error(e: std::io::Error) -> Error {
    Error::invalid(e.to_string())
}
fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & F16_SIGN_MASK) << HIGH_HALF_SHIFT;
    let exponent = (bits >> F16_MANTISSA_BITS) & F16_MAX_EXPONENT;
    let mantissa = u32::from(bits & F16_MANTISSA_MASK);
    let output = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let shift = mantissa.leading_zeros() - F16_SUBNORMAL_ALIGN_SHIFT;
            sign | ((F16_SUBNORMAL_F32_EXPONENT - shift) << F32_MANTISSA_BITS)
                | ((mantissa << shift & u32::from(F16_MANTISSA_MASK)) << F16_MANTISSA_TO_F32_SHIFT)
        }
        F16_MAX_EXPONENT => {
            sign | F32_EXPONENT_FIELD_MASK | (mantissa << F16_MANTISSA_TO_F32_SHIFT)
        }
        _ => {
            sign | ((u32::from(exponent) + F16_TO_F32_EXPONENT_DELTA) << F32_MANTISSA_BITS)
                | (mantissa << F16_MANTISSA_TO_F32_SHIFT)
        }
    };
    f32::from_bits(output)
}

fn validate_float_bytes(input: &[u8], dtype: TensorDtype) -> Result<()> {
    let valid =
        match dtype {
            TensorDtype::F32 => input
                .as_chunks::<{ crate::constants::F32_BYTES }>()
                .0
                .iter()
                .all(|b| f32::from_le_bytes(*b).is_finite()),
            TensorDtype::BF16 => input.as_chunks::<2>().0.iter().all(|b| {
                u16::from_le_bytes(*b) & BF16_EXPONENT_FIELD_MASK != BF16_EXPONENT_FIELD_MASK
            }),
            TensorDtype::F16 => input.as_chunks::<2>().0.iter().all(|b| {
                u16::from_le_bytes(*b) & F16_EXPONENT_FIELD_MASK != F16_EXPONENT_FIELD_MASK
            }),
            _ => return Err(Error::unsupported("float conversion format")),
        };
    if valid {
        Ok(())
    } else {
        Err(Error::invalid("non-finite weight tensor"))
    }
}

/// Convert an aligned, finite floating-point payload to little-endian F32 bytes.
/// # Errors
/// Returns a format, alignment, numeric, or size overflow error for invalid input.
pub fn convert_float_bytes(input: &[u8], dtype: TensorDtype) -> Result<Vec<u8>> {
    let width = usize::try_from(dtype.bytes()).map_err(|_| Error::invalid("weight byte width"))?;
    if !input.len().is_multiple_of(width) {
        return Err(Error::invalid("unaligned float payload"));
    }
    validate_float_bytes(input, dtype)?;
    if dtype == TensorDtype::F32 {
        return Ok(input.to_vec());
    }
    if !matches!(dtype, TensorDtype::BF16 | TensorDtype::F16) {
        return Err(Error::unsupported("float conversion format"));
    }
    let size = input
        .len()
        .checked_mul(2)
        .ok_or_else(|| Error::invalid("conversion size overflow"))?;
    let mut output = Vec::with_capacity(size);
    for b in input.as_chunks::<2>().0 {
        let bits = u16::from_le_bytes(*b);
        let value = if dtype == TensorDtype::BF16 {
            f32::from_bits(u32::from(bits) << HIGH_HALF_SHIFT)
        } else {
            half_to_f32(bits)
        };
        output.extend_from_slice(&value.to_le_bytes());
    }
    Ok(output)
}

/// Small deterministic fixtures and package export; data is row-major, little-endian.
///
/// # Errors
/// Returns an I/O or invalid-input error for invalid tensor shapes or values, size overflow, or failed writes.
pub fn write_safetensors(
    path: impl AsRef<Path>,
    tensors: &BTreeMap<String, HostTensor>,
) -> Result<()> {
    use std::io::Write;
    if tensors.is_empty() {
        return Err(Error::invalid("empty tensor export"));
    }
    let mut entries = BTreeMap::new();
    let mut offset = 0u64;
    for (name, tensor) in tensors {
        tensor.validate()?;
        let end = offset
            .checked_add(tensor.data.len() as u64 * crate::constants::F32_BYTES_U64)
            .ok_or_else(|| Error::invalid("tensor export overflow"))?;
        entries.insert(
            name,
            TensorHeader {
                dtype: TensorDtype::F32,
                shape: tensor.shape.clone(),
                data_offsets: [offset, end],
            },
        );
        offset = end;
    }
    let mut header = serde_json::to_vec(&entries).map_err(|e| Error::invalid(e.to_string()))?;
    while !header.len().is_multiple_of(HEADER_LEN_BYTES) {
        header.push(b' ');
    }
    let mut file = File::create(path).map_err(io_error)?;
    file.write_all(&(header.len() as u64).to_le_bytes())
        .map_err(io_error)?;
    file.write_all(&header).map_err(io_error)?;
    for tensor in tensors.values() {
        for value in &tensor.data {
            file.write_all(&value.to_le_bytes()).map_err(io_error)?;
        }
    }
    Ok(())
}

/// The mapping owns its virtual address range and outlives all borrowed payloads.
#[expect(
    unsafe_code,
    reason = "Read-only file mapping is isolated here; model assets must remain immutable while loaded"
)]
fn map_weights(file: &File) -> Result<memmap2::Mmap> {
    // SAFETY: Only immutable byte slices are exposed, bounded by the mapping and
    // tied to its lifetime. Model assets must not be modified or truncated while
    // loaded (the SafetensorsFile contract); deploy replacements by rename.
    unsafe { memmap2::MmapOptions::new().map(file) }.map_err(io_error)
}
