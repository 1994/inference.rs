use infer_core::{Error, ErrorCode, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap, fs::File, io::Read, io::Seek, io::SeekFrom, path::Path, path::PathBuf,
};

const MAX_HEADER: u64 = 16 * 1024 * 1024;

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
impl TensorDtype {
    #[must_use]
    pub const fn bytes(self) -> u64 {
        match self {
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::BF16 | Self::F16 | Self::I16 | Self::U16 => 2,
            _ => 1,
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
/// Validates only bounded headers at open. Payload is read by tensor/row on demand.
pub struct SafetensorsFile {
    file: File,
    pub path: PathBuf,
    pub tensors: BTreeMap<String, TensorHeader>,
    data_start: u64,
}
impl SafetensorsFile {
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
        let length = usize::try_from(header.byte_len().min(chunk as u64))
            .map_err(|_| Error::invalid("weight chunk exceeds address space"))?;
        let mut buffer = vec![0; length];
        self.file
            .seek(SeekFrom::Start(self.data_start + header.data_offsets[0]))
            .map_err(io_error)?;
        let mut offset = 0;
        while offset < header.byte_len() {
            let count = usize::try_from((header.byte_len() - offset).min(buffer.len() as u64))
                .map_err(|_| Error::invalid("weight chunk exceeds address space"))?;
            let input = &mut buffer[..count];
            self.file.read_exact(input).map_err(io_error)?;
            validate_float_bytes(input, header.dtype)?;
            consume(offset, input)?;
            offset += count as u64;
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
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix).map_err(io_error)?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len == 0 || header_len > MAX_HEADER || header_len > file_len.saturating_sub(8) {
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
        let data_start = header_len + 8;
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
            file,
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
        &mut self,
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
            .checked_mul(4)
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
        self.file
            .seek(SeekFrom::Start(
                self.data_start + header.data_offsets[0] + offset as u64 * header.dtype.bytes(),
            ))
            .map_err(io_error)?;
        let input_size = usize::try_from(input_bytes)
            .map_err(|_| Error::invalid("tensor size exceeds address space"))?;
        let mut bytes = vec![0; input_size];
        self.file.read_exact(&mut bytes).map_err(io_error)?;
        let data: Vec<f32> = match header.dtype {
            TensorDtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            TensorDtype::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
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
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x3ff);
    let output = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let shift = mantissa.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((mantissa << shift & 0x3ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(output)
}

fn validate_float_bytes(input: &[u8], dtype: TensorDtype) -> Result<()> {
    let valid = match dtype {
        TensorDtype::F32 => input
            .as_chunks::<4>()
            .0
            .iter()
            .all(|b| f32::from_le_bytes(*b).is_finite()),
        TensorDtype::BF16 => input
            .as_chunks::<2>()
            .0
            .iter()
            .all(|b| u16::from_le_bytes(*b) & 0x7f80 != 0x7f80),
        TensorDtype::F16 => input
            .as_chunks::<2>()
            .0
            .iter()
            .all(|b| u16::from_le_bytes(*b) & 0x7c00 != 0x7c00),
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
            f32::from_bits(u32::from(bits) << 16)
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
            .checked_add(tensor.data.len() as u64 * 4)
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
    while !header.len().is_multiple_of(8) {
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
