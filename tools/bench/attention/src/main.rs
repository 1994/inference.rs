//! Isolated Candle performance reference. No dependency is added to the production engine.
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;

mod device;
mod measure;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

struct Inputs {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cos: Tensor,
    sin: Tensor,
    lengths_q: Tensor,
    lengths_k: Tensor,
    max_q: usize,
    max_k: usize,
    left: Option<usize>,
    right: Option<usize>,
    rope: bool,
    tokens: usize,
    heads: usize,
    head: usize,
    half: usize,
}

fn number(shape: &serde_json::Value, key: &str) -> Result<usize> {
    Ok(usize::try_from(
        shape[key].as_u64().ok_or("missing shape field")?,
    )?)
}

impl Inputs {
    fn new(
        tensors: &HashMap<String, Tensor>,
        shape: &serde_json::Value,
        device: &Device,
    ) -> Result<Self> {
        let name = shape["id"].as_str().ok_or("missing case id")?;
        let get = |key| {
            tensors
                .get(&format!("{name}/{key}"))
                .cloned()
                .ok_or("missing tensor")
        };
        let tokens = number(shape, "tokens")?;
        let heads = number(shape, "heads")?;
        let head = number(shape, "head_dim")?;
        let half = (head / 2).next_power_of_two();
        let keys = number(shape, "kv_tokens")?;
        let mask = shape["mask"].as_str().ok_or("missing mask")?;
        let max_q = if mask == "segments" {
            number(shape, "frame_tokens")?
        } else {
            tokens
        };
        let max_k = if mask == "segments" {
            number(shape, "frame_tokens")?
        } else {
            keys
        };
        let lengths = |count: usize, frame: usize| -> Result<Tensor> {
            let mut offsets: Vec<u32> = (0..count)
                .step_by(frame)
                .map(u32::try_from)
                .collect::<std::result::Result<_, _>>()?;
            offsets.push(u32::try_from(count)?);
            let len = offsets.len();
            Ok(Tensor::from_vec(offsets, len, device)?)
        };
        let (left, right) = match mask {
            "causal" => (None, Some(0)),
            "window" => (Some(number(shape, "left")?), Some(number(shape, "right")?)),
            "none" | "segments" => (None, None),
            _ => return Err("unknown mask".into()),
        };
        Ok(Self {
            q: get("q")?,
            k: get("k")?,
            v: get("v")?,
            cos: get("cos")?.unsqueeze(1)?,
            sin: get("sin")?.unsqueeze(1)?,
            lengths_q: lengths(tokens, max_q)?,
            lengths_k: lengths(keys, max_k)?,
            max_q,
            max_k,
            left,
            right,
            rope: shape["rope"].as_bool().ok_or("missing pipeline kind")?,
            tokens,
            heads,
            head,
            half,
        })
    }

    fn rotate(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let low = x.narrow(2, 0, 1)?;
        let high = x.narrow(2, 1, 1)?.neg()?;
        let swapped = Tensor::cat(&[high, low], 2)?;
        x.broadcast_mul(&self.cos)?
            .add(&swapped.broadcast_mul(&self.sin)?)
    }

    fn compact(&self, x: &Tensor, dtype: DType) -> candle_core::Result<Tensor> {
        x.narrow(3, 0, self.head / 2)?
            .contiguous()?
            .reshape((x.dim(0)?, x.dim(1)?, self.head))?
            .to_dtype(dtype)
    }

    fn prepare(&self, dtype: DType) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let q = if self.rope {
            self.rotate(&self.q)?
        } else {
            self.q.clone()
        };
        let k = if self.rope {
            self.rotate(&self.k)?
        } else {
            self.k.clone()
        };
        Ok((
            self.compact(&q, dtype)?,
            self.compact(&k, dtype)?,
            self.compact(&self.v, dtype)?,
        ))
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "Attention head dimensions are small exact integers"
    )]
    fn core(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> candle_core::Result<Tensor> {
        candle_flash_attn::flash_attn_varlen_windowed(
            q,
            k,
            v,
            &self.lengths_q,
            &self.lengths_k,
            self.max_q,
            self.max_k,
            1.0 / (self.head as f32).sqrt(),
            self.left,
            self.right,
        )
    }

    fn execute(&self, dtype: DType) -> candle_core::Result<Tensor> {
        let (q, k, v) = self.prepare(dtype)?;
        let output = self.core(&q, &k, &v)?;
        // Include the conversion back to the engine's F32 padded layout in timing.
        output
            .to_dtype(DType::F32)?
            .reshape((self.tokens, self.heads, 2, self.head / 2))?
            .pad_with_zeros(3, 0, self.half - self.head / 2)
    }
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("attention gate requires release".into());
    }
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: candle-reference FIXTURES [f16|bf16]")?;
    let dtype = match args.next().as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        _ => return Err("explicit reference dtype required".into()),
    };
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(
        std::path::Path::new(&path).with_extension("json"),
    )?)?;
    let device = device::reference_device()?;
    let tensors = candle_core::safetensors::load(path, &device)?;
    for shape in manifest["cases"].as_array().ok_or("missing cases")? {
        let inputs = Inputs::new(&tensors, shape, &device)?;
        let name = shape["id"].as_str().ok_or("missing case id")?;
        let expected = tensors
            .get(&format!("{name}/output"))
            .ok_or("missing oracle")?;
        measure::run(&device, &inputs, dtype, expected, shape)?;
    }
    Ok(())
}
