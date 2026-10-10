use super::{
    attention::attention,
    capture::{Capture, CaptureMode, error},
};
use cutile::prelude::*;
use infer_ir::{TensorNode, TensorOp};

type AttentionResult<E> = Result<(Tensor<f32>, Tensor<E>, Tensor<E>), DeviceError>;

/// Lane rows per tile in the tensor-core attention twin; the graph's lane count must be a
/// multiple of it so every tile is full.
const TILED_QUERY_TILE: usize = 16;
/// Key/value rows per tile in the tensor-core attention twin.
const TILED_KEY_BLOCK: usize = 32;
/// Head width the tensor-core twin has been measured at; other widths keep the SIMT kernel.
const TILED_HEAD_DIM: usize = 256;

impl Capture<'_> {
    pub fn record_attention<E: DType>(
        &mut self,
        node: &TensorNode,
        output: Tensor<f32>,
        keys: Tensor<E>,
        values: Tensor<E>,
        scales: [f32; 2],
    ) -> AttentionResult<E> {
        let TensorOp::Attention {
            query_heads,
            kv_heads,
            head_dim,
            window,
        } = node.op
        else {
            return Err(error("expected attention"));
        };
        if kv_heads == 0 || !query_heads.is_multiple_of(kv_heads) || !head_dim.is_power_of_two() {
            return Err(error("resident attention dimensions"));
        }
        if self.mode == CaptureMode::Prefill {
            return self.record_prefill_attention(node, output, keys, values, scales);
        }
        let mut keys = keys.reshape(&[kv_heads, self.capacity, head_dim])?;
        let mut values = values.reshape(&[kv_heads, self.capacity, head_dim])?;
        self.scope.record(
            attention::append(
                (&mut keys).partition([1, self.capacity, head_dim]),
                (&mut values).partition([1, self.capacity, head_dim]),
                &self.input(node.inputs[1])?.view(&[kv_heads, head_dim])?,
                &self.input(node.inputs[2])?.view(&[kv_heads, head_dim])?,
                self.metadata,
                scales[0],
                scales[1],
            )
            .generics(vec![
                E::DTYPE.as_str().into(),
                head_dim.to_string(),
                self.capacity.to_string(),
                i32::from(E::DTYPE != f32::DTYPE).to_string(),
            ]),
        )?;
        let mut output = output.reshape(&[query_heads, head_dim])?;
        if self.attention.supports(query_heads, head_dim)
            && window.is_none_or(|w| w >= crate::constants::ATTENTION_SPLIT_THRESHOLD)
        {
            let query = if let Some(query) = self.weights.constants.get(&node.inputs[0]) {
                query.view(&[query_heads * head_dim])?
            } else {
                match self.mode {
                    CaptureMode::Row(lane) => {
                        self.arena.row(node.inputs[0], lane).map_err(error)?
                    }
                    _ => self
                        .arena
                        .get(node.inputs[0])
                        .map_err(error)?
                        .view(&[query_heads * head_dim])?,
                }
            };
            self.attention.record(
                self.scope,
                &mut output,
                &query.view(&[query_heads, head_dim])?,
                &keys,
                &values,
                self.metadata,
                query_heads,
                kv_heads,
                head_dim,
                i32::try_from(window.unwrap_or(0)).map_err(error)?,
                scales,
            )?;
            return Ok((output, keys, values));
        }
        self.scope.record(
            attention::decode(
                (&mut output).partition([1, head_dim]),
                &self.input(node.inputs[0])?.view(&[query_heads, head_dim])?,
                &keys,
                &values,
                self.metadata,
                i32::try_from(window.unwrap_or(0)).map_err(error)?,
                scales[0],
                scales[1],
            )
            .generics(vec![
                E::DTYPE.as_str().into(),
                head_dim.to_string(),
                (query_heads / kv_heads).to_string(),
            ]),
        )?;
        Ok((output, keys, values))
    }
    fn record_prefill_attention<E: DType>(
        &self,
        node: &TensorNode,
        output: Tensor<f32>,
        keys: Tensor<E>,
        values: Tensor<E>,
        scales: [f32; 2],
    ) -> AttentionResult<E> {
        use super::attention_prefill::kernels;
        let TensorOp::Attention {
            query_heads,
            kv_heads,
            head_dim,
            window,
        } = node.op
        else {
            return Err(error("expected attention"));
        };
        let lanes = self.arena.lanes();
        let mut keys = keys.reshape(&[kv_heads, self.capacity, head_dim])?;
        let mut values = values.reshape(&[kv_heads, self.capacity, head_dim])?;
        self.scope.record(
            kernels::append(
                (&mut keys).partition([1, self.capacity, head_dim]),
                (&mut values).partition([1, self.capacity, head_dim]),
                &self
                    .input(node.inputs[1])?
                    .view(&[lanes, kv_heads, head_dim])?,
                &self
                    .input(node.inputs[2])?
                    .view(&[lanes, kv_heads, head_dim])?,
                self.metadata,
                self.table,
                scales[0],
                scales[1],
            )
            .generics(vec![
                E::DTYPE.as_str().into(),
                head_dim.to_string(),
                self.capacity.to_string(),
                i32::from(E::DTYPE != f32::DTYPE).to_string(),
                crate::constants::KV_BLOCK_TOKENS.to_string(),
            ]),
        )?;
        let mut output = output.reshape(&[lanes * query_heads, head_dim])?;
        // Tensor-core twin, off unless asked for: the two lowerings differ in the last bits and the
        // SIMT path is the established one. The binding is what matters here. q and out are shaped
        // `[lanes, heads*head_dim]` and partitioned `[QT, head_dim]`, so the launch grid is
        // (lanes/QT, heads) and grid axis 1 is the head; the KV arrives whole and is indexed
        // device-side by `pid.1 / GROUP`, exactly as the SIMT kernel below does it. Binding them as
        // `[lanes*heads, head_dim]` instead collapses that axis to one, pid.1 becomes zero and every
        // CTA reads KV head 0 - finite, crash-free and wrong.
        if std::env::var_os("INFER_CUDA_TILED_ATTENTION").is_some()
            && head_dim == TILED_HEAD_DIM
            && lanes.is_multiple_of(TILED_QUERY_TILE)
            && query_heads.is_multiple_of(kv_heads)
        {
            let scale = 1.0f32 / f32::from(u16::try_from(head_dim).map_err(error)?).sqrt();
            let query = self.input(node.inputs[0])?;
            let query = query.view(&[lanes, query_heads * head_dim])?;
            let mut flat = output.reshape(&[lanes, query_heads * head_dim])?;
            self.scope.record(
                kernels::decode_tiled(
                    (&mut flat).partition([TILED_QUERY_TILE, head_dim]),
                    &query,
                    &keys,
                    &values,
                    self.metadata,
                    i32::try_from(window.unwrap_or(0)).map_err(error)?,
                    scales[0],
                    scales[1],
                    scale,
                )
                .generics(vec![
                    E::DTYPE.as_str().into(),
                    head_dim.to_string(),
                    (query_heads / kv_heads).to_string(),
                    TILED_QUERY_TILE.to_string(),
                    TILED_KEY_BLOCK.to_string(),
                ]),
            )?;
            return Ok((
                flat.reshape(&[lanes * query_heads, head_dim])?,
                keys,
                values,
            ));
        }
        self.scope.record(
            kernels::decode(
                (&mut output).partition([1, head_dim]),
                &self
                    .input(node.inputs[0])?
                    .view(&[lanes * query_heads, head_dim])?,
                &keys,
                &values,
                self.metadata,
                i32::try_from(window.unwrap_or(0)).map_err(error)?,
                scales[0],
                scales[1],
            )
            .generics(vec![
                E::DTYPE.as_str().into(),
                head_dim.to_string(),
                (query_heads / kv_heads).to_string(),
                query_heads.to_string(),
            ]),
        )?;
        Ok((output, keys, values))
    }
}
