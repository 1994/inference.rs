use super::{
    attention::attention,
    capture::{Capture, CaptureMode, error},
};
use cutile::prelude::*;
use infer_ir::{TensorNode, TensorOp};

type AttentionResult<E> = Result<(Tensor<f32>, Tensor<E>, Tensor<E>), DeviceError>;

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
        let mut output = output.reshape(&[lanes * query_heads, head_dim])?;
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
