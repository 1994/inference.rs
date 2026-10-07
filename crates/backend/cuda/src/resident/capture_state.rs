use super::{
    capture::{Capture, error},
    recurrent::recurrent,
};
use cutile::prelude::*;
use infer_ir::{TensorNode, TensorOp};

impl Capture<'_> {
    fn conv(
        &self,
        node: &TensorNode,
        output: &mut Tensor<f32>,
        state: &mut [Tensor<f32>],
        channels: usize,
    ) -> Result<(), DeviceError> {
        let [h0, h1, h2] = state else {
            return Err(error("convolution state layout"));
        };
        self.scope.record(recurrent::conv4(
            output.partition([crate::constants::CONV_KERNEL_TILE]),
            h0.partition([crate::constants::CONV_KERNEL_TILE]),
            h1.partition([crate::constants::CONV_KERNEL_TILE]),
            h2.partition([crate::constants::CONV_KERNEL_TILE]),
            &self.input(node.inputs[0])?,
            &self
                .input(node.inputs[1])?
                .view(&[channels, crate::constants::CONV_KERNEL_SIZE])?,
            self.metadata,
        ))?;
        Ok(())
    }

    pub fn record_state(
        &mut self,
        node: &TensorNode,
        mut output: Tensor<f32>,
    ) -> Result<Tensor<f32>, DeviceError> {
        let id = *node
            .states
            .first()
            .ok_or_else(|| error("missing state binding"))?;
        if let Some((keys, values)) = self.fp8_states.remove(&id) {
            let (output, keys, values) =
                self.record_attention(node, output, keys, values, self.weights.kv_scales[&id])?;
            self.fp8_states.insert(id, (keys, values));
            return Ok(output);
        }
        let mut state = self
            .states
            .remove(&id)
            .ok_or_else(|| error("missing state allocation"))?;
        match node.op {
            TensorOp::Conv {
                channels,
                kernel: crate::constants::CONV_KERNEL_SIZE,
            } => {
                self.conv(node, &mut output, &mut state, channels)?;
            }
            TensorOp::Delta {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
            } => {
                if key_dim != value_dim
                    || !key_dim.is_power_of_two()
                    || key_heads == 0
                    || !value_heads.is_multiple_of(key_heads)
                {
                    return Err(error("resident delta dimensions"));
                }
                output = output.reshape(&[value_heads, value_dim])?;
                self.scope.record(
                    recurrent::delta(
                        (&mut output).partition([1, value_dim]),
                        (&mut state[0]).partition([1, key_dim, value_dim]),
                        &self.input(node.inputs[0])?,
                        &self.input(node.inputs[1])?,
                        &self.input(node.inputs[2])?,
                        &self.input(node.inputs[3])?,
                        &self.input(node.inputs[4])?,
                        self.metadata,
                    )
                    .generics(vec![
                        key_heads.to_string(),
                        value_heads.to_string(),
                        key_dim.to_string(),
                    ]),
                )?;
            }
            TensorOp::Attention { .. } => {
                let keys = state.remove(0);
                let values = state.remove(0);
                let (result, keys, values) =
                    self.record_attention(node, output, keys, values, [1.0, 1.0])?;
                output = result;
                state.push(keys);
                state.push(values);
            }
            _ => return Err(error("unsupported resident state operation")),
        }
        self.states.insert(id, state);
        Ok(output)
    }
}
