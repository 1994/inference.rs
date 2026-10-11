//! Sparse `SwiGLU` with optional sigmoid-gated shared expert.
use crate::decoder::Builder;
use infer_core::{Error, Result, TensorId};
use infer_ir::{FeedForward, ModelIr, TensorOp};

impl Builder {
    pub(crate) fn moe(&mut self, x: TensorId, prefix: &str, model: &ModelIr) -> Result<TensorId> {
        let FeedForward::Moe {
            experts,
            active,
            intermediate,
            normalize_topk,
            shared_intermediate,
            ..
        } = model.feed_forward
        else {
            return Err(Error::invariant("MoE recipe"));
        };
        let hidden = model.hidden_size;
        let norm = self.norm(
            x,
            format!("{prefix}.post_attention_layernorm.weight"),
            hidden,
            hidden,
            model,
        )?;
        let scores = self.linear(norm, format!("{prefix}.mlp.gate.weight"), hidden, experts)?;
        let routes = self.one(
            TensorOp::MoeRoute {
                experts,
                active,
                normalize: normalize_topk,
            },
            vec![scores],
            active
                .checked_mul(2)
                .ok_or_else(|| Error::invalid("routing overflow"))?,
        )?;
        let gate = self.expert_linear(
            (norm, routes),
            prefix,
            "gate_proj",
            (hidden, intermediate),
            (experts, active),
            false,
        )?;
        let up = self.expert_linear(
            (norm, routes),
            prefix,
            "up_proj",
            (hidden, intermediate),
            (experts, active),
            false,
        )?;
        let width = active
            .checked_mul(intermediate)
            .ok_or_else(|| Error::invalid("expert activation overflow"))?;
        let gate = self.one(TensorOp::Silu, vec![gate], width)?;
        let product = self.one(TensorOp::Multiply, vec![gate, up], width)?;
        let down = self.expert_linear(
            (product, routes),
            prefix,
            "down_proj",
            (intermediate, hidden),
            (experts, active),
            true,
        )?;
        let mut output = self.one(
            TensorOp::MoeCombine { active, hidden },
            vec![down, routes],
            hidden,
        )?;
        if let Some(size) = shared_intermediate {
            let shared = self.shared_expert(norm, prefix, hidden, size)?;
            output = self.one(TensorOp::Add, vec![output, shared], hidden)?;
        }
        self.one(TensorOp::Add, vec![x, output], hidden)
    }

    fn shared_expert(
        &mut self,
        norm: TensorId,
        prefix: &str,
        hidden: usize,
        size: usize,
    ) -> Result<TensorId> {
        let shared_prefix = format!("{prefix}.mlp.shared_expert");
        let gate = self.linear(
            norm,
            format!("{shared_prefix}.gate_proj.weight"),
            hidden,
            size,
        )?;
        let up = self.linear(
            norm,
            format!("{shared_prefix}.up_proj.weight"),
            hidden,
            size,
        )?;
        let gate = self.one(TensorOp::Silu, vec![gate], size)?;
        let product = self.one(TensorOp::Multiply, vec![gate, up], size)?;
        let shared = self.linear(
            product,
            format!("{shared_prefix}.down_proj.weight"),
            size,
            hidden,
        )?;
        let gate = self.linear(
            norm,
            format!("{prefix}.mlp.shared_expert_gate.weight"),
            hidden,
            1,
        )?;
        let gate = self.one(TensorOp::Sigmoid, vec![gate], 1)?;
        // Broadcast is explicit in the graph and shared by every backend.
        self.one(TensorOp::Scale, vec![shared, gate], hidden)
    }

    fn expert_linear(
        &mut self,
        tensors: (TensorId, TensorId),
        prefix: &str,
        projection: &str,
        dimensions: (usize, usize),
        routing: (usize, usize),
        routed_input: bool,
    ) -> Result<TensorId> {
        let (x, routes) = tensors;
        let (input, output) = dimensions;
        let (experts, active) = routing;
        let mut inputs = vec![x, routes];
        for expert in 0..experts {
            inputs.push(self.weight(
                format!("{prefix}.mlp.experts.{expert}.{projection}.weight"),
                vec![output, input],
            )?);
        }
        self.one(
            TensorOp::ExpertLinear {
                experts,
                active,
                input,
                output,
                routed_input,
            },
            inputs,
            active
                .checked_mul(output)
                .ok_or_else(|| Error::invalid("expert output overflow"))?,
        )
    }
}
