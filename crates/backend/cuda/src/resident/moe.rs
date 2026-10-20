//! Device-resident routing and selected-expert projections.
use super::capture::{Capture, error};
use crate::constants::AUX_KERNEL_TILE;
use crate::constants::FP8_BLOCK_COLUMNS;
use crate::mlp::ProjectionWeight;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_ir::{TensorNode, TensorOp};
use std::collections::BTreeMap;
use std::result::Result as StdResult;

/// Matches the fixed output Tensor tile declared by all selected-expert kernel entries.
const EXPERT_PROJECTION_TILE: [usize; 2] = [1, 4];
/// Existing supported routing reduction extent; this is a shape legality bound.
const MAX_ROUTING_EXPERTS: usize = 4096;

/// Contiguous expert-major matrices. Routing indexes this bank on the GPU.
pub struct ExpertBank {
    pub projection: ProjectionWeight,
    pub experts: usize,
    pub rows: usize,
    pub columns: usize,
    /// Follow the checkpoint activation quantization contract for selected experts.
    pub quantized_inputs: bool,
}

impl ExpertBank {
    pub(crate) fn validate(&self, experts: usize, rows: usize, columns: usize) -> Result<()> {
        if (self.experts, self.rows, self.columns) != (experts, rows, columns) {
            return Err(Error::invalid("expert bank geometry"));
        }
        self.projection.validate(
            experts
                .checked_mul(rows)
                .ok_or_else(|| Error::invalid("expert bank overflow"))?,
            columns,
        )
    }
}

impl Capture<'_> {
    pub(super) fn record_moe(
        &self,
        node: &TensorNode,
        output: Tensor<f32>,
    ) -> StdResult<Tensor<f32>, DeviceError> {
        match node.op {
            TensorOp::MoeRoute {
                experts,
                active,
                normalize,
            } => {
                let mut output = output;
                self.scope.record(
                    kernels::route(
                        (&mut output).partition([1]),
                        &self.input(node.inputs[0])?,
                        f32::NEG_INFINITY,
                    )
                    .generics(vec![
                        experts.to_string(),
                        experts.next_power_of_two().to_string(),
                        active.to_string(),
                        i32::from(normalize).to_string(),
                    ]),
                )?;
                Ok(output)
            }
            TensorOp::MoeCombine { active, hidden } => {
                let mut output = output;
                self.scope.record(
                    kernels::combine(
                        (&mut output).partition([AUX_KERNEL_TILE]),
                        &self
                            .input(node.inputs[0])?
                            .view(&[active, hidden])
                            .map_err(error)?,
                        &self.input(node.inputs[1])?,
                    )
                    .generics(vec![active.to_string(), hidden.to_string()]),
                )?;
                Ok(output)
            }
            TensorOp::ExpertLinear {
                experts: _,
                active,
                input,
                output: rows,
                routed_input,
            } => self.record_experts(node, output, active, input, rows, routed_input),
            _ => Err(error("MoE capture operation")),
        }
    }
    fn record_experts(
        &self,
        node: &TensorNode,
        output: Tensor<f32>,
        active: usize,
        input: usize,
        rows: usize,
        routed_input: bool,
    ) -> StdResult<Tensor<f32>, DeviceError> {
        let bank = self
            .weights
            .expert_banks
            .get(&node.inputs[2])
            .ok_or_else(|| error("missing expert bank"))?;
        let x = self.input(node.inputs[0])?;
        let x = x
            .view(&[if routed_input { active } else { 1 }, input])
            .map_err(error)?;
        let routes = self.input(node.inputs[1])?;
        let mut output = output.reshape(&[active, rows])?;
        let mut generics = vec![
            rows.to_string(),
            input.to_string(),
            i32::from(routed_input).to_string(),
        ];
        match &bank.projection {
            ProjectionWeight::Dense(w) => {
                self.scope.record(
                    kernels::dense(
                        (&mut output).partition(EXPERT_PROJECTION_TILE),
                        &x,
                        &routes,
                        &w.view(&[bank.experts, rows, input]).map_err(error)?,
                    )
                    .generics(generics),
                )?;
            }
            ProjectionWeight::Fp8(w, scales) => {
                generics.extend([
                    "1".into(),
                    i32::from(bank.quantized_inputs).to_string(),
                    input.next_power_of_two().to_string(),
                ]);
                self.scope.record(
                    kernels::fp8(
                        (&mut output).partition(EXPERT_PROJECTION_TILE),
                        &x,
                        &routes,
                        &w.view(&[bank.experts, rows, input]).map_err(error)?,
                        &scales.view(&[bank.experts, rows, 1]).map_err(error)?,
                    )
                    .generics(generics),
                )?;
            }
            ProjectionWeight::Fp8Block(w, scales) => {
                generics.extend([
                    "0".into(),
                    i32::from(bank.quantized_inputs).to_string(),
                    input.next_power_of_two().to_string(),
                ]);
                self.scope.record(
                    kernels::fp8(
                        (&mut output).partition(EXPERT_PROJECTION_TILE),
                        &x,
                        &routes,
                        &w.view(&[bank.experts, rows, input]).map_err(error)?,
                        &scales
                            .view(&[bank.experts, rows, input / FP8_BLOCK_COLUMNS])
                            .map_err(error)?,
                    )
                    .generics(generics),
                )?;
            }
            ProjectionWeight::Fp4(..) => {
                return Err(error(
                    "NVFP4 expert activation quantization requires an explicit numerical profile",
                ));
            }
        }
        Ok(output.reshape(&[active * rows])?)
    }
}

pub(super) fn validate_node(
    node: &TensorNode,
    specs: &BTreeMap<infer_core::TensorId, &infer_ir::TensorSpec>,
    weights: &super::ProgramWeights,
    input: usize,
    output: usize,
) -> Result<()> {
    if let TensorOp::ExpertLinear {
        experts,
        active,
        input: columns,
        output: rows,
        routed_input,
    } = node.op
    {
        if experts == 0
            || active == 0
            || active > experts
            || columns == 0
            || rows == 0
            || columns.checked_mul(if routed_input { active } else { 1 }) != Some(input)
            || active.checked_mul(rows) != Some(output)
            || active.checked_mul(2) != Some(specs[&node.inputs[1]].elements()?)
        {
            return Err(Error::invalid("expert projection geometry"));
        }
        weights
            .expert_banks
            .get(&node.inputs[2])
            .ok_or_else(|| Error::invalid("missing expert bank"))?
            .validate(experts, rows, columns)?;
    }
    if let TensorOp::MoeRoute {
        experts, active, ..
    } = node.op
        && (experts == 0
            || experts > MAX_ROUTING_EXPERTS
            || active == 0
            || active > experts
            || input != experts
            || active.checked_mul(2) != Some(output))
    {
        return Err(Error::invalid("routing geometry"));
    }
    if let TensorOp::MoeCombine { active, hidden } = node.op
        && (active == 0
            || hidden == 0
            || active.checked_mul(hidden) != Some(input)
            || output != hidden
            || active.checked_mul(2) != Some(specs[&node.inputs[1]].elements()?))
    {
        return Err(Error::invalid("expert combine geometry"));
    }
    if node.op == TensorOp::Scale && (input != output || specs[&node.inputs[1]].elements()? != 1) {
        return Err(Error::invalid("broadcast scale geometry"));
    }
    Ok(())
}

#[cutile::module]
pub(crate) mod kernels {
    use cutile::core::{
        BroadcastScalar, LoadTileLike, Reshape, Shape_0, Shape_1, Shape_2, Shape_3,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tensor_3, Tile_0, Tile_1, Tile_2, Tile_3,
        absf, bf16, convert_scalar, convert_tile, eq_tile, exp, f8e4m3fn, get_tile_block_id, iota,
        lt_tile, max_tile, min_tile, reduce_max, reduce_sum, select, tile_to_scalar,
    };

    #[cutile::entry()]
    fn route<const EXPERTS: i32, const B: i32, const A: i32, const NORMALIZE: i32>(
        out: &mut Tensor<f32, { [1] }>,
        scores: &Tensor<f32, { [-1] }>,
        excluded: f32,
    ) {
        let pid = get_tile_block_id();
        let ids: Tile<i32, { [B] }> = iota(shape![B]);
        let indices: Tile<f32, { [B] }> = convert_tile(ids);
        let raw = scores.partition(shape![B]).load([0i32]);
        let mut remaining = select(
            lt_tile(ids, EXPERTS.broadcast(shape![B])),
            raw,
            excluded.broadcast(shape![B]),
        );
        let maximum: Tile<f32, { [] }> = reduce_max(remaining, 0i32);
        let probabilities = exp(remaining - maximum.reshape(shape![1]).broadcast(shape![B]));
        let total: Tile<f32, { [] }> = reduce_sum(probabilities, 0i32);
        let mut selected = 0.0f32;
        let mut probability = 0.0f32;
        let mut selected_total = 0.0f32;
        for rank in 0i32..A {
            let best: Tile<f32, { [] }> = reduce_max(remaining, 0i32);
            let candidates = select(
                eq_tile(remaining, best.reshape(shape![1]).broadcast(shape![B])),
                0.0f32.broadcast(shape![B]) - indices,
                excluded.broadcast(shape![B]),
            );
            let negative_id: Tile<f32, { [] }> = reduce_max(candidates, 0i32);
            let negative: f32 = tile_to_scalar(negative_id);
            let expert = 0.0f32 - negative;
            let p: f32 = tile_to_scalar(exp(best - maximum));
            if rank == pid.0 % A {
                selected = expert;
                probability = p;
            }
            selected_total = selected_total + p;
            remaining = select(
                eq_tile(indices, expert.broadcast(shape![B])),
                excluded.broadcast(shape![B]),
                remaining,
            );
        }
        if pid.0 < A {
            out.store(selected.broadcast(shape![1]));
        } else if NORMALIZE == 1 {
            out.store((probability / selected_total).broadcast(shape![1]));
        } else {
            let sum: f32 = tile_to_scalar(total);
            out.store((probability / sum).broadcast(shape![1]));
        }
    }

    #[cutile::entry()]
    fn dense<const O: i32, const I: i32, const ROUTED: i32>(
        out: &mut Tensor<f32, { [1, 4] }>,
        x: &Tensor<f32, { [-1, I] }>,
        routes: &Tensor<f32, { [-1] }>,
        w: &Tensor<bf16, { [-1, O, I] }>,
    ) {
        let pid = get_tile_block_id();
        let id = routes.partition(shape![1]).load([pid.0]).reshape(shape![]);
        let scalar: f32 = tile_to_scalar(id);
        let expert: i32 = convert_scalar(scalar);
        let row = if ROUTED == 1 { pid.0 } else { 0i32 };
        let mut sum: Tile<f32, { [4, 32] }> = 0.0f32.broadcast(shape![4, 32]);
        for k in 0i32..((I + 31) / 32) {
            let a = x.partition(shape![1, 32]).load([row, k]);
            let b: Tile<f32, { [1, 4, 32] }> =
                convert_tile(w.partition(shape![1, 4, 32]).load([expert, pid.1, k]));
            sum = sum + b.reshape(shape![4, 32]) * a.broadcast(shape![4, 32]);
        }
        let reduced: Tile<f32, { [4] }> = reduce_sum(sum, 1i32);
        out.store(reduced.reshape(shape![1, 4]));
    }

    #[cutile::entry()]
    fn fp8<
        const O: i32,
        const I: i32,
        const ROUTED: i32,
        const CHANNEL: i32,
        const QUANT: i32,
        const PAD: i32,
    >(
        out: &mut Tensor<f32, { [1, 4] }>,
        x: &Tensor<f32, { [-1, I] }>,
        routes: &Tensor<f32, { [-1] }>,
        w: &Tensor<f8e4m3fn, { [-1, O, I] }>,
        scales: &Tensor<f32, { [-1, O, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let id = routes.partition(shape![1]).load([pid.0]).reshape(shape![]);
        let scalar: f32 = tile_to_scalar(id);
        let expert: i32 = convert_scalar(scalar);
        let row = if ROUTED == 1 { pid.0 } else { 0i32 };
        let token_scale: Tile<f32, { [1] }> = if QUANT == 1 && CHANNEL == 1 {
            let whole = x.partition(shape![1, PAD]).load([row, 0i32]);
            let maximum: Tile<f32, { [1] }> = reduce_max(absf(whole), 1i32);
            max_tile(
                maximum / 448.0f32.broadcast(shape![1]),
                1.0e-12f32.broadcast(shape![1]),
            )
        } else {
            1.0f32.broadcast(shape![1])
        };
        let mut sum: Tile<f32, { [4] }> = 0.0f32.broadcast(shape![4]);
        for k in 0i32..((I + 127) / 128) {
            let mut a = x.partition(shape![1, 128]).load([row, k]);
            let mut input_scale = token_scale;
            if QUANT == 1 {
                if CHANNEL == 0 {
                    let maximum: Tile<f32, { [1] }> = reduce_max(absf(a), 1i32);
                    input_scale = max_tile(
                        maximum / 448.0f32.broadcast(shape![1]),
                        1.0e-12f32.broadcast(shape![1]),
                    );
                }
                let scale = input_scale.reshape(shape![1, 1]).broadcast(shape![1, 128]);
                let normalized = min_tile(
                    max_tile(a / scale, (-448.0f32).broadcast(shape![1, 128])),
                    448.0f32.broadcast(shape![1, 128]),
                );
                let packed: Tile<f8e4m3fn, { [1, 128] }> = convert_tile(normalized);
                a = convert_tile(packed);
            }
            let b: Tile<f32, { [1, 4, 128] }> =
                convert_tile(w.partition(shape![1, 4, 128]).load([expert, pid.1, k]));
            let value: Tile<f32, { [4] }> = reduce_sum(
                b.reshape(shape![4, 128]) * a.broadcast(shape![4, 128]),
                1i32,
            );
            let scale = scales
                .partition(shape![1, 4, 1])
                .load([expert, pid.1, if CHANNEL == 1 { 0i32 } else { k }])
                .reshape(shape![4]);
            sum = sum + value * scale * input_scale.broadcast(shape![4]);
        }
        out.store(sum.reshape(shape![1, 4]));
    }

    #[cutile::entry()]
    fn combine<const A: i32, const H: i32>(
        out: &mut Tensor<f32, { [256] }>,
        x: &Tensor<f32, { [A, H] }>,
        routes: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let mut sum: Tile<f32, { [256] }> = 0.0f32.broadcast(shape![256]);
        for rank in 0i32..A {
            let p = routes
                .partition(shape![1])
                .load([A + rank])
                .reshape(shape![]);
            let row = x
                .partition(shape![1, 256])
                .load([rank, pid.0])
                .reshape(shape![256]);
            sum = sum + row * p.reshape(shape![1]).broadcast(shape![256]);
        }
        out.store(sum);
    }

    #[cutile::entry()]
    fn scale(
        out: &mut Tensor<f32, { [256] }>,
        x: &Tensor<f32, { [-1] }>,
        scalar: &Tensor<f32, { [-1] }>,
    ) {
        let value = scalar.partition(shape![1]).load([0i32]).reshape(shape![]);
        out.store(x.load_like(out) * value.reshape(shape![1]).broadcast(shape![256]));
    }
}

#[cfg(test)]
#[path = "../../tests/unit/moe.rs"]
mod tests;
