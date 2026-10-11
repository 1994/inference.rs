//! Independent scalar `MoE` reference for graph validation.
use infer_core::{Error, Result};

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "Validated expert ids are bounded by the reference input length and exactly representable in F32"
)]
pub fn route(scores: &[f32], experts: usize, active: usize, normalize: bool) -> Result<Vec<f32>> {
    if experts != scores.len()
        || active == 0
        || active > experts
        || scores.iter().any(|x| !x.is_finite())
    {
        return Err(Error::invalid("routing scores"));
    }
    let mut ids: Vec<_> = (0..experts).collect();
    ids.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b)));
    let max = f64::from(scores[ids[0]]);
    let probabilities: Vec<_> = scores.iter().map(|s| (f64::from(*s) - max).exp()).collect();
    let denominator: f64 = if normalize {
        ids[..active].iter().map(|id| probabilities[*id]).sum()
    } else {
        probabilities.iter().sum()
    };
    let mut output: Vec<_> = ids[..active].iter().map(|id| *id as f32).collect();
    output.extend(
        ids[..active]
            .iter()
            .map(|id| (probabilities[*id] / denominator) as f32),
    );
    Ok(output)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Expert ids are checked for finite integral values and range before conversion; F64 sums round to the F32 contract"
)]
pub fn linear(
    inputs: &[&[f32]],
    experts: usize,
    active: usize,
    input: usize,
    output: usize,
    routed: bool,
) -> Result<Vec<f32>> {
    if inputs.len() != 2 + experts
        || inputs[1].len() != 2 * active
        || inputs[0].len() != input * if routed { active } else { 1 }
        || inputs[2..].iter().any(|w| w.len() != input * output)
    {
        return Err(Error::invalid("expert projection binding"));
    }
    let mut result = Vec::new();
    for rank in 0..active {
        let id = inputs[1][rank];
        if !id.is_finite() || id < 0.0 || id.fract() != 0.0 || id as usize >= experts {
            return Err(Error::invalid("expert id"));
        }
        let start = if routed { rank * input } else { 0 };
        let x = &inputs[0][start..start + input];
        for row in inputs[2 + id as usize].chunks_exact(input) {
            result.push(
                row.iter()
                    .zip(x)
                    .map(|(w, x)| f64::from(*w) * f64::from(*x))
                    .sum::<f64>() as f32,
            );
        }
    }
    Ok(result)
}

pub fn combine(x: &[f32], routes: &[f32], active: usize, hidden: usize) -> Result<Vec<f32>> {
    if x.len() != active * hidden || routes.len() != 2 * active {
        return Err(Error::invalid("expert combine binding"));
    }
    let mut result = vec![0.0; hidden];
    for (rank, row) in x.chunks_exact(hidden).enumerate() {
        for (out, value) in result.iter_mut().zip(row) {
            *out += value * routes[active + rank];
        }
    }
    Ok(result)
}

pub fn execute(op: &infer_ir::TensorOp, inputs: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
    use infer_ir::TensorOp;
    let output = match *op {
        TensorOp::MoeRoute {
            experts,
            active,
            normalize,
        } => route(inputs[0], experts, active, normalize)?,
        TensorOp::ExpertLinear {
            experts,
            active,
            input,
            output,
            routed_input,
        } => linear(inputs, experts, active, input, output, routed_input)?,
        TensorOp::MoeCombine { active, hidden } => combine(inputs[0], inputs[1], active, hidden)?,
        TensorOp::Scale => {
            if inputs[1].len() != 1 {
                return Err(Error::invalid("scale binding"));
            }
            inputs[0].iter().map(|x| x * inputs[1][0]).collect()
        }
        _ => return Err(Error::invalid("MoE reference operation")),
    };
    Ok(vec![output])
}
