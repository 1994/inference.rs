//! Independent host F64 scalar recurrence, from original F32 inputs.
use infer_core::{Error, Result};

pub(super) struct Fixture {
    pub shape: [usize; 4],
    pub qkv: Vec<f32>,
    pub beta: Vec<f32>,
    pub alpha: Vec<f32>,
    pub a_log: Vec<f32>,
    pub bias: Vec<f32>,
    pub initial: Vec<f32>,
    pub state: Vec<f64>,
    pub output: Vec<f64>,
}
fn values(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            (f32::from(u16::try_from((i * salt + 3) % 101).unwrap_or_default()) - 50.0) / 100.0
        })
        .collect()
}
impl Fixture {
    pub fn new(shape: [usize; 4], correlated: bool) -> Result<Self> {
        let [kh, vh, dim, lanes] = shape;
        if kh == 0
            || !vh.is_multiple_of(kh)
            || ![32, 128].contains(&dim)
            || lanes == 0
            || lanes > 256
            || !lanes.is_multiple_of(32)
        {
            return Err(Error::invalid("unsupported WY reference geometry"));
        }
        let mut qkv = values(lanes * (2 * kh + vh) * dim, 17);
        if correlated {
            for lane in 0..lanes {
                qkv[lane * (2 * kh + vh) * dim..][..2 * kh * dim].fill(1.0);
            }
        }
        let beta = if correlated {
            vec![8.0; lanes * vh]
        } else {
            values(lanes * vh, 13)
        };
        let alpha = if correlated {
            vec![0.0; lanes * vh]
        } else {
            values(lanes * vh, 7)
        };
        let a_log = if correlated {
            vec![-10.0; vh]
        } else {
            values(vh, 3)
        };
        let bias = if correlated {
            vec![0.0; vh]
        } else {
            values(vh, 5)
        };
        let initial = values(vh * dim * dim, 23);
        let mut state: Vec<f64> = initial.iter().copied().map(f64::from).collect();
        let mut output = vec![0.0; lanes * vh * dim];
        let dim_f64 =
            f64::from(u32::try_from(dim).map_err(|_| Error::invalid("reference dimension"))?);
        for lane in 0..lanes {
            for head in 0..vh {
                let base = lane * (2 * kh + vh) * dim;
                let key = head / (vh / kh);
                let q: Vec<f64> = qkv[base + key * dim..][..dim]
                    .iter()
                    .copied()
                    .map(f64::from)
                    .collect();
                let k: Vec<f64> = qkv[base + (kh + key) * dim..][..dim]
                    .iter()
                    .copied()
                    .map(f64::from)
                    .collect();
                let qnorm = (q.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt() * dim_f64.sqrt();
                let knorm = (k.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt();
                let b = 1.0 / (1.0 + (-f64::from(beta[lane * vh + head])).exp());
                let a = f64::from(alpha[lane * vh + head]) + f64::from(bias[head]);
                let soft = if a > 20.0 { a } else { a.exp().ln_1p() };
                let decay = (-f64::from(a_log[head]).exp() * soft).exp();
                let st = &mut state[head * dim * dim..][..dim * dim];
                for x in st.iter_mut() {
                    *x *= decay;
                }
                for col in 0..dim {
                    let prediction = (0..dim)
                        .map(|row| st[row * dim + col] * k[row] / knorm)
                        .sum::<f64>();
                    let delta =
                        (f64::from(qkv[base + (2 * kh + head) * dim + col]) - prediction) * b;
                    for row in 0..dim {
                        st[row * dim + col] = (k[row] / knorm).mul_add(delta, st[row * dim + col]);
                    }
                    output[(lane * vh + head) * dim + col] = (0..dim)
                        .map(|row| st[row * dim + col] * q[row] / qnorm)
                        .sum();
                }
            }
        }
        Ok(Self {
            shape,
            qkv,
            beta,
            alpha,
            a_log,
            bias,
            initial,
            state,
            output,
        })
    }
}
pub(super) fn maximum_error(actual: &[f32], expected: &[f64]) -> Result<f64> {
    if actual.is_empty()
        || actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(a, b)| !a.is_finite() || !b.is_finite())
    {
        return Err(Error::invalid(
            "WY reference result shape or nonfinite value",
        ));
    }
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (f64::from(*a) - b).abs())
        .fold(0.0_f64, f64::max);
    if error >= 2e-5 {
        return Err(Error::invalid(format!(
            "WY independent F64 error {error} exceeds original 2e-5 gate"
        )));
    }
    Ok(error)
}
