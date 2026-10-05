//! Math reference implementation.
#[expect(
    clippy::cast_possible_truncation,
    reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "Floating-point statistics, normalization, and deterministic sampling intentionally convert bounded counts to floating-point values"
)]
pub(super) fn rms_norm(x: &[f32], epsilon: f32) -> Vec<f32> {
    let scale = (x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / x.len() as f64
        + f64::from(epsilon))
    .sqrt();
    x.iter().map(|v| (f64::from(*v) / scale) as f32).collect()
}
#[expect(
    clippy::cast_possible_truncation,
    reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
)]
pub(super) fn matvec(weights: &[f32], x: &[f32]) -> Vec<f32> {
    weights
        .chunks_exact(x.len())
        .map(|row| {
            row.iter()
                .zip(x)
                .map(|(w, x)| f64::from(*w) * f64::from(*x))
                .sum::<f64>() as f32
        })
        .collect()
}
pub(super) fn add_in_place(x: &mut [f32], other: &[f32]) {
    for (x, y) in x.iter_mut().zip(other) {
        *x += y;
    }
}
#[expect(
    clippy::cast_possible_truncation,
    reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "Floating-point statistics, normalization, and deterministic sampling intentionally convert bounded counts to floating-point values"
)]
#[expect(
    clippy::suboptimal_flops,
    reason = "Separate multiply/add rounding preserves the numerical contract of the independent Torch golden and the scalar reference"
)]
pub(super) fn rope(x: &mut [f32], position: usize, theta: f64) {
    let half = x.len() / 2;
    let width = x.len();
    for i in 0..half {
        let angle = position as f64 / theta.powf((2 * i) as f64 / width as f64);
        let (sin, cos) = angle.sin_cos();
        let (a, b) = (f64::from(x[i]), f64::from(x[i + half]));
        x[i] = (a * cos - b * sin) as f32;
        x[i + half] = (a * sin + b * cos) as f32;
    }
}
