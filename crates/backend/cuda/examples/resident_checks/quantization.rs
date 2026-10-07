//! Independent nearest-even E4M3FN reference, enumerating all positive finite codes.
pub fn round(values: &[f32], scale: f32) -> Vec<f32> {
    values
        .iter()
        .map(|value| {
            let target = (value / scale).abs().min(448.0);
            let mut nearest = 0.0f32;
            let mut distance = target;
            for code in 1u8..=126 {
                let exponent = code >> 3;
                let mantissa = f32::from(code & 7);
                let candidate = if exponent == 0 {
                    mantissa / 512.0
                } else {
                    (1.0 + mantissa / 8.0) * 2.0f32.powi(i32::from(exponent) - 7)
                };
                let difference = (candidate - target).abs();
                let ordering = difference.total_cmp(&distance);
                if ordering.is_lt() || (ordering.is_eq() && code.is_multiple_of(2)) {
                    nearest = candidate;
                    distance = difference;
                }
            }
            nearest.copysign(*value) * scale
        })
        .collect()
}
