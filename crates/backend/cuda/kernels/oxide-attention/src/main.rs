// SPDX-License-Identifier: MIT
#![allow(static_mut_refs, reason = "CUDA shared arrays are scoped per CTA and synchronized by explicit block barriers")]

// Fixed SM120 geometry and canonical reduction offsets; not user options.
const QUERY_SCALE: f32 = 0.0625;
const INDEX_03: usize = 3;
const INDEX_04: usize = 4;
const INDEX_05: usize = 5;
const INDEX_06: usize = 6;
const INDEX_07: usize = 7;
const STRIPE_LANES: usize = 8;
const INDEX_09: usize = 9;
const INDEX_10: usize = 10;
const INDEX_11: usize = 11;
const INDEX_12: usize = 12;
const INDEX_13: usize = 13;
const INDEX_14: usize = 14;
const INDEX_15: usize = 15;
const DOT_LANES: usize = 16;
const INDEX_17: usize = 17;
const INDEX_18: usize = 18;
const INDEX_19: usize = 19;
const INDEX_20: usize = 20;
const INDEX_21: usize = 21;
const INDEX_22: usize = 22;
const INDEX_23: usize = 23;
const INDEX_24: usize = 24;
const INDEX_25: usize = 25;
const INDEX_26: usize = 26;
const INDEX_27: usize = 27;
const INDEX_28: usize = 28;
const INDEX_29: usize = 29;
const INDEX_30: usize = 30;
const INDEX_31: usize = 31;
const WARP_SIZE: usize = 32;
const FP8_EXPONENT_BIAS: u32 = 120;
const FP8_MAGNITUDE_MASK: u32 = 127;
const FP8_SIGN_BIT: u32 = 128;
const HEAD_DIM: usize = 256;
const FP8_SUBNORMAL_DIVISOR: f32 = 512.0f32;
const MASKED_SCORE_MAGNITUDE: f32 = 1.0e30;
use cuda_device::{SharedArray, kernel, thread, warp};
use cuda_host::cuda_module;
#[cuda_module]
mod kernels {
    use super::*;
    #[inline(always)]
    fn fp8(x: u8) -> f32 {
        let sign = (x as u32 & FP8_SIGN_BIT) << INDEX_24;
        let exponent = (x as u32 & FP8_MAGNITUDE_MASK) >> INDEX_03;
        let mantissa = x as u32 & INDEX_07 as u32;
        if exponent == 0 {
            let magnitude = mantissa as f32 * (1.0f32 / FP8_SUBNORMAL_DIVISOR);
            if sign == 0 { magnitude } else { -magnitude }
        } else {
            f32::from_bits(sign | ((exponent + FP8_EXPONENT_BIAS) << INDEX_23) | (mantissa << INDEX_20))
        }
    }
    #[inline(always)]
    fn sum8(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> f32 {
        ((a + b) + (c + d)) + ((e + f) + (g + h))
    }
    /// Qwen geometry: 24 query heads, four KV heads, dimension 256.
    /// Launch exactly 256 threads per query/head, with one CTA per output row.
    #[kernel]
    pub unsafe fn attention_fp8_256(
        query: *const f32,
        keys: *const u8,
        values: *const u8,
        metadata: *const i32,
        table: *const i32,
        out: *mut f32,
        capacity: u32,
        window: i32,
        k_scale: f32,
        v_scale: f32,
    ) {
        static mut SCORES: SharedArray<f32, WARP_SIZE> = SharedArray::UNINIT;
        static mut PROBS: SharedArray<f32, WARP_SIZE> = SharedArray::UNINIT;
        static mut STATE: SharedArray<f32, INDEX_04> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let warp_id = tid / WARP_SIZE;
        let lane = tid % WARP_SIZE;
        let row = thread::blockIdx_x() as usize;
        let query_lane = row / INDEX_24;
        let head = row % INDEX_24;
        let position = unsafe { *metadata } + query_lane as i32 + unsafe { *metadata.add(2) };
        let count = unsafe { *metadata.add(1) };
        if query_lane as i32 >= count || position < 0 {
            unsafe {
                *out.add(row * HEAD_DIM + tid) = 0.0;
            }
            return;
        }
        let start = if window > 0 && position + 1 > window {
            position + 1 - window
        } else {
            0
        };
        let dot_lane = lane & INDEX_15;
        let qbase = row * HEAD_DIM + dot_lane * DOT_LANES;
        let q0 = unsafe { *query.add(qbase + 0) };
        let q1 = unsafe { *query.add(qbase + 1) };
        let q2 = unsafe { *query.add(qbase + 2) };
        let q3 = unsafe { *query.add(qbase + INDEX_03) };
        let q4 = unsafe { *query.add(qbase + INDEX_04) };
        let q5 = unsafe { *query.add(qbase + INDEX_05) };
        let q6 = unsafe { *query.add(qbase + INDEX_06) };
        let q7 = unsafe { *query.add(qbase + INDEX_07) };
        let q8 = unsafe { *query.add(qbase + STRIPE_LANES) };
        let q9 = unsafe { *query.add(qbase + INDEX_09) };
        let q10 = unsafe { *query.add(qbase + INDEX_10) };
        let q11 = unsafe { *query.add(qbase + INDEX_11) };
        let q12 = unsafe { *query.add(qbase + INDEX_12) };
        let q13 = unsafe { *query.add(qbase + INDEX_13) };
        let q14 = unsafe { *query.add(qbase + INDEX_14) };
        let q15 = unsafe { *query.add(qbase + INDEX_15) };
        if tid == 0 {
            unsafe {
                STATE[0] = -MASKED_SCORE_MAGNITUDE;
                STATE[1] = 0.0;
            }
        }
        thread::sync_threads();
        let mut numerator = 0.0f32;
        let mut block = start / WARP_SIZE as i32;
        while block < position / WARP_SIZE as i32 + 1 {
            let physical = unsafe { *table.add(block as usize) } as usize;
            let cachebase = (head / INDEX_06 * capacity as usize + physical * WARP_SIZE) * HEAD_DIM;
            let mut keyrow = warp_id * 2 + lane / DOT_LANES;
            while keyrow < WARP_SIZE {
                let kb = cachebase + keyrow * HEAD_DIM + dot_lane * DOT_LANES;
                let p0 = q0 * (fp8(unsafe { *keys.add(kb + 0) }) * k_scale);
                let p1 = q1 * (fp8(unsafe { *keys.add(kb + 1) }) * k_scale);
                let p2 = q2 * (fp8(unsafe { *keys.add(kb + 2) }) * k_scale);
                let p3 = q3 * (fp8(unsafe { *keys.add(kb + INDEX_03) }) * k_scale);
                let p4 = q4 * (fp8(unsafe { *keys.add(kb + INDEX_04) }) * k_scale);
                let p5 = q5 * (fp8(unsafe { *keys.add(kb + INDEX_05) }) * k_scale);
                let p6 = q6 * (fp8(unsafe { *keys.add(kb + INDEX_06) }) * k_scale);
                let p7 = q7 * (fp8(unsafe { *keys.add(kb + INDEX_07) }) * k_scale);
                let p8 = q8 * (fp8(unsafe { *keys.add(kb + STRIPE_LANES) }) * k_scale);
                let p9 = q9 * (fp8(unsafe { *keys.add(kb + INDEX_09) }) * k_scale);
                let p10 = q10 * (fp8(unsafe { *keys.add(kb + INDEX_10) }) * k_scale);
                let p11 = q11 * (fp8(unsafe { *keys.add(kb + INDEX_11) }) * k_scale);
                let p12 = q12 * (fp8(unsafe { *keys.add(kb + INDEX_12) }) * k_scale);
                let p13 = q13 * (fp8(unsafe { *keys.add(kb + INDEX_13) }) * k_scale);
                let p14 = q14 * (fp8(unsafe { *keys.add(kb + INDEX_14) }) * k_scale);
                let p15 = q15 * (fp8(unsafe { *keys.add(kb + INDEX_15) }) * k_scale);
                let even = ((p0 + p2) + (p4 + p6)) + ((p8 + p10) + (p12 + p14));
                let odd = ((p1 + p3) + (p5 + p7)) + ((p9 + p11) + (p13 + p15));
                let mut score = even + odd;
                score += warp::shuffle_xor_f32(score, STRIPE_LANES as u32);
                score += warp::shuffle_xor_f32(score, INDEX_04 as u32);
                score += warp::shuffle_xor_f32(score, 2);
                score += warp::shuffle_xor_f32(score, 1);
                if dot_lane == 0 {
                    let keyposition = block * WARP_SIZE as i32 + keyrow as i32;
                    unsafe {
                        SCORES[keyrow] = if keyposition >= start && keyposition <= position {
                            score * QUERY_SCALE
                        } else {
                            -MASKED_SCORE_MAGNITUDE
                        };
                    }
                }
                keyrow += DOT_LANES;
            }
            thread::sync_threads();
            if warp_id == 0 {
                let score = unsafe { SCORES[lane] };
                let mut maximum = score;
                maximum = maximum.max(warp::shuffle_xor_f32(maximum, 1));
                maximum = maximum.max(warp::shuffle_xor_f32(maximum, 2));
                maximum = maximum.max(warp::shuffle_xor_f32(maximum, INDEX_04 as u32));
                maximum = maximum.max(warp::shuffle_xor_f32(maximum, STRIPE_LANES as u32));
                maximum = maximum.max(warp::shuffle_xor_f32(maximum, DOT_LANES as u32));
                let old = unsafe { STATE[0] };
                let next = if maximum > old { maximum } else { old };
                let old_scale = (old - next).exp();
                let probability = (score - next).exp();
                unsafe {
                    PROBS[lane] = probability;
                }
                let local_lane = lane as u32 & INDEX_07 as u32;
                let mut sum = warp::shuffle_f32(probability, local_lane);
                sum += warp::shuffle_f32(probability, local_lane + STRIPE_LANES as u32);
                sum += warp::shuffle_f32(probability, local_lane + DOT_LANES as u32);
                sum += warp::shuffle_f32(probability, local_lane + INDEX_24 as u32);
                sum += warp::shuffle_xor_f32(sum, 1);
                sum += warp::shuffle_xor_f32(sum, INDEX_04 as u32);
                sum += warp::shuffle_xor_f32(sum, 2);
                if lane == 0 {
                    unsafe {
                        STATE[0] = next;
                        STATE[1] = STATE[1] * old_scale + sum;
                        STATE[2] = old_scale;
                    }
                }
            }
            thread::sync_threads();
            let v0 = fp8(unsafe { *values.add(cachebase + 0 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[0] };
            let v1 = fp8(unsafe { *values.add(cachebase + 1 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[1] };
            let v2 = fp8(unsafe { *values.add(cachebase + 2 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[2] };
            let v3 = fp8(unsafe { *values.add(cachebase + INDEX_03 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_03] };
            let v4 = fp8(unsafe { *values.add(cachebase + INDEX_04 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_04] };
            let v5 = fp8(unsafe { *values.add(cachebase + INDEX_05 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_05] };
            let v6 = fp8(unsafe { *values.add(cachebase + INDEX_06 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_06] };
            let v7 = fp8(unsafe { *values.add(cachebase + INDEX_07 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_07] };
            let v8 = fp8(unsafe { *values.add(cachebase + STRIPE_LANES * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[STRIPE_LANES] };
            let v9 = fp8(unsafe { *values.add(cachebase + INDEX_09 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_09] };
            let v10 = fp8(unsafe { *values.add(cachebase + INDEX_10 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_10] };
            let v11 = fp8(unsafe { *values.add(cachebase + INDEX_11 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_11] };
            let v12 = fp8(unsafe { *values.add(cachebase + INDEX_12 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_12] };
            let v13 = fp8(unsafe { *values.add(cachebase + INDEX_13 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_13] };
            let v14 = fp8(unsafe { *values.add(cachebase + INDEX_14 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_14] };
            let v15 = fp8(unsafe { *values.add(cachebase + INDEX_15 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_15] };
            let v16 = fp8(unsafe { *values.add(cachebase + DOT_LANES * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[DOT_LANES] };
            let v17 = fp8(unsafe { *values.add(cachebase + INDEX_17 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_17] };
            let v18 = fp8(unsafe { *values.add(cachebase + INDEX_18 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_18] };
            let v19 = fp8(unsafe { *values.add(cachebase + INDEX_19 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_19] };
            let v20 = fp8(unsafe { *values.add(cachebase + INDEX_20 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_20] };
            let v21 = fp8(unsafe { *values.add(cachebase + INDEX_21 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_21] };
            let v22 = fp8(unsafe { *values.add(cachebase + INDEX_22 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_22] };
            let v23 = fp8(unsafe { *values.add(cachebase + INDEX_23 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_23] };
            let v24 = fp8(unsafe { *values.add(cachebase + INDEX_24 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_24] };
            let v25 = fp8(unsafe { *values.add(cachebase + INDEX_25 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_25] };
            let v26 = fp8(unsafe { *values.add(cachebase + INDEX_26 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_26] };
            let v27 = fp8(unsafe { *values.add(cachebase + INDEX_27 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_27] };
            let v28 = fp8(unsafe { *values.add(cachebase + INDEX_28 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_28] };
            let v29 = fp8(unsafe { *values.add(cachebase + INDEX_29 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_29] };
            let v30 = fp8(unsafe { *values.add(cachebase + INDEX_30 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_30] };
            let v31 = fp8(unsafe { *values.add(cachebase + INDEX_31 * HEAD_DIM + tid) })
                * v_scale
                * unsafe { PROBS[INDEX_31] };
            let g0 = ((v0 + v8) + v16) + v24;
            let g1 = ((v1 + v9) + v17) + v25;
            let g2 = ((v2 + v10) + v18) + v26;
            let g3 = ((v3 + v11) + v19) + v27;
            let g4 = ((v4 + v12) + v20) + v28;
            let g5 = ((v5 + v13) + v21) + v29;
            let g6 = ((v6 + v14) + v22) + v30;
            let g7 = ((v7 + v15) + v23) + v31;
            let contribution = ((g0 + g1) + (g4 + g5)) + ((g2 + g3) + (g6 + g7));
            numerator = numerator * unsafe { STATE[2] } + contribution;
            thread::sync_threads();
            block += 1;
        }
        unsafe {
            *out.add(row * HEAD_DIM + tid) = numerator / STATE[1];
        }
    }
}
fn main() {}
