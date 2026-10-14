// SPDX-License-Identifier: MIT
#![allow(static_mut_refs)]
use cuda_device::{SharedArray, kernel, thread, warp};
use cuda_host::cuda_module;
#[cuda_module]
mod kernels {
    use super::*;
    #[inline(always)]
    fn fp8(x: u8) -> f32 {
        let sign = (x as u32 & 128) << 24;
        let exponent = (x as u32 & 127) >> 3;
        let mantissa = x as u32 & 7;
        if exponent == 0 {
            let magnitude = mantissa as f32 * (1.0f32 / 512.0f32);
            if sign == 0 { magnitude } else { -magnitude }
        } else {
            f32::from_bits(sign | ((exponent + 120) << 23) | (mantissa << 20))
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
        static mut SCORES: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut PROBS: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut STATE: SharedArray<f32, 4> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let warp_id = tid / 32;
        let lane = tid % 32;
        let row = thread::blockIdx_x() as usize;
        let query_lane = row / 24;
        let head = row % 24;
        let position = unsafe { *metadata } + query_lane as i32 + unsafe { *metadata.add(2) };
        let count = unsafe { *metadata.add(1) };
        if query_lane as i32 >= count || position < 0 {
            unsafe {
                *out.add(row * 256 + tid) = 0.0;
            }
            return;
        }
        let start = if window > 0 && position + 1 > window {
            position + 1 - window
        } else {
            0
        };
        let dot_lane = lane & 15;
        let qbase = row * 256 + dot_lane * 16;
        let q0 = unsafe { *query.add(qbase + 0) };
        let q1 = unsafe { *query.add(qbase + 1) };
        let q2 = unsafe { *query.add(qbase + 2) };
        let q3 = unsafe { *query.add(qbase + 3) };
        let q4 = unsafe { *query.add(qbase + 4) };
        let q5 = unsafe { *query.add(qbase + 5) };
        let q6 = unsafe { *query.add(qbase + 6) };
        let q7 = unsafe { *query.add(qbase + 7) };
        let q8 = unsafe { *query.add(qbase + 8) };
        let q9 = unsafe { *query.add(qbase + 9) };
        let q10 = unsafe { *query.add(qbase + 10) };
        let q11 = unsafe { *query.add(qbase + 11) };
        let q12 = unsafe { *query.add(qbase + 12) };
        let q13 = unsafe { *query.add(qbase + 13) };
        let q14 = unsafe { *query.add(qbase + 14) };
        let q15 = unsafe { *query.add(qbase + 15) };
        if tid == 0 {
            unsafe {
                STATE[0] = -1.0e30;
                STATE[1] = 0.0;
            }
        }
        thread::sync_threads();
        let mut numerator = 0.0f32;
        let mut block = start / 32;
        while block < position / 32 + 1 {
            let physical = unsafe { *table.add(block as usize) } as usize;
            let cachebase = (head / 6 * capacity as usize + physical * 32) * 256;
            let mut keyrow = warp_id * 2 + lane / 16;
            while keyrow < 32 {
                let kb = cachebase + keyrow * 256 + dot_lane * 16;
                let p0 = q0 * (fp8(unsafe { *keys.add(kb + 0) }) * k_scale);
                let p1 = q1 * (fp8(unsafe { *keys.add(kb + 1) }) * k_scale);
                let p2 = q2 * (fp8(unsafe { *keys.add(kb + 2) }) * k_scale);
                let p3 = q3 * (fp8(unsafe { *keys.add(kb + 3) }) * k_scale);
                let p4 = q4 * (fp8(unsafe { *keys.add(kb + 4) }) * k_scale);
                let p5 = q5 * (fp8(unsafe { *keys.add(kb + 5) }) * k_scale);
                let p6 = q6 * (fp8(unsafe { *keys.add(kb + 6) }) * k_scale);
                let p7 = q7 * (fp8(unsafe { *keys.add(kb + 7) }) * k_scale);
                let p8 = q8 * (fp8(unsafe { *keys.add(kb + 8) }) * k_scale);
                let p9 = q9 * (fp8(unsafe { *keys.add(kb + 9) }) * k_scale);
                let p10 = q10 * (fp8(unsafe { *keys.add(kb + 10) }) * k_scale);
                let p11 = q11 * (fp8(unsafe { *keys.add(kb + 11) }) * k_scale);
                let p12 = q12 * (fp8(unsafe { *keys.add(kb + 12) }) * k_scale);
                let p13 = q13 * (fp8(unsafe { *keys.add(kb + 13) }) * k_scale);
                let p14 = q14 * (fp8(unsafe { *keys.add(kb + 14) }) * k_scale);
                let p15 = q15 * (fp8(unsafe { *keys.add(kb + 15) }) * k_scale);
                let even = ((p0 + p2) + (p4 + p6)) + ((p8 + p10) + (p12 + p14));
                let odd = ((p1 + p3) + (p5 + p7)) + ((p9 + p11) + (p13 + p15));
                let mut score = even + odd;
                score += warp::shuffle_xor_f32(score, 8);
                score += warp::shuffle_xor_f32(score, 4);
                score += warp::shuffle_xor_f32(score, 2);
                score += warp::shuffle_xor_f32(score, 1);
                if dot_lane == 0 {
                    let keyposition = block * 32 + keyrow as i32;
                    unsafe {
                        SCORES[keyrow] = if keyposition >= start && keyposition <= position {
                            score * 0.0625
                        } else {
                            -1.0e30
                        };
                    }
                }
                keyrow += 16;
            }
            thread::sync_threads();
            if warp_id == 0 {
                let score = unsafe { SCORES[lane] };
                let mut maximum = score;
                maximum = maximum.max(warp::shuffle_xor_f32(maximum, 1));
                maximum = maximum.max(warp::shuffle_xor_f32(maximum, 2));
                maximum = maximum.max(warp::shuffle_xor_f32(maximum, 4));
                maximum = maximum.max(warp::shuffle_xor_f32(maximum, 8));
                maximum = maximum.max(warp::shuffle_xor_f32(maximum, 16));
                let old = unsafe { STATE[0] };
                let next = if maximum > old { maximum } else { old };
                let old_scale = (old - next).exp();
                let probability = (score - next).exp();
                unsafe {
                    PROBS[lane] = probability;
                }
                let local_lane = lane as u32 & 7;
                let mut sum = warp::shuffle_f32(probability, local_lane);
                sum += warp::shuffle_f32(probability, local_lane + 8);
                sum += warp::shuffle_f32(probability, local_lane + 16);
                sum += warp::shuffle_f32(probability, local_lane + 24);
                sum += warp::shuffle_xor_f32(sum, 1);
                sum += warp::shuffle_xor_f32(sum, 4);
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
            let v0 = fp8(unsafe { *values.add(cachebase + 0 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[0] };
            let v1 = fp8(unsafe { *values.add(cachebase + 1 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[1] };
            let v2 = fp8(unsafe { *values.add(cachebase + 2 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[2] };
            let v3 = fp8(unsafe { *values.add(cachebase + 3 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[3] };
            let v4 = fp8(unsafe { *values.add(cachebase + 4 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[4] };
            let v5 = fp8(unsafe { *values.add(cachebase + 5 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[5] };
            let v6 = fp8(unsafe { *values.add(cachebase + 6 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[6] };
            let v7 = fp8(unsafe { *values.add(cachebase + 7 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[7] };
            let v8 = fp8(unsafe { *values.add(cachebase + 8 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[8] };
            let v9 = fp8(unsafe { *values.add(cachebase + 9 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[9] };
            let v10 = fp8(unsafe { *values.add(cachebase + 10 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[10] };
            let v11 = fp8(unsafe { *values.add(cachebase + 11 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[11] };
            let v12 = fp8(unsafe { *values.add(cachebase + 12 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[12] };
            let v13 = fp8(unsafe { *values.add(cachebase + 13 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[13] };
            let v14 = fp8(unsafe { *values.add(cachebase + 14 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[14] };
            let v15 = fp8(unsafe { *values.add(cachebase + 15 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[15] };
            let v16 = fp8(unsafe { *values.add(cachebase + 16 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[16] };
            let v17 = fp8(unsafe { *values.add(cachebase + 17 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[17] };
            let v18 = fp8(unsafe { *values.add(cachebase + 18 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[18] };
            let v19 = fp8(unsafe { *values.add(cachebase + 19 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[19] };
            let v20 = fp8(unsafe { *values.add(cachebase + 20 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[20] };
            let v21 = fp8(unsafe { *values.add(cachebase + 21 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[21] };
            let v22 = fp8(unsafe { *values.add(cachebase + 22 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[22] };
            let v23 = fp8(unsafe { *values.add(cachebase + 23 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[23] };
            let v24 = fp8(unsafe { *values.add(cachebase + 24 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[24] };
            let v25 = fp8(unsafe { *values.add(cachebase + 25 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[25] };
            let v26 = fp8(unsafe { *values.add(cachebase + 26 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[26] };
            let v27 = fp8(unsafe { *values.add(cachebase + 27 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[27] };
            let v28 = fp8(unsafe { *values.add(cachebase + 28 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[28] };
            let v29 = fp8(unsafe { *values.add(cachebase + 29 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[29] };
            let v30 = fp8(unsafe { *values.add(cachebase + 30 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[30] };
            let v31 = fp8(unsafe { *values.add(cachebase + 31 * 256 + tid) })
                * v_scale
                * unsafe { PROBS[31] };
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
            *out.add(row * 256 + tid) = numerator / STATE[1];
        }
    }
}
fn main() {}
