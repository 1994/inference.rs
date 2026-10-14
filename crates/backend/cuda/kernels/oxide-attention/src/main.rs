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
use cuda_device::{SharedArray, kernel, thread, warp, convert};
use cuda_host::cuda_module;
#[cuda_module]
mod kernels {
    use super::*;
    #[inline(always)]
    fn fp8_original(x: u8) -> f32 {
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
    fn fp8(x: u8) -> f32 {
        // The original decoder maps the two reserved NaN codes to signed 480.
        // Preserve that behavior while finite codes use exact hardware widening.
        if u32::from(x) & FP8_MAGNITUDE_MASK == FP8_MAGNITUDE_MASK {
            fp8_original(x)
        } else {
            convert::cvt_f32_f16x2_lo(convert::cvt_rn_f16x2_e4m3x2(u16::from(x)))
        }
    }
    #[inline(always)]
    fn fp8_pair_bits(packed: u16) -> (f32,f32) {
        let (mut lo,mut hi)=convert::cvt_f32x2_f16x2(convert::cvt_rn_f16x2_e4m3x2(packed));
        let a=packed as u8;
        let b=(packed >> STRIPE_LANES) as u8;
        if u32::from(a) & FP8_MAGNITUDE_MASK == FP8_MAGNITUDE_MASK {lo=fp8_original(a);}
        if u32::from(b) & FP8_MAGNITUDE_MASK == FP8_MAGNITUDE_MASK {hi=fp8_original(b);}
        (lo,hi)
    }
    #[inline(always)]
    fn sum8(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> f32 {
        ((a + b) + (c + d)) + ((e + f) + (g + h))
    }
    /// Qwen geometry: 24 query heads, four KV heads, dimension 256.
    /// Launch exactly 128 threads per query/head, with one CTA per output row.
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
                *out.add(row * HEAD_DIM + tid*2) = 0.0;
                *out.add(row * HEAD_DIM + tid*2+1) = 0.0;
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
        let mut numerator_lo = 0.0f32;
        let mut numerator_hi = 0.0f32;
        let mut block = start / WARP_SIZE as i32;
        while block < position / WARP_SIZE as i32 + 1 {
            let physical = unsafe { *table.add(block as usize) } as usize;
            let cachebase = (head / INDEX_06 * capacity as usize + physical * WARP_SIZE) * HEAD_DIM;
            let mut keyrow = warp_id * 2 + lane / DOT_LANES;
            while keyrow < WARP_SIZE {
                let kb = cachebase + keyrow * HEAD_DIM + dot_lane * DOT_LANES;
                let (k0,k1)=fp8_pair_bits(unsafe { *keys.add(kb + 0).cast::<u16>() });
                let p0=q0*(k0*k_scale);
                let p1=q1*(k1*k_scale);
                let (k2,k3)=fp8_pair_bits(unsafe { *keys.add(kb + 2).cast::<u16>() });
                let p2=q2*(k2*k_scale);
                let p3=q3*(k3*k_scale);
                let (k4,k5)=fp8_pair_bits(unsafe { *keys.add(kb + INDEX_04).cast::<u16>() });
                let p4=q4*(k4*k_scale);
                let p5=q5*(k5*k_scale);
                let (k6,k7)=fp8_pair_bits(unsafe { *keys.add(kb + INDEX_06).cast::<u16>() });
                let p6=q6*(k6*k_scale);
                let p7=q7*(k7*k_scale);
                let (k8,k9)=fp8_pair_bits(unsafe { *keys.add(kb + STRIPE_LANES).cast::<u16>() });
                let p8=q8*(k8*k_scale);
                let p9=q9*(k9*k_scale);
                let (k10,k11)=fp8_pair_bits(unsafe { *keys.add(kb + INDEX_10).cast::<u16>() });
                let p10=q10*(k10*k_scale);
                let p11=q11*(k11*k_scale);
                let (k12,k13)=fp8_pair_bits(unsafe { *keys.add(kb + INDEX_12).cast::<u16>() });
                let p12=q12*(k12*k_scale);
                let p13=q13*(k13*k_scale);
                let (k14,k15)=fp8_pair_bits(unsafe { *keys.add(kb + INDEX_14).cast::<u16>() });
                let p14=q14*(k14*k_scale);
                let p15=q15*(k15*k_scale);
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
                keyrow += STRIPE_LANES;
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
            let (v0_lo,v0_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+0*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[0] };
            let v0_lo=v0_lo*v_scale*probability;
            let v0_hi=v0_hi*v_scale*probability;
            let (v8_lo,v8_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+STRIPE_LANES*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[STRIPE_LANES] };
            let v8_lo=v8_lo*v_scale*probability;
            let v8_hi=v8_hi*v_scale*probability;
            let (v16_lo,v16_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+DOT_LANES*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[DOT_LANES] };
            let v16_lo=v16_lo*v_scale*probability;
            let v16_hi=v16_hi*v_scale*probability;
            let (v24_lo,v24_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_24*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_24] };
            let v24_lo=v24_lo*v_scale*probability;
            let v24_hi=v24_hi*v_scale*probability;
            let g0_lo=((v0_lo+v8_lo)+v16_lo)+v24_lo;
            let g0_hi=((v0_hi+v8_hi)+v16_hi)+v24_hi;
            let (v1_lo,v1_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+1*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[1] };
            let v1_lo=v1_lo*v_scale*probability;
            let v1_hi=v1_hi*v_scale*probability;
            let (v9_lo,v9_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_09*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_09] };
            let v9_lo=v9_lo*v_scale*probability;
            let v9_hi=v9_hi*v_scale*probability;
            let (v17_lo,v17_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_17*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_17] };
            let v17_lo=v17_lo*v_scale*probability;
            let v17_hi=v17_hi*v_scale*probability;
            let (v25_lo,v25_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_25*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_25] };
            let v25_lo=v25_lo*v_scale*probability;
            let v25_hi=v25_hi*v_scale*probability;
            let g1_lo=((v1_lo+v9_lo)+v17_lo)+v25_lo;
            let g1_hi=((v1_hi+v9_hi)+v17_hi)+v25_hi;
            let (v2_lo,v2_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+2*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[2] };
            let v2_lo=v2_lo*v_scale*probability;
            let v2_hi=v2_hi*v_scale*probability;
            let (v10_lo,v10_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_10*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_10] };
            let v10_lo=v10_lo*v_scale*probability;
            let v10_hi=v10_hi*v_scale*probability;
            let (v18_lo,v18_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_18*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_18] };
            let v18_lo=v18_lo*v_scale*probability;
            let v18_hi=v18_hi*v_scale*probability;
            let (v26_lo,v26_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_26*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_26] };
            let v26_lo=v26_lo*v_scale*probability;
            let v26_hi=v26_hi*v_scale*probability;
            let g2_lo=((v2_lo+v10_lo)+v18_lo)+v26_lo;
            let g2_hi=((v2_hi+v10_hi)+v18_hi)+v26_hi;
            let (v3_lo,v3_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_03*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_03] };
            let v3_lo=v3_lo*v_scale*probability;
            let v3_hi=v3_hi*v_scale*probability;
            let (v11_lo,v11_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_11*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_11] };
            let v11_lo=v11_lo*v_scale*probability;
            let v11_hi=v11_hi*v_scale*probability;
            let (v19_lo,v19_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_19*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_19] };
            let v19_lo=v19_lo*v_scale*probability;
            let v19_hi=v19_hi*v_scale*probability;
            let (v27_lo,v27_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_27*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_27] };
            let v27_lo=v27_lo*v_scale*probability;
            let v27_hi=v27_hi*v_scale*probability;
            let g3_lo=((v3_lo+v11_lo)+v19_lo)+v27_lo;
            let g3_hi=((v3_hi+v11_hi)+v19_hi)+v27_hi;
            let (v4_lo,v4_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_04*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_04] };
            let v4_lo=v4_lo*v_scale*probability;
            let v4_hi=v4_hi*v_scale*probability;
            let (v12_lo,v12_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_12*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_12] };
            let v12_lo=v12_lo*v_scale*probability;
            let v12_hi=v12_hi*v_scale*probability;
            let (v20_lo,v20_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_20*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_20] };
            let v20_lo=v20_lo*v_scale*probability;
            let v20_hi=v20_hi*v_scale*probability;
            let (v28_lo,v28_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_28*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_28] };
            let v28_lo=v28_lo*v_scale*probability;
            let v28_hi=v28_hi*v_scale*probability;
            let g4_lo=((v4_lo+v12_lo)+v20_lo)+v28_lo;
            let g4_hi=((v4_hi+v12_hi)+v20_hi)+v28_hi;
            let (v5_lo,v5_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_05*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_05] };
            let v5_lo=v5_lo*v_scale*probability;
            let v5_hi=v5_hi*v_scale*probability;
            let (v13_lo,v13_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_13*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_13] };
            let v13_lo=v13_lo*v_scale*probability;
            let v13_hi=v13_hi*v_scale*probability;
            let (v21_lo,v21_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_21*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_21] };
            let v21_lo=v21_lo*v_scale*probability;
            let v21_hi=v21_hi*v_scale*probability;
            let (v29_lo,v29_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_29*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_29] };
            let v29_lo=v29_lo*v_scale*probability;
            let v29_hi=v29_hi*v_scale*probability;
            let g5_lo=((v5_lo+v13_lo)+v21_lo)+v29_lo;
            let g5_hi=((v5_hi+v13_hi)+v21_hi)+v29_hi;
            let (v6_lo,v6_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_06*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_06] };
            let v6_lo=v6_lo*v_scale*probability;
            let v6_hi=v6_hi*v_scale*probability;
            let (v14_lo,v14_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_14*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_14] };
            let v14_lo=v14_lo*v_scale*probability;
            let v14_hi=v14_hi*v_scale*probability;
            let (v22_lo,v22_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_22*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_22] };
            let v22_lo=v22_lo*v_scale*probability;
            let v22_hi=v22_hi*v_scale*probability;
            let (v30_lo,v30_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_30*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_30] };
            let v30_lo=v30_lo*v_scale*probability;
            let v30_hi=v30_hi*v_scale*probability;
            let g6_lo=((v6_lo+v14_lo)+v22_lo)+v30_lo;
            let g6_hi=((v6_hi+v14_hi)+v22_hi)+v30_hi;
            let (v7_lo,v7_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_07*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_07] };
            let v7_lo=v7_lo*v_scale*probability;
            let v7_hi=v7_hi*v_scale*probability;
            let (v15_lo,v15_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_15*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_15] };
            let v15_lo=v15_lo*v_scale*probability;
            let v15_hi=v15_hi*v_scale*probability;
            let (v23_lo,v23_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_23*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_23] };
            let v23_lo=v23_lo*v_scale*probability;
            let v23_hi=v23_hi*v_scale*probability;
            let (v31_lo,v31_hi)=fp8_pair_bits(unsafe { *values.add(cachebase+INDEX_31*HEAD_DIM+tid*2).cast::<u16>() });
            let probability=unsafe { PROBS[INDEX_31] };
            let v31_lo=v31_lo*v_scale*probability;
            let v31_hi=v31_hi*v_scale*probability;
            let g7_lo=((v7_lo+v15_lo)+v23_lo)+v31_lo;
            let g7_hi=((v7_hi+v15_hi)+v23_hi)+v31_hi;
            let contribution_lo=((g0_lo+g1_lo)+(g4_lo+g5_lo))+((g2_lo+g3_lo)+(g6_lo+g7_lo));
            numerator_lo=numerator_lo*unsafe { STATE[2] }+contribution_lo;
            let contribution_hi=((g0_hi+g1_hi)+(g4_hi+g5_hi))+((g2_hi+g3_hi)+(g6_hi+g7_hi));
            numerator_hi=numerator_hi*unsafe { STATE[2] }+contribution_hi;
            thread::sync_threads();
            block += 1;
        }
        unsafe {
            *out.add(row * HEAD_DIM + tid*2) = numerator_lo / STATE[1];
            *out.add(row * HEAD_DIM + tid*2+1) = numerator_hi / STATE[1];
        }
    }
}
fn main() {}
