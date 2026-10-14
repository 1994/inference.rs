// SPDX-License-Identifier: MIT
#![allow(static_mut_refs)]
use cuda_device::{SharedArray,kernel,thread,warp};
use cuda_host::cuda_module;
#[cuda_module]
mod kernels {
    use super::*;
    // One CTA owns one independent eight-column state slab.
    // Exactly 128 threads: 16 key workers by eight value columns.
    #[kernel]
    pub unsafe fn gdn_f32_128(
        state:*mut f32,out:*mut f32,qkv:*const f32,normalized:*const f32,
        betas:*const f32,decays:*const f32,metadata:*const i32,lanes:u32,
    ) {
        static mut PRED:SharedArray<f32,256>=SharedArray::UNINIT;
        static mut READ:SharedArray<f32,128>=SharedArray::UNINIT;
        static mut RESULT:SharedArray<f32,8>=SharedArray::UNINIT;
        let tid=thread::threadIdx_x() as usize;
        let r=tid/8;
        let c=tid%8;
        let block=thread::blockIdx_x() as usize;
        let head=block/16;
        let column=(block%16)*8+c;
        let key=head/3;
        let width=lanes as usize;
        let count=unsafe{*metadata.add(1)};
        let displacement=unsafe{*metadata}+unsafe{*metadata.add(2)};
        let first=if displacement<0 {(-displacement) as usize} else {0};
        let limit=if count<=0 {0} else if count as usize>width {width} else {count as usize};
        let mut lane=r;
        while lane<width {
            unsafe{*out.add((head*width+lane)*128+column)=0.0;}
            lane+=16;
        }
        let mut old0=unsafe{*state.add((head*128+r+0*16)*128+column)};
        let mut old1=unsafe{*state.add((head*128+r+1*16)*128+column)};
        let mut old2=unsafe{*state.add((head*128+r+2*16)*128+column)};
        let mut old3=unsafe{*state.add((head*128+r+3*16)*128+column)};
        let mut old4=unsafe{*state.add((head*128+r+4*16)*128+column)};
        let mut old5=unsafe{*state.add((head*128+r+5*16)*128+column)};
        let mut old6=unsafe{*state.add((head*128+r+6*16)*128+column)};
        let mut old7=unsafe{*state.add((head*128+r+7*16)*128+column)};
        lane=first;
        while lane<limit {
            let decay=unsafe{*decays.add(lane*48+head)};
            let beta=unsafe{*betas.add(lane*48+head)};
            let value=unsafe{*qkv.add((lane*80+32+head)*128+column)};
            let q0=unsafe{*normalized.add((lane*32+key)*128+r+0*16)};
            let k0=unsafe{*normalized.add((lane*32+16+key)*128+r+0*16)};
            let d0=old0*decay;
            let p0=d0*k0;
            let q1=unsafe{*normalized.add((lane*32+key)*128+r+1*16)};
            let k1=unsafe{*normalized.add((lane*32+16+key)*128+r+1*16)};
            let d1=old1*decay;
            let p1=d1*k1;
            let q2=unsafe{*normalized.add((lane*32+key)*128+r+2*16)};
            let k2=unsafe{*normalized.add((lane*32+16+key)*128+r+2*16)};
            let d2=old2*decay;
            let p2=d2*k2;
            let q3=unsafe{*normalized.add((lane*32+key)*128+r+3*16)};
            let k3=unsafe{*normalized.add((lane*32+16+key)*128+r+3*16)};
            let d3=old3*decay;
            let p3=d3*k3;
            let q4=unsafe{*normalized.add((lane*32+key)*128+r+4*16)};
            let k4=unsafe{*normalized.add((lane*32+16+key)*128+r+4*16)};
            let d4=old4*decay;
            let p4=d4*k4;
            let q5=unsafe{*normalized.add((lane*32+key)*128+r+5*16)};
            let k5=unsafe{*normalized.add((lane*32+16+key)*128+r+5*16)};
            let d5=old5*decay;
            let p5=d5*k5;
            let q6=unsafe{*normalized.add((lane*32+key)*128+r+6*16)};
            let k6=unsafe{*normalized.add((lane*32+16+key)*128+r+6*16)};
            let d6=old6*decay;
            let p6=d6*k6;
            let q7=unsafe{*normalized.add((lane*32+key)*128+r+7*16)};
            let k7=unsafe{*normalized.add((lane*32+16+key)*128+r+7*16)};
            let d7=old7*decay;
            let p7=d7*k7;
            let low=(((0.0+p0)+p2)+p4)+p6;
            let high=(((0.0+p1)+p3)+p5)+p7;
            unsafe{PRED[r*8+c]=low;PRED[(r+16)*8+c]=high;}
            thread::sync_threads();
            if tid<32 {
                let worker=r;
                let t0=unsafe{PRED[(worker+0*4)*8+c]};
                let t1=unsafe{PRED[(worker+1*4)*8+c]};
                let t2=unsafe{PRED[(worker+2*4)*8+c]};
                let t3=unsafe{PRED[(worker+3*4)*8+c]};
                let t4=unsafe{PRED[(worker+4*4)*8+c]};
                let t5=unsafe{PRED[(worker+5*4)*8+c]};
                let t6=unsafe{PRED[(worker+6*4)*8+c]};
                let t7=unsafe{PRED[(worker+7*4)*8+c]};
                let mut predicted=t7+(t6+(t5+(t4+(t3+(t2+(t0+t1))))));
                predicted+=warp::shuffle_xor_f32(predicted,16);
                predicted+=warp::shuffle_xor_f32(predicted,8);
                if tid<8 {unsafe{RESULT[c]=predicted;}}
            }
            thread::sync_threads();
            let predicted=unsafe{RESULT[c]};
            let diff=(value-predicted)*beta;
            old0=d0+k0*diff;
            let y0=old0*q0;
            old1=d1+k1*diff;
            let y1=old1*q1;
            old2=d2+k2*diff;
            let y2=old2*q2;
            old3=d3+k3*diff;
            let y3=old3*q3;
            old4=d4+k4*diff;
            let y4=old4*q4;
            old5=d5+k5*diff;
            let y5=old5*q5;
            old6=d6+k6*diff;
            let y6=old6*q6;
            old7=d7+k7*diff;
            let y7=old7*q7;
            let mut subtotal=0.0f32;
            subtotal+=y0;
            subtotal+=y1;
            subtotal+=y2;
            subtotal+=y3;
            subtotal+=y4;
            subtotal+=y5;
            subtotal+=y6;
            subtotal+=y7;

            unsafe{READ[r*8+c]=subtotal;}
            thread::sync_threads();
            if tid<32 {
                let worker=r;
                let s0=unsafe{READ[worker*8+c]};
                let s1=unsafe{READ[(worker+4)*8+c]};
                let s2=unsafe{READ[(worker+8)*8+c]};
                let s3=unsafe{READ[(worker+12)*8+c]};
                let mut result=s3+(s2+(s0+s1));
                result+=warp::shuffle_xor_f32(result,16);
                result+=warp::shuffle_xor_f32(result,8);
                if tid<8 {unsafe{*out.add((head*width+lane)*128+column)=result;}}
            }
            // READ is distinct from PRED/RESULT. The next prediction barrier prevents
            // this warp from racing the next readout without adding a fourth barrier.
            lane+=1;
        }
        unsafe{*state.add((head*128+r+0*16)*128+column)=old0;}
        unsafe{*state.add((head*128+r+1*16)*128+column)=old1;}
        unsafe{*state.add((head*128+r+2*16)*128+column)=old2;}
        unsafe{*state.add((head*128+r+3*16)*128+column)=old3;}
        unsafe{*state.add((head*128+r+4*16)*128+column)=old4;}
        unsafe{*state.add((head*128+r+5*16)*128+column)=old5;}
        unsafe{*state.add((head*128+r+6*16)*128+column)=old6;}
        unsafe{*state.add((head*128+r+7*16)*128+column)=old7;}
    }
}
fn main() {}
