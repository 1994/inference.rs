// SPDX-License-Identifier: MIT
#![allow(static_mut_refs, reason = "CUDA shared arrays are scoped per CTA and synchronized by explicit block barriers")]

// Fixed SM120 geometry and canonical reduction offsets; not user options.
const INDEX_03: usize = 3;
const INDEX_04: usize = 4;
const INDEX_05: usize = 5;
const INDEX_06: usize = 6;
const INDEX_07: usize = 7;
const VALUE_COLUMNS: usize = 8;
const INDEX_12: usize = 12;
const KEY_WORKERS: usize = 16;
const WARP_SIZE: usize = 32;
const VALUE_HEADS: usize = 48;
const PROJECTED_HEADS: usize = 80;
const HEAD_DIM: usize = 128;
const PREDICTION_ELEMENTS: usize = 256;
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
        static mut PRED:SharedArray<f32,PREDICTION_ELEMENTS>=SharedArray::UNINIT;
        static mut READ:SharedArray<f32,HEAD_DIM>=SharedArray::UNINIT;
        static mut RESULT:SharedArray<f32,VALUE_COLUMNS>=SharedArray::UNINIT;
        let tid=thread::threadIdx_x() as usize;
        let r=tid/VALUE_COLUMNS;
        let c=tid%VALUE_COLUMNS;
        let block=thread::blockIdx_x() as usize;
        let head=block/KEY_WORKERS;
        let column=(block%KEY_WORKERS)*VALUE_COLUMNS+c;
        let key=head/INDEX_03;
        let width=lanes as usize;
        let count=unsafe{*metadata.add(1)};
        let displacement=unsafe{*metadata}+unsafe{*metadata.add(2)};
        let first=if displacement<0 {(-displacement) as usize} else {0};
        let limit=if count<=0 {0} else if count as usize>width {width} else {count as usize};
        let mut lane=r;
        while lane<width {
            unsafe{*out.add((head*width+lane)*HEAD_DIM+column)=0.0;}
            lane+=KEY_WORKERS;
        }
        let mut old0=unsafe{*state.add((head*HEAD_DIM+r+0*KEY_WORKERS)*HEAD_DIM+column)};
        let mut old1=unsafe{*state.add((head*HEAD_DIM+r+1*KEY_WORKERS)*HEAD_DIM+column)};
        let mut old2=unsafe{*state.add((head*HEAD_DIM+r+2*KEY_WORKERS)*HEAD_DIM+column)};
        let mut old3=unsafe{*state.add((head*HEAD_DIM+r+INDEX_03*KEY_WORKERS)*HEAD_DIM+column)};
        let mut old4=unsafe{*state.add((head*HEAD_DIM+r+INDEX_04*KEY_WORKERS)*HEAD_DIM+column)};
        let mut old5=unsafe{*state.add((head*HEAD_DIM+r+INDEX_05*KEY_WORKERS)*HEAD_DIM+column)};
        let mut old6=unsafe{*state.add((head*HEAD_DIM+r+INDEX_06*KEY_WORKERS)*HEAD_DIM+column)};
        let mut old7=unsafe{*state.add((head*HEAD_DIM+r+INDEX_07*KEY_WORKERS)*HEAD_DIM+column)};
        lane=first;
        while lane<limit {
            let decay=unsafe{*decays.add(lane*VALUE_HEADS+head)};
            let beta=unsafe{*betas.add(lane*VALUE_HEADS+head)};
            let value=unsafe{*qkv.add((lane*PROJECTED_HEADS+WARP_SIZE+head)*HEAD_DIM+column)};
            let q0=unsafe{*normalized.add((lane*WARP_SIZE+key)*HEAD_DIM+r+0*KEY_WORKERS)};
            let k0=unsafe{*normalized.add((lane*WARP_SIZE+KEY_WORKERS+key)*HEAD_DIM+r+0*KEY_WORKERS)};
            let d0=old0*decay;
            let p0=d0*k0;
            let q1=unsafe{*normalized.add((lane*WARP_SIZE+key)*HEAD_DIM+r+1*KEY_WORKERS)};
            let k1=unsafe{*normalized.add((lane*WARP_SIZE+KEY_WORKERS+key)*HEAD_DIM+r+1*KEY_WORKERS)};
            let d1=old1*decay;
            let p1=d1*k1;
            let q2=unsafe{*normalized.add((lane*WARP_SIZE+key)*HEAD_DIM+r+2*KEY_WORKERS)};
            let k2=unsafe{*normalized.add((lane*WARP_SIZE+KEY_WORKERS+key)*HEAD_DIM+r+2*KEY_WORKERS)};
            let d2=old2*decay;
            let p2=d2*k2;
            let q3=unsafe{*normalized.add((lane*WARP_SIZE+key)*HEAD_DIM+r+INDEX_03*KEY_WORKERS)};
            let k3=unsafe{*normalized.add((lane*WARP_SIZE+KEY_WORKERS+key)*HEAD_DIM+r+INDEX_03*KEY_WORKERS)};
            let d3=old3*decay;
            let p3=d3*k3;
            let q4=unsafe{*normalized.add((lane*WARP_SIZE+key)*HEAD_DIM+r+INDEX_04*KEY_WORKERS)};
            let k4=unsafe{*normalized.add((lane*WARP_SIZE+KEY_WORKERS+key)*HEAD_DIM+r+INDEX_04*KEY_WORKERS)};
            let d4=old4*decay;
            let p4=d4*k4;
            let q5=unsafe{*normalized.add((lane*WARP_SIZE+key)*HEAD_DIM+r+INDEX_05*KEY_WORKERS)};
            let k5=unsafe{*normalized.add((lane*WARP_SIZE+KEY_WORKERS+key)*HEAD_DIM+r+INDEX_05*KEY_WORKERS)};
            let d5=old5*decay;
            let p5=d5*k5;
            let q6=unsafe{*normalized.add((lane*WARP_SIZE+key)*HEAD_DIM+r+INDEX_06*KEY_WORKERS)};
            let k6=unsafe{*normalized.add((lane*WARP_SIZE+KEY_WORKERS+key)*HEAD_DIM+r+INDEX_06*KEY_WORKERS)};
            let d6=old6*decay;
            let p6=d6*k6;
            let q7=unsafe{*normalized.add((lane*WARP_SIZE+key)*HEAD_DIM+r+INDEX_07*KEY_WORKERS)};
            let k7=unsafe{*normalized.add((lane*WARP_SIZE+KEY_WORKERS+key)*HEAD_DIM+r+INDEX_07*KEY_WORKERS)};
            let d7=old7*decay;
            let p7=d7*k7;
            let low=(((0.0+p0)+p2)+p4)+p6;
            let high=(((0.0+p1)+p3)+p5)+p7;
            unsafe{PRED[r*VALUE_COLUMNS+c]=low;PRED[(r+KEY_WORKERS)*VALUE_COLUMNS+c]=high;}
            thread::sync_threads();
            if tid<WARP_SIZE {
                let worker=r;
                let t0=unsafe{PRED[(worker+0*INDEX_04)*VALUE_COLUMNS+c]};
                let t1=unsafe{PRED[(worker+1*INDEX_04)*VALUE_COLUMNS+c]};
                let t2=unsafe{PRED[(worker+2*INDEX_04)*VALUE_COLUMNS+c]};
                let t3=unsafe{PRED[(worker+INDEX_03*INDEX_04)*VALUE_COLUMNS+c]};
                let t4=unsafe{PRED[(worker+INDEX_04*INDEX_04)*VALUE_COLUMNS+c]};
                let t5=unsafe{PRED[(worker+INDEX_05*INDEX_04)*VALUE_COLUMNS+c]};
                let t6=unsafe{PRED[(worker+INDEX_06*INDEX_04)*VALUE_COLUMNS+c]};
                let t7=unsafe{PRED[(worker+INDEX_07*INDEX_04)*VALUE_COLUMNS+c]};
                let mut predicted=t7+(t6+(t5+(t4+(t3+(t2+(t0+t1))))));
                predicted+=warp::shuffle_xor_f32(predicted,KEY_WORKERS as u32);
                predicted+=warp::shuffle_xor_f32(predicted,VALUE_COLUMNS as u32);
                if tid<VALUE_COLUMNS {unsafe{RESULT[c]=predicted;}}
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

            unsafe{READ[r*VALUE_COLUMNS+c]=subtotal;}
            thread::sync_threads();
            if tid<WARP_SIZE {
                let worker=r;
                let s0=unsafe{READ[worker*VALUE_COLUMNS+c]};
                let s1=unsafe{READ[(worker+INDEX_04)*VALUE_COLUMNS+c]};
                let s2=unsafe{READ[(worker+VALUE_COLUMNS)*VALUE_COLUMNS+c]};
                let s3=unsafe{READ[(worker+INDEX_12)*VALUE_COLUMNS+c]};
                let mut result=s3+(s2+(s0+s1));
                result+=warp::shuffle_xor_f32(result,KEY_WORKERS as u32);
                result+=warp::shuffle_xor_f32(result,VALUE_COLUMNS as u32);
                if tid<VALUE_COLUMNS {unsafe{*out.add((head*width+lane)*HEAD_DIM+column)=result;}}
            }
            // READ is distinct from PRED/RESULT. The next prediction barrier prevents
            // this warp from racing the next readout without adding a fourth barrier.
            lane+=1;
        }
        unsafe{*state.add((head*HEAD_DIM+r+0*KEY_WORKERS)*HEAD_DIM+column)=old0;}
        unsafe{*state.add((head*HEAD_DIM+r+1*KEY_WORKERS)*HEAD_DIM+column)=old1;}
        unsafe{*state.add((head*HEAD_DIM+r+2*KEY_WORKERS)*HEAD_DIM+column)=old2;}
        unsafe{*state.add((head*HEAD_DIM+r+INDEX_03*KEY_WORKERS)*HEAD_DIM+column)=old3;}
        unsafe{*state.add((head*HEAD_DIM+r+INDEX_04*KEY_WORKERS)*HEAD_DIM+column)=old4;}
        unsafe{*state.add((head*HEAD_DIM+r+INDEX_05*KEY_WORKERS)*HEAD_DIM+column)=old5;}
        unsafe{*state.add((head*HEAD_DIM+r+INDEX_06*KEY_WORKERS)*HEAD_DIM+column)=old6;}
        unsafe{*state.add((head*HEAD_DIM+r+INDEX_07*KEY_WORKERS)*HEAD_DIM+column)=old7;}
    }
}
fn main() {}
