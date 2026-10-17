// v-combined -- bit-exact composition of the two accepted variants.
// Base source: crates/backend/cuda/kernels/vector-attention/attention-sm120.cu
//   git origin/codex/fa4-architecture-prefill, sha256 feea1072e8480a7235cfec7430aeb40e7960c00e1d3aea0b725b3b623421d2f9
//
// Staging side (verbatim from v-vec): 16B v4.u32 LDG.128 loads plus a per-CTA
// 2x256-entry shared widen LUT (lut_lo[b]=widen_pair(b)&0xffff,
// lut_hi[b]=(widen_pair(b)&0xffff)<<16) and two STS.128 per 8-pair chunk.
//   ko[i]=lut_lo[kw[i]&0xff]|lut_hi[(kw[i]>>8)&0xff] == widen_pair(kw[i]).
//
// PV consumer side (verbatim from v-pv): 8 consecutive dims per lane, one
// LDS.128 per shared_values row per lane, identity layout slot(row,c)=row*128+c,
// and the per-dim G tree
//   G(i)=((V(i)+V(i+8))+V(i+16))+V(i+24),
//   contribution=((G0+G1)+(G4+G5))+((G2+G3)+(G6+G7))
// with the same operand order and __fadd_rn/__fmul_rn pairing.
//
// The two changes are disjoint: v-vec touches only the staging producer of
// shared_keys/shared_values, v-pv touches only the PV consumer of shared_values.
// shared_values keeps the identity layout on both sides, so no re-layout is
// needed.  The only structural merge is the LUT: it is initialised once before
// the block loop and guarded by the existing __syncthreads() (one extra barrier),
// which changes no numeric value.  QK dot-lattice, online softmax recurrence,
// canonical_exp, FP8 +/-480 codes and the F32 reduction tree order are byte-for-
// byte the pinned baseline.
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

// Preserve all 256 existing FP8 code meanings, including reserved +/-480.
__device__ __forceinline__ unsigned widen_pair(unsigned short raw) {
    union { __half2_raw half; unsigned bits; } packed;
    packed.half=__nv_cvt_fp8x2_to_halfraw2(raw,__NV_E4M3);
    if((raw&0x7fu)==0x7fu)
        packed.bits=(packed.bits&0xffff0000u)|0x5f80u|((unsigned(raw)&0x80u)<<8u);
    if((raw&0x7f00u)==0x7f00u)
        packed.bits=(packed.bits&0x0000ffffu)|0x5f800000u|((unsigned(raw)&0x8000u)<<16u);
    return packed.bits;
}
// Preserve the pinned Rust/PTX exponential, including its explicit range reduction.
// Using expf or __expf here would change the numerical contract independently of layout.
__device__ __forceinline__ float canonical_exp(float x) {
    float result;
    asm volatile(
        "{ .reg .f32 a,b,c,d,e,f,g,h; .reg .b32 bits; "
        "fma.rn.f32 a,%1,0f3BBB989D,0f3F000000; "
        "cvt.sat.f32.f32 b,a; "
        "fma.rm.f32 c,b,0f437C0000,0f4B400001; "
        "add.rn.f32 d,c,0fCB40007F; "
        "neg.f32 e,d; "
        "fma.rn.f32 f,%1,0f3FB8AA3B,e; "
        "fma.rn.f32 g,%1,0f32A57060,f; "
        "mov.b32 bits,c; shl.b32 bits,bits,23; "
        "ex2.approx.ftz.f32 h,g; "
        "mov.b32 a,bits; mul.rn.f32 %0,h,a; }"
        : "=f"(result) : "f"(x));
    return result;
}
// Each half of a16-element dot-lane fragment is contiguous and aligned16bytes.
// Consumer warp lanes address consecutive16byte fragments, so every128byte shared
// transaction covers eight distinct16byte fragments without consumer bank conflicts.
__device__ __forceinline__ int key_layout(int index) {
    const int row=index/256,dimension=index%256;
    return row*256+(dimension/8%2)*128+(dimension/16)*8+dimension%8;
}
__device__ __forceinline__ float2 dot_eight(
    const float* q,const __half* key,float scale) {
    unsigned a,b,c,d;
    const unsigned address=unsigned(__cvta_generic_to_shared(key));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3},[%4];"
                 : "=r"(a),"=r"(b),"=r"(c),"=r"(d) : "r"(address));
    union { unsigned bits; __half2 half; } pair;
    float p[8];
    pair.bits=a;
    p[0]=__fmul_rn(q[0],__fmul_rn(__low2float(pair.half),scale));
    p[1]=__fmul_rn(q[1],__fmul_rn(__high2float(pair.half),scale));
    pair.bits=b;
    p[2]=__fmul_rn(q[2],__fmul_rn(__low2float(pair.half),scale));
    p[3]=__fmul_rn(q[3],__fmul_rn(__high2float(pair.half),scale));
    pair.bits=c;
    p[4]=__fmul_rn(q[4],__fmul_rn(__low2float(pair.half),scale));
    p[5]=__fmul_rn(q[5],__fmul_rn(__high2float(pair.half),scale));
    pair.bits=d;
    p[6]=__fmul_rn(q[6],__fmul_rn(__low2float(pair.half),scale));
    p[7]=__fmul_rn(q[7],__fmul_rn(__high2float(pair.half),scale));
    return make_float2((p[0]+p[2])+(p[4]+p[6]),
                       (p[1]+p[3])+(p[5]+p[7]));
}

// Paired shared loads preserve the original two scalar F32 values.
__device__ __forceinline__ float2 paired_half(const __half* pointer) {
    unsigned raw;
    const unsigned address=unsigned(__cvta_generic_to_shared(pointer));
    asm volatile("ld.shared.b32 %0,[%1];" : "=r"(raw) : "r"(address));
    union { unsigned raw; __half2 value; } pair;
    pair.raw=raw;
    return make_float2(__low2float(pair.value),__high2float(pair.value));
}
__device__ __forceinline__ float2 add_pair(float2 a,float2 b) {
    return make_float2(__fadd_rn(a.x,b.x),__fadd_rn(a.y,b.y));
}
__device__ __forceinline__ float2 weighted_value(
    const __half* values,const float* probabilities,int row,int dimension,float scale) {
    const float2 value=paired_half(values+row*256+dimension);
    const float p=probabilities[row];
    return make_float2(__fmul_rn(__fmul_rn(value.x,scale),p),
                      __fmul_rn(__fmul_rn(value.y,scale),p));
}
// One row's 8 halves for lane l: dims 8l..8l+7, as four float2 (dim 2c, 2c+1).
// The four u32 are 16B-aligned and contiguous; LDS.128 is conflict-free because
// consecutive lanes step 16B, so each 8-lane phase hits banks {0,4,...,28}.
__device__ __forceinline__ void weighted_quad(
    const __half* values,const float* probabilities,int row,int lane,float scale,float2 out[4]) {
    const unsigned address=unsigned(__cvta_generic_to_shared(values+row*256+lane*8));
    unsigned a,b,c,d;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3},[%4];"
                 : "=r"(a),"=r"(b),"=r"(c),"=r"(d) : "r"(address));
    const float p=probabilities[row];
    const unsigned regs[4]={a,b,c,d};
    #pragma unroll
    for(int q=0;q<4;++q) {
        union { unsigned bits; __half2 half; } u; u.bits=regs[q];
        out[q].x=__fmul_rn(__fmul_rn(__low2float(u.half),scale),p);
        out[q].y=__fmul_rn(__fmul_rn(__high2float(u.half),scale),p);
    }
}
// G(i)=((V(i)+V(i+8))+V(i+16))+V(i+24) for all four c of lane l at once.
__device__ __forceinline__ void g_quad(
    const __half* values,const float* probabilities,int i,int lane,float scale,float2 out[4]) {
    float2 t[4],u[4];
    weighted_quad(values,probabilities,i,lane,scale,t);
    weighted_quad(values,probabilities,i+8,lane,scale,u);
    #pragma unroll
    for(int q=0;q<4;++q) t[q]=add_pair(t[q],u[q]);
    weighted_quad(values,probabilities,i+16,lane,scale,u);
    #pragma unroll
    for(int q=0;q<4;++q) t[q]=add_pair(t[q],u[q]);
    weighted_quad(values,probabilities,i+24,lane,scale,u);
    #pragma unroll
    for(int q=0;q<4;++q) out[q]=add_pair(t[q],u[q]);
}

extern "C" __global__ __launch_bounds__(384,2) void attention_fp8_256_vec_k_combined(
    const float* __restrict__ query,
    const unsigned char* __restrict__ keys,
    const unsigned char* __restrict__ values,
    const int* __restrict__ metadata,
    const int* __restrict__ table,
    float* __restrict__ output,unsigned capacity,int window,float key_scale,float value_scale) {
    __shared__ __align__(16) __half shared_keys[32*256],shared_values[32*256];
    __shared__ float scores[12][32],probabilities[12][32],state[12][3];
    __shared__ unsigned lut_lo[256],lut_hi[256];
    const unsigned full=0xffffffffu;
    const int tid=int(threadIdx.x),lane=tid&31,warp=tid>>5;
    const int kv_head=int(blockIdx.x)%4,tile=(int(blockIdx.x)/4)*2;
    const int query_lane=tile+warp/6,head=kv_head*6+warp%6,row=query_lane*24+head;
    const int base=metadata[0],offset=metadata[2],count=metadata[1],position=base+query_lane+offset;
    const bool active=query_lane<count && position>=0;
    const int start=window>0 && position+1>window?position+1-window:0;
    const int last_lane=min(tile+1,count-1),last_position=base+last_lane+offset;
    const int earliest=base+tile+offset;
    const int first_start=window>0?max(0,earliest+1-window):0;
    const int dot_lane=lane&15;
    float q[16];
    #pragma unroll
    for(int i=0;i<16;++i)q[i]=query[row*256+dot_lane*16+i];
    // num[m] is the online numerator for dim 8*lane+m; each dim is updated once
    // per block with alpha=state[warp][2].
    float num[8]={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    if(lane==0) {state[warp][0]=-1.e30f;state[warp][1]=0.0f;}
    __syncwarp(full);
    // One-time 2x256 pad table: lut_lo[b]=widen_pair(b)&0xffff, lut_hi[b]=that<<16.
    for(int b=tid;b<256;b+=384) {
        const unsigned bits=widen_pair((unsigned short)b)&0xffffu;
        lut_lo[b]=bits; lut_hi[b]=bits<<16u;
    }
    __syncthreads();
    const int end_block=tile<count && last_position>=0?last_position/32:-1;
    for(int block=first_start/32;block<=end_block;++block) {
        const size_t cache=(size_t(kv_head)*capacity+size_t(table[block])*32u)*256u;
        for(int chunk=tid;chunk<512;chunk+=384) {          // 512 = 4096 pair / 8
            const int p0=chunk*8;
            const uint4 kraw=*reinterpret_cast<const uint4*>(keys+cache+p0*2);
            const uint4 vraw=*reinterpret_cast<const uint4*>(values+cache+p0*2);
            const unsigned short* kw=reinterpret_cast<const unsigned short*>(&kraw);
            const unsigned short* vw=reinterpret_cast<const unsigned short*>(&vraw);
            unsigned ko[8],vo[8];
            #pragma unroll
            for(int i=0;i<8;++i) {
                ko[i]=lut_lo[kw[i]&0xffu]|lut_hi[(kw[i]>>8)&0xffu];
                vo[i]=lut_lo[vw[i]&0xffu]|lut_hi[(vw[i]>>8)&0xffu];
            }
            const unsigned kb=unsigned(key_layout(p0*2)/2);
            unsigned* shared_key_words=reinterpret_cast<unsigned*>(shared_keys);
            unsigned* shared_value_words=reinterpret_cast<unsigned*>(shared_values);
            *reinterpret_cast<uint4*>(shared_key_words+kb)=make_uint4(ko[0],ko[1],ko[2],ko[3]);
            *reinterpret_cast<uint4*>(shared_key_words+kb+64)=make_uint4(ko[4],ko[5],ko[6],ko[7]);
            *reinterpret_cast<uint4*>(shared_value_words+p0)=make_uint4(vo[0],vo[1],vo[2],vo[3]);
            *reinterpret_cast<uint4*>(shared_value_words+p0+4)=make_uint4(vo[4],vo[5],vo[6],vo[7]);
        }
        __syncthreads();
        if(active && block>=start/32 && block<=position/32) {
            #pragma unroll 1
            for(int pair=0;pair<16;++pair) {
                const int keyrow=pair*2+lane/16;
                const __half* key=shared_keys+keyrow*256+dot_lane*8;
                const float2 first=dot_eight(q,key,key_scale);
                const float2 last=dot_eight(q+8,key+128,key_scale);
                const float even=first.x+last.x,odd=first.y+last.y;
                float score=even+odd;
                score+=__shfl_xor_sync(full,score,8);
                score+=__shfl_xor_sync(full,score,4);
                score+=__shfl_xor_sync(full,score,2);
                score+=__shfl_xor_sync(full,score,1);
                if(dot_lane==0) {
                    const int keyposition=block*32+keyrow;
                    scores[warp][keyrow]=keyposition>=start && keyposition<=position?score*0.0625f:-1.e30f;
                }
            }
            __syncwarp(full);
            const float score=scores[warp][lane];
            float maximum=score;
            maximum=fmaxf(maximum,__shfl_xor_sync(full,maximum,1));
            maximum=fmaxf(maximum,__shfl_xor_sync(full,maximum,2));
            maximum=fmaxf(maximum,__shfl_xor_sync(full,maximum,4));
            maximum=fmaxf(maximum,__shfl_xor_sync(full,maximum,8));
            maximum=fmaxf(maximum,__shfl_xor_sync(full,maximum,16));
            const float old=state[warp][0],next=maximum>old?maximum:old;
            const float old_scale=canonical_exp(old-next),probability=canonical_exp(score-next);
            probabilities[warp][lane]=probability;
            const int local=lane&7;
            float sum=__shfl_sync(full,probability,local);
            sum+=__shfl_sync(full,probability,local+8);
            sum+=__shfl_sync(full,probability,local+16);
            sum+=__shfl_sync(full,probability,local+24);
            sum+=__shfl_xor_sync(full,sum,1);
            sum+=__shfl_xor_sync(full,sum,4);
            sum+=__shfl_xor_sync(full,sum,2);
            if(lane==0) {
                state[warp][0]=next;
                state[warp][1]=__fadd_rn(__fmul_rn(state[warp][1],old_scale),sum);
                state[warp][2]=old_scale;
            }
            __syncwarp(full);
            const float alpha=state[warp][2];
            float2 A[4],B[4],T[4],U[4];
            g_quad(shared_values,probabilities[warp],0,lane,value_scale,A);      // A = G0
            g_quad(shared_values,probabilities[warp],1,lane,value_scale,T);      // T = G1
            #pragma unroll
            for(int q=0;q<4;++q) A[q]=add_pair(A[q],T[q]);                       // A = G0+G1
            g_quad(shared_values,probabilities[warp],4,lane,value_scale,T);      // T = G4
            g_quad(shared_values,probabilities[warp],5,lane,value_scale,U);      // U = G5
            #pragma unroll
            for(int q=0;q<4;++q) { T[q]=add_pair(T[q],U[q]); A[q]=add_pair(A[q],T[q]); } // A=(G0+G1)+(G4+G5)
            g_quad(shared_values,probabilities[warp],2,lane,value_scale,B);      // B = G2
            g_quad(shared_values,probabilities[warp],3,lane,value_scale,T);      // T = G3
            #pragma unroll
            for(int q=0;q<4;++q) B[q]=add_pair(B[q],T[q]);                       // B = G2+G3
            g_quad(shared_values,probabilities[warp],6,lane,value_scale,T);      // T = G6
            g_quad(shared_values,probabilities[warp],7,lane,value_scale,U);      // U = G7
            #pragma unroll
            for(int q=0;q<4;++q) { T[q]=add_pair(T[q],U[q]); B[q]=add_pair(B[q],T[q]); } // B=(G2+G3)+(G6+G7)
            #pragma unroll
            for(int q=0;q<4;++q) {
                const float2 contribution=add_pair(A[q],B[q]);
                num[2*q]  =__fadd_rn(__fmul_rn(num[2*q],  alpha),contribution.x);
                num[2*q+1]=__fadd_rn(__fmul_rn(num[2*q+1],alpha),contribution.y);
            }
        }
        __syncthreads();
    }
    // Recompute readout indexing after the recurrence instead of keeping row and
    // its output address live through every QK/V loop. The row identity is unchanged.
    unsigned final_cta,final_tid;
    asm volatile("mov.u32 %0, %%ctaid.x; mov.u32 %1, %%tid.x;"
                 : "=r"(final_cta),"=r"(final_tid));
    const int final_warp=int(final_tid)>>5;
    const int output_row=(int(final_cta)/4*2+final_warp/6)*24
                        +int(final_cta)%4*6+final_warp%6;
    #pragma unroll
    for(int m=0;m<8;++m)
        output[output_row*256+lane*8+m]=active?num[m]/state[warp][1]:0.0f;
}
