// Stage one FP8 KV block once for twelve GQA queries with canonical F32 arithmetic.
// FP8 codes are exactly representable in FP16; scales and all reductions remain F32.
// One warp per query, twelve queries per CTA (two positions and six GQA heads).
// The F32 reduction tree, online 32-key recurrence and FP8 cache contract stay canonical.
// The backend selects this geometry through exact-output and hardware timing gates.
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float widen(unsigned char x) {
    const unsigned sign=(unsigned(x)&128u)<<24u;
    const unsigned exponent=(unsigned(x)&127u)>>3u;
    const unsigned mantissa=unsigned(x)&7u;
    if(exponent==0u) {
        float magnitude=float(mantissa)*(1.0f/512.0f);
        return sign==0u?magnitude:-magnitude;
    }
    return __uint_as_float(sign|((exponent+120u)<<23u)|(mantissa<<20u));
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
__device__ __forceinline__ int key_swizzle(int index) {
    return index ^ ((((index & 255)>>4)&12)>>1) ^ (((index>>8)&1)<<3);
}
extern "C" __global__ __launch_bounds__(384,2) void attention_fp8_256(
    const float* __restrict__ query,
    const unsigned char* __restrict__ keys,
    const unsigned char* __restrict__ values,
    const int* __restrict__ metadata,
    const int* __restrict__ table,
    float* __restrict__ output,unsigned capacity,int window,float key_scale,float value_scale) {
    __shared__ __half shared_keys[32*256],shared_values[32*256];
    __shared__ float scores[12][32],probabilities[12][32],state[12][3];
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
    float numerator0=0.0f,numerator1=0.0f,numerator2=0.0f,numerator3=0.0f,numerator4=0.0f,numerator5=0.0f,numerator6=0.0f,numerator7=0.0f;
    if(lane==0) {state[warp][0]=-1.e30f;state[warp][1]=0.0f;}
    __syncwarp(full);
    const int end_block=tile<count && last_position>=0?last_position/32:-1;
    for(int block=first_start/32;block<=end_block;++block) {
        const size_t cache=(size_t(kv_head)*capacity+size_t(table[block])*32u)*256u;
        for(int index=tid;index<32*256;index+=384) {
            shared_keys[key_swizzle(index)]=__float2half_rn(widen(keys[cache+index]));
            shared_values[index]=__float2half_rn(widen(values[cache+index]));
        }
        __syncthreads();
        if(active && block>=start/32 && block<=position/32) {
            #pragma unroll 1
            for(int pair=0;pair<16;++pair) {
                const int keyrow=pair*2+lane/16;
                float p[16];
                #pragma unroll
                for(int i=0;i<16;++i) {
                    const int index=keyrow*256+dot_lane*16+i;
                    const float key=__half2float(shared_keys[key_swizzle(index)]);
                    p[i]=__fmul_rn(q[i],__fmul_rn(key,key_scale));
                }
                const float even=((p[0]+p[2])+(p[4]+p[6]))+((p[8]+p[10])+(p[12]+p[14]));
                const float odd=((p[1]+p[3])+(p[5]+p[7]))+((p[9]+p[11])+(p[13]+p[15]));
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
            #pragma unroll 1
            for(int j=0;j<8;++j) {
                const int dim=j*32+lane;
                #define V(i) __fmul_rn(__fmul_rn(__half2float(shared_values[i*256+dim]),value_scale),probabilities[warp][i])
                const float g0=((V(0)+V(8))+V(16))+V(24);
                const float g1=((V(1)+V(9))+V(17))+V(25);
                const float g2=((V(2)+V(10))+V(18))+V(26);
                const float g3=((V(3)+V(11))+V(19))+V(27);
                const float g4=((V(4)+V(12))+V(20))+V(28);
                const float g5=((V(5)+V(13))+V(21))+V(29);
                const float g6=((V(6)+V(14))+V(22))+V(30);
                const float g7=((V(7)+V(15))+V(23))+V(31);
                #undef V
                const float contribution=((g0+g1)+(g4+g5))+((g2+g3)+(g6+g7));
                float accumulator=0.0f;
                switch(j){case 0:accumulator=numerator0;break;case 1:accumulator=numerator1;break;case 2:accumulator=numerator2;break;case 3:accumulator=numerator3;break;case 4:accumulator=numerator4;break;case 5:accumulator=numerator5;break;case 6:accumulator=numerator6;break;case 7:accumulator=numerator7;break;}
                accumulator=__fadd_rn(__fmul_rn(accumulator,state[warp][2]),contribution);
                switch(j){case 0:numerator0=accumulator;break;case 1:numerator1=accumulator;break;case 2:numerator2=accumulator;break;case 3:numerator3=accumulator;break;case 4:numerator4=accumulator;break;case 5:numerator5=accumulator;break;case 6:numerator6=accumulator;break;case 7:numerator7=accumulator;break;}
            }
        }
        __syncthreads();
    }
    output[row*256+0*32+lane]=active?numerator0/state[warp][1]:0.0f;
    output[row*256+1*32+lane]=active?numerator1/state[warp][1]:0.0f;
    output[row*256+2*32+lane]=active?numerator2/state[warp][1]:0.0f;
    output[row*256+3*32+lane]=active?numerator3/state[warp][1]:0.0f;
    output[row*256+4*32+lane]=active?numerator4/state[warp][1]:0.0f;
    output[row*256+5*32+lane]=active?numerator5/state[warp][1]:0.0f;
    output[row*256+6*32+lane]=active?numerator6/state[warp][1]:0.0f;
    output[row*256+7*32+lane]=active?numerator7/state[warp][1]:0.0f;
}
