// Stage one FP8 KV block once for twelve GQA queries with canonical F32 arithmetic.
// FP8 codes are exactly representable in FP16; scales and all reductions remain F32.
// One warp per query, twelve queries per CTA (two positions and six GQA heads).
// The F32 reduction tree, online 32-key recurrence and FP8 cache contract stay canonical.
// The backend selects this geometry through exact-output and hardware timing gates.
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
__device__ __forceinline__ int key_swizzle(int index) {
    return index ^ ((((index & 255)>>4)&12)>>1) ^ (((index>>8)&1)<<3);
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

extern "C" __global__ __launch_bounds__(384,2) void attention_fp8_256(
    const float* __restrict__ query,
    const unsigned char* __restrict__ keys,
    const unsigned char* __restrict__ values,
    const int* __restrict__ metadata,
    const int* __restrict__ table,
    float* __restrict__ output,unsigned capacity,int window,float key_scale,float value_scale) {
    __shared__ __align__(4) __half shared_keys[32*256],shared_values[32*256];
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
        for(int pair=tid;pair<32*128;pair+=384) {
            const int index=pair*2;
            const unsigned short k=reinterpret_cast<const unsigned short*>(keys+cache)[pair];
            const unsigned short v=reinterpret_cast<const unsigned short*>(values+cache)[pair];
            reinterpret_cast<unsigned*>(shared_keys)[key_swizzle(index)/2]=widen_pair(k);
            reinterpret_cast<unsigned*>(shared_values)[pair]=widen_pair(v);
        }
        __syncthreads();
        if(active && block>=start/32 && block<=position/32) {
            #pragma unroll 1
            for(int pair=0;pair<16;++pair) {
                const int keyrow=pair*2+lane/16;
                float p[16];
                #pragma unroll
                for(int i=0;i<8;++i) {
                    const int index=keyrow*256+dot_lane*16+i*2;
                    const float2 key=paired_half(shared_keys+key_swizzle(index));
                    p[i*2]=__fmul_rn(q[i*2],__fmul_rn(key.x,key_scale));
                    p[i*2+1]=__fmul_rn(q[i*2+1],__fmul_rn(key.y,key_scale));
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
            for(int j=0;j<4;++j) {
                const int dim=j*64+lane*2;
                #define V(i) weighted_value(shared_values,probabilities[warp],i,dim,value_scale)
                #define G(i) add_pair(add_pair(add_pair(V(i),V(i+8)),V(i+16)),V(i+24))
                const float2 contribution=add_pair(
                    add_pair(add_pair(G(0),G(1)),add_pair(G(4),G(5))),
                    add_pair(add_pair(G(2),G(3)),add_pair(G(6),G(7))));
                #undef G
                #undef V
                float a=0.0f,b=0.0f;
                switch(j){case 0:a=numerator0;b=numerator1;break;case 1:a=numerator2;b=numerator3;break;case 2:a=numerator4;b=numerator5;break;case 3:a=numerator6;b=numerator7;break;}
                const float alpha=state[warp][2];
                a=__fadd_rn(__fmul_rn(a,alpha),contribution.x);
                b=__fadd_rn(__fmul_rn(b,alpha),contribution.y);
                switch(j){case 0:numerator0=a;numerator1=b;break;case 1:numerator2=a;numerator3=b;break;case 2:numerator4=a;numerator5=b;break;case 3:numerator6=a;numerator7=b;break;}
            }
        }
        __syncthreads();
    }
    output[row*256+0*64+lane*2+0]=active?numerator0/state[warp][1]:0.0f;
    output[row*256+0*64+lane*2+1]=active?numerator1/state[warp][1]:0.0f;
    output[row*256+1*64+lane*2+0]=active?numerator2/state[warp][1]:0.0f;
    output[row*256+1*64+lane*2+1]=active?numerator3/state[warp][1]:0.0f;
    output[row*256+2*64+lane*2+0]=active?numerator4/state[warp][1]:0.0f;
    output[row*256+2*64+lane*2+1]=active?numerator5/state[warp][1]:0.0f;
    output[row*256+3*64+lane*2+0]=active?numerator6/state[warp][1]:0.0f;
    output[row*256+3*64+lane*2+1]=active?numerator7/state[warp][1]:0.0f;
}
