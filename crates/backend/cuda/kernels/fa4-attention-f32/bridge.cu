#include <cuda_runtime.h>

__device__ __forceinline__ unsigned bf16_rn(float x) {
    const unsigned bits=__float_as_uint(x);
    return (bits+0x7fffu+((bits>>16)&1u))>>16;
}
// Match the existing preparation ABI. F32 cache storage has no FP8 descale.
extern "C" __global__ void prepare_fa4_f32(
    const float* q,const float* k,const float* v,
    const int* metadata,const int* table,unsigned* qo,unsigned* ko,unsigned* vo,
    int* used,unsigned capacity,unsigned lanes,float,float) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t stride=size_t(gridDim.x)*blockDim.x;
    const int count=max(0,min(int(lanes),metadata[1]));
    const int end=count>0?max(0,min(int(capacity),metadata[0]+metadata[2]+count)):0;
    if(index==0) { used[0]=count;used[1]=end; }
    for(size_t pair=index;pair<size_t(lanes)*24*128;pair+=stride)
        qo[pair]=bf16_rn(q[pair*2])|(bf16_rn(q[pair*2+1])<<16);
    for(size_t pair=index;pair<size_t(end)*4*128;pair+=stride) {
        const unsigned position=unsigned(pair/(4*128));
        const unsigned head=unsigned(pair/128)%4,dim_pair=unsigned(pair)%128;
        const unsigned physical=unsigned(table[position/32])*32+position%32;
        const size_t input=(size_t(head)*capacity+physical)*256+dim_pair*2;
        ko[pair]=bf16_rn(k[input])|(bf16_rn(k[input+1])<<16);
        vo[pair]=bf16_rn(v[input])|(bf16_rn(v[input+1])<<16);
    }
}
