#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
__device__ __forceinline__ unsigned widen_pair(unsigned short raw) {
    union { __half2_raw half; unsigned bits; } packed;
    packed.half=__nv_cvt_fp8x2_to_halfraw2(raw,__NV_E4M3);
    if((raw&0x7fu)==0x7fu)
        packed.bits=(packed.bits&0xffff0000u)|0x5f80u|((unsigned(raw)&0x80u)<<8u);
    if((raw&0x7f00u)==0x7f00u)
        packed.bits=(packed.bits&0x0000ffffu)|0x5f800000u|((unsigned(raw)&0x8000u)<<16u);
    return packed.bits;
}


__device__ __forceinline__ unsigned bf16_rn(float x) {
    const unsigned bits=__float_as_uint(x);
    return (bits+0x7fffu+((bits>>16)&1u))>>16;
}
// Fixed resident grid and GPU bounds avoid empty work proportional to model context.
extern "C" __global__ void prepare_fa4(
    const float* q,const unsigned char* k,const unsigned char* v,
    const int* metadata,const int* table,unsigned* qo,unsigned* ko,unsigned* vo,
    int* used,unsigned capacity,unsigned lanes,float ks,float vs) {
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
        const size_t input=(size_t(head)*capacity+physical)*128+dim_pair;
        union { unsigned bits; __half2 value; } kh,vh;
        kh.bits=widen_pair(reinterpret_cast<const unsigned short*>(k)[input]);
        vh.bits=widen_pair(reinterpret_cast<const unsigned short*>(v)[input]);
        ko[pair]=bf16_rn(__fmul_rn(__low2float(kh.value),ks))|
            (bf16_rn(__fmul_rn(__high2float(kh.value),ks))<<16);
        vo[pair]=bf16_rn(__fmul_rn(__low2float(vh.value),vs))|
            (bf16_rn(__fmul_rn(__high2float(vh.value),vs))<<16);
    }
}
extern "C" __global__ void finish_fa4(
    const unsigned short* input,float* output,const int* metadata,unsigned lanes) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=size_t(lanes)*24*256) return;
    const int row=int(index/(24*256)),position=metadata[0]+metadata[2]+row;
    output[index]=row<metadata[1] && position>=0?
        __uint_as_float(unsigned(input[index])<<16):0.0f;
}
