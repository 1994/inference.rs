#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Exact same FP8 pair widening as the production BF16 bridge.
__device__ __forceinline__ unsigned widen_pair(unsigned short raw) {
    union { __half2_raw half; unsigned bits; } packed;
    packed.half=__nv_cvt_fp8x2_to_halfraw2(raw,__NV_E4M3);
    if((raw&0x7fu)==0x7fu)
        packed.bits=(packed.bits&0xffff0000u)|0x5f80u|((unsigned(raw)&0x80u)<<8u);
    if((raw&0x7f00u)==0x7f00u)
        packed.bits=(packed.bits&0x0000ffffu)|0x5f800000u|((unsigned(raw)&0x8000u)<<16u);
    return packed.bits;
}

__device__ __forceinline__ unsigned short f16_rn(float x) {
    return __half_as_ushort(__float2half_rn(x));
}

// Minimal Float16 adaptation of prepare_fa4: same indexing, same scale
// placement, same reserved-code handling, same used/end metadata contract.
extern "C" __global__ void prepare_fa4_f16(
    const float* q,const unsigned char* k,const unsigned char* v,
    const int* metadata,const int* table,unsigned* qo,unsigned* ko,unsigned* vo,
    int* used,unsigned capacity,unsigned lanes,float ks,float vs) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t stride=size_t(gridDim.x)*blockDim.x;
    const int count=max(0,min(int(lanes),metadata[1]));
    const int end=count>0?max(0,min(int(capacity),metadata[0]+metadata[2]+count)):0;
    if(index==0) { used[0]=count;used[1]=end; }
    for(size_t pair=index;pair<size_t(lanes)*24*128;pair+=stride)
        qo[pair]=f16_rn(q[pair*2])|(unsigned(f16_rn(q[pair*2+1]))<<16);
    for(size_t pair=index;pair<size_t(end)*4*128;pair+=stride) {
        const unsigned position=unsigned(pair/(4*128));
        const unsigned head=unsigned(pair/128)%4,dim_pair=unsigned(pair)%128;
        const unsigned physical=unsigned(table[position/32])*32+position%32;
        const size_t input=(size_t(head)*capacity+physical)*128+dim_pair;
        union { unsigned bits; __half2 value; } kh,vh;
        kh.bits=widen_pair(reinterpret_cast<const unsigned short*>(k)[input]);
        vh.bits=widen_pair(reinterpret_cast<const unsigned short*>(v)[input]);
        ko[pair]=f16_rn(__fmul_rn(__low2float(kh.value),ks))|
            (unsigned(f16_rn(__fmul_rn(__high2float(kh.value),ks)))<<16);
        vo[pair]=f16_rn(__fmul_rn(__low2float(vh.value),vs))|
            (unsigned(f16_rn(__fmul_rn(__high2float(vh.value),vs)))<<16);
    }
}

// Minimal Float16 adaptation of finish_fa4: read FP16, widen to F32, and
// preserve the exact inactive-row zeroing contract.
extern "C" __global__ void finish_fa4_f16(
    const unsigned short* input,float* output,const int* metadata,unsigned lanes) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=size_t(lanes)*24*256) return;
    const int row=int(index/(24*256)),position=metadata[0]+metadata[2]+row;
    if(row<metadata[1] && position>=0) {
        output[index]=__half2float(__ushort_as_half(input[index]));
    } else {
        output[index]=0.0f;
    }
}
