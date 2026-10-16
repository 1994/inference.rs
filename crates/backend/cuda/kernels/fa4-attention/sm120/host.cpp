// Host ABI for CuTe-generated launch code; all CUDA calls use the installed toolkit runtime.
#include "attention_host.h"
#include <cuda.h>

extern "C" {
int _cudaDeviceGetAttribute(int* v,int a,int d) {
    return int(cudaDeviceGetAttribute(v,cudaDeviceAttr(a),d));
}
int _cudaGetDevice(int* d) { return int(cudaGetDevice(d)); }
int _cudaFuncSetAttribute(const void* f,int a,int v) {
    return int(cudaFuncSetAttribute(f,cudaFuncAttribute(a),v));
}
int _cudaKernelSetAttributeForDevice(cudaKernel_t k,int a,int v,int d) {
    return int(cudaKernelSetAttributeForDevice(k,cudaFuncAttribute(a),v,d));
}
int _cuKernelGetAttribute(int* v,int a,CUkernel k,int d) {
    CUdevice device;
    CUresult result=cuDeviceGet(&device,d);
    return result==CUDA_SUCCESS?int(cuKernelGetAttribute(v,CUfunction_attribute(a),k,device)):int(result);
}
int _cudaLibraryLoadData(cudaLibrary_t* l,const void* c,cudaJitOption* j,void** jv,unsigned nj,
    cudaLibraryOption* o,void** ov,unsigned no) {
    return int(cudaLibraryLoadData(l,c,j,jv,nj,o,ov,no));
}
int _cudaLibraryGetKernel(cudaKernel_t* k,cudaLibrary_t l,const char* n) {
    return int(cudaLibraryGetKernel(k,l,n));
}
int _cudaLaunchKernelEx(const cudaLaunchConfig_t* c,const void* f,void** a) {
    return int(cudaLaunchKernelExC(c,f,a));
}
}

static infer_fa4_sm120_Kernel_Module_t module;

extern "C" __attribute__((visibility("default"))) int infer_fa4_initialize(int device) {
    int result=0;
    cudaLibrary_t* library=&module.module;
    if(module.module==nullptr) {
        void* args[]={&library,&result};
        _mlir_infer_fa4_sm120_cuda_init(args);
        if(result) return result;
    }
    void* args[]={&library,&device,&result};
    _mlir_infer_fa4_sm120_cuda_load_to_device(args);
    return result;
}

extern "C" __attribute__((visibility("default"))) int infer_fa4_forward(
    void* q,void* k,void* v,void* o,void* used_q,void* used_k,
    int lanes,int capacity,float scale,cudaStream_t stream) {
    infer_fa4_sm120_Tensor_q_t q_arg={q,{lanes},{int64_t(lanes)*24*256}};
    infer_fa4_sm120_Tensor_k_t k_arg={k,{capacity},{int64_t(capacity)*4*256}};
    infer_fa4_sm120_Tensor_v_t v_arg={v,{capacity},{int64_t(capacity)*4*256}};
    infer_fa4_sm120_Tensor_o_t o_arg={o,{lanes},{int64_t(lanes)*24*256}};
    infer_fa4_sm120_Tensor_usedq_t used_q_arg={used_q};
    infer_fa4_sm120_Tensor_usedk_t used_k_arg={used_k};
    return cute_dsl_infer_fa4_sm120_wrapper(&module,&q_arg,&k_arg,&v_arg,&o_arg,
        &used_q_arg,&used_k_arg,scale,stream);
}
