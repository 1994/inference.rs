
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>


// Macro to check for cuda errors.
#ifndef CUTE_DSL_CUDA_ERROR_CHECK
#define CUTE_DSL_CUDA_ERROR_CHECK(err) { \
    if ((err) != cudaSuccess) { \
        printf("Got Cuda Error %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
    } \
}

#endif

typedef struct {
    cudaLibrary_t module;
} infer_fa4_sm90a_Kernel_Module_t;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir_infer_fa4_sm90a_cuda_init(void **);
void _mlir_infer_fa4_sm90a_cuda_load_to_device(void **);
static inline void infer_fa4_sm90a_Kernel_Module_Load(infer_fa4_sm90a_Kernel_Module_t *module) {
    cudaLibrary_t *libraryPtr = &(module->module);
    cudaError_t ret = cudaSuccess;
    struct {
        cudaLibrary_t **libraryPtr;
        cudaError_t *ret;
    } initArgs = {&libraryPtr, &ret};
    _mlir_infer_fa4_sm90a_cuda_init((void **)(&initArgs));
    CUTE_DSL_CUDA_ERROR_CHECK(ret);
    int32_t device_id = 0;
    struct {
        cudaLibrary_t **library;
        int32_t *device_id;
        cudaError_t *ret;
    } loadArgs = {&libraryPtr, &device_id, &ret};
    int32_t device_count;
    CUTE_DSL_CUDA_ERROR_CHECK(cudaGetDeviceCount(&device_count));
    for (int32_t i = 0; i < device_count; i++) {
        device_id = i;
        _mlir_infer_fa4_sm90a_cuda_load_to_device((void **)(&loadArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
    }
}

static inline void infer_fa4_sm90a_Kernel_Module_Unload(infer_fa4_sm90a_Kernel_Module_t *module) {
    CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
}

#ifdef __cplusplus
}
#endif

typedef struct {
    void *data;
    int32_t dynamic_shapes[1];
    int64_t dynamic_strides[1];
} infer_fa4_sm90a_Tensor_q_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[1];
    int64_t dynamic_strides[1];
} infer_fa4_sm90a_Tensor_k_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[1];
    int64_t dynamic_strides[1];
} infer_fa4_sm90a_Tensor_v_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[1];
    int64_t dynamic_strides[1];
} infer_fa4_sm90a_Tensor_o_t;


typedef struct {
    void *data;
} infer_fa4_sm90a_Tensor_usedq_t;


typedef struct {
    void *data;
} infer_fa4_sm90a_Tensor_usedk_t;

#ifdef __cplusplus
extern "C"
#endif
void _mlir_infer_fa4_sm90a__mlir_ciface_cutlass_entry_FakeTensorBFloat161div6424256i64div39321661442561_FakeTensorBFloat161div324256i64div3276810242561_FakeTensorBFloat161div324256i64div3276810242561_FakeTensorBFloat161d(void **args, int32_t num_args);

static inline int32_t cute_dsl_infer_fa4_sm90a_wrapper(infer_fa4_sm90a_Kernel_Module_t *module, infer_fa4_sm90a_Tensor_q_t *q, infer_fa4_sm90a_Tensor_k_t *k, infer_fa4_sm90a_Tensor_v_t *v, infer_fa4_sm90a_Tensor_o_t *o, infer_fa4_sm90a_Tensor_usedq_t *usedq, infer_fa4_sm90a_Tensor_usedk_t *usedk, float scale, cudaStream_t stream) {
    int32_t ret = 0;
    void *args[9] = {
        q, k, v, o, usedq, usedk, &scale, &stream,
        &ret
    };
    _mlir_infer_fa4_sm90a__mlir_ciface_cutlass_entry_FakeTensorBFloat161div6424256i64div39321661442561_FakeTensorBFloat161div324256i64div3276810242561_FakeTensorBFloat161div324256i64div3276810242561_FakeTensorBFloat161d(args, 9);
    return ret;
}
