/*******************************************************************************
 *  FILENAME:      imgsqrt.h
 *
 *  AUTHORS:       Chen Fting    START DATE: Thursday August 5th 2021
 *
 *  LAST MODIFIED: Wednesday, August 11th 2021, 11:14:14 am
 *
 *  CONTACT:       fting.chen@smartmore.com
 *******************************************************************************/
#ifndef __SMARTMORE_CUDAOP_IMAGE_SQRT_H__
#define __SMARTMORE_CUDAOP_IMAGE_SQRT_H__

#include <cuda_runtime_api.h>
#include <cudaop/common/utils.h>
#include <cudaop/types.h>

namespace smartmore {
namespace cudaop {
template <int tile>
__global__ void SqrtKernel(void *in, void *out, int length) {
#pragma unroll
    for (int i = 0; i < tile; i++) {
        //访存合并
        int idx = threadIdx.x + blockIdx.x * blockDim.x + i * blockDim.x * gridDim.x;
        //没有访存合并
        // int idx = threadIdx.x * tile + blockIdx.x * blockDim.x * tile + i;
        if (idx >= length) {
            return;
        }
        static_cast<float *>(out)[idx] = sqrt(static_cast<float *>(in)[idx]);
    }
}

template <DataType data_type>
void CudaSqrt(void *in, void *out, int length, cudaStream_t stream) {
    const int thread_per_block = 512;

    const int tile = 2;
    int grid_x = (length + (thread_per_block * tile) - 1) / (thread_per_block * tile);
    SqrtKernel<tile><<<grid_x, thread_per_block, 0, stream>>>(in, out, length);
}

template <DataType data_type>
void Sqrt(void *in, void *out, int length) {
    CudaSqrt<data_type>(in, out, length, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}

template <DataType data_type>
void SqrtAsync(void *in, void *out, int length, cudaStream_t stream) {
    CudaSqrt<data_type>(in, out, length, stream);
}

}  // namespace cudaop
}  // namespace smartmore

#endif