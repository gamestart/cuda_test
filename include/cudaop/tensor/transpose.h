/*******************************************************************************
 *  FILENAME:      transpose.h
 *
 *  AUTHORS:       Wang Shengxiang    START DATE: Friday September 3rd 2021
 *
 *  LAST MODIFIED: Friday, September 3rd 2021, 3:25:58 pm
 *
 *  CONTACT:       shengxiang.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_TENSOR_TRANSPOSE_H__
#define __SMARTMORE_CUDAOP_TENSOR_TRANSPOSE_H__

#include <cuda_runtime_api.h>
#include <cudaop/common/utils.h>
#include <cudaop/type_traits/data_type_traits.h>
#include <cudaop/types.h>

namespace smartmore {
namespace cudaop {
template <DataType data_type>
using DataType_t = typename DataTypeTraits<data_type>::Type;

template <DataType data_type>
static __global__ void TransposeKernel(void *src, void *dst, int in_h, int in_w) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx >= in_w || idy >= in_h) return;

    *(static_cast<DataType_t<data_type> *>(dst) + idy + in_h * idx) =
        *(static_cast<DataType_t<data_type> *>(src) + idx + in_w * idy);

    return;
}

template <DataType data_type>
void CudaTranspose(void *src, void *dst, int in_h, int in_w, cudaStream_t stream) {
    dim3 block(32, 16);
    int grid_x = (in_w + block.x - 1) / block.x;
    int grid_y = (in_h + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);
    TransposeKernel<data_type><<<grid, block, 0, stream>>>(src, dst, in_h, in_w);
    return;
}

template <DataType data_type>
void Transpose(void *src, void *dst, int in_h, int in_w) {
    CudaTranspose<data_type>(src, dst, in_h, in_w, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
    return;
}

template <DataType data_type>
void TransposeAsync(void *src, void *dst, int in_h, int in_w, cudaStream_t stream) {
    CudaTranspose<data_type>(src, dst, in_h, in_w, stream);
    return;
}
}  // namespace cudaop
}  // namespace smartmore
#endif