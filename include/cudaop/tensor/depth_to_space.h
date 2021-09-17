/******************************************************************************
 * FILENAME:      depth_to_space.h
 *
 * AUTHORS:       Yutong Huang
 *
 * LAST MODIFIED: Wed 24 Mar 2021 01:59:55 PM CST
 *
 * CONTACT:       yutong.huang@smartmore.com
 ******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_TENSOR_DEPTH_TO_SPACE_H__
#define __SMARTMORE_CUDAOP_TENSOR_DEPTH_TO_SPACE_H__

#include <cuda_runtime_api.h>

#include <stdexcept>

namespace smartmore {
namespace cudaop {
template <typename DataType, bool is_NCHW, unsigned int block_size, unsigned int in_C, unsigned int in_H,
          unsigned int in_W>
__global__ void CudaDepthToSpaceKernel(DataType *src, DataType *dst, unsigned int n) {
    const unsigned int out_H = in_H * block_size;
    const unsigned int out_W = in_W * block_size;
    const unsigned int out_C = in_C / (block_size * block_size);
    const unsigned int h = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int w = blockIdx.y * blockDim.y + threadIdx.y;
    if (h >= out_H || w >= out_W) {
        return;
    }
    const unsigned int src_n = n;
    const unsigned int src_h = h / block_size;
    const unsigned int src_w = w / block_size;
    if (is_NCHW) {
        for (int c = 0; c < out_C; c++) {
            const unsigned int src_c = h % block_size * block_size + w % block_size + c * block_size * block_size;

            const unsigned int src_index = src_n * in_C * in_H * in_W + src_c * in_H * in_W + src_h * in_W + src_w;

            const unsigned int dst_index = n * out_C * out_H * out_W + c * out_H * out_W + h * out_W + w;

            dst[dst_index] = src[src_index];
        }
    } else {
        for (int c = 0; c < out_C; c++) {
            const unsigned int src_c = h % block_size * block_size + w % block_size + c * block_size * block_size;

            const unsigned int src_index = src_n * in_H * in_W * in_C + src_h * in_W * in_C + src_w * in_C + src_c;

            const unsigned int dst_index = n * out_H * out_W * out_C + h * out_W * out_C + w * out_C + c;

            dst[dst_index] = src[src_index];
        }
    }
}

template <typename DataType, bool is_NCHW, unsigned int block_size, unsigned int in_N, unsigned int in_C,
          unsigned int in_H, unsigned int in_W>
void CudaDepthToSpace(DataType *src, DataType *dst, cudaStream_t stream) {
    static_assert(in_C % (block_size * block_size) == 0, "Channel not divisible by block size");
    for (int n = 0; n < in_N; n++) {
        dim3 block(32, 32);
        dim3 grid((in_H * block_size + 31) / 32, (in_W * block_size + 31) / 32);
        CudaDepthToSpaceKernel<DataType, is_NCHW, block_size, in_C, in_H, in_W>
            <<<grid, block, 0, stream>>>(src, dst, n);
    }
}

template <typename DataType, bool is_NCHW, unsigned int block_size, unsigned int in_N, unsigned int in_C,
          unsigned int in_H, unsigned int in_W>
void DepthToSpace(DataType *src, DataType *dst) {
    CudaDepthToSpace<DataType, is_NCHW, block_size, in_N, in_C, in_H, in_W>(src, dst, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}

template <typename DataType, bool is_NCHW, unsigned int block_size, unsigned int in_N, unsigned int in_C,
          unsigned int in_H, unsigned int in_W>
void DepthToSpaceAsync(DataType *src, DataType *dst, cudaStream_t stream) {
    CudaDepthToSpace<DataType, is_NCHW, block_size, in_N, in_C, in_H, in_W>(src, dst, stream);
}
}  // namespace cudaop
}  // namespace smartmore

#endif  // __SMARTMORE_DEPTH_TO_SPACE_H__
