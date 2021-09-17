/******************************************************************************
 * FILENAME:      concat.cu
 *
 * AUTHORS:       Yutong Huang
 *
 * LAST MODIFIED: Thu 18 Mar 2021 01:45:55 PM CST
 *
 * CONTACT:       yutong.huang@smartmore.com
 ******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_TENSOR_CONCAT_H__
#define __SMARTMORE_CUDAOP_TENSOR_CONCAT_H__

#include <cudaop/common/utils.h>
#include <cudaop/cudaop.h>

#include <stdexcept>
#include <vector>

namespace smartmore {
namespace cudaop {

template <typename DataType>
__global__ void CudaConcatLeftKernel(DataType *dst, DataType *lhs, size_t lhs_chunk_size, size_t whole_size) {
    size_t lhs_index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_chunk = lhs_index / lhs_chunk_size;
    size_t dst_index = num_chunk * whole_size + lhs_index % lhs_chunk_size;
    dst[dst_index] = lhs[lhs_index];
}

template <typename DataType>
__global__ void CudaConcatRightKernel(DataType *dst, DataType *rhs, size_t rhs_chunk_size, size_t whole_size) {
    size_t rhs_index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_chunk = rhs_index / rhs_chunk_size;
    size_t dst_index = num_chunk * whole_size + rhs_index % rhs_chunk_size + whole_size - rhs_chunk_size;
    dst[dst_index] = rhs[rhs_index];
}

template <typename DataType>
void CudaConcat(DataType *lhs, DataType *rhs, DataType *dst, long axis, const std::vector<size_t> &lhs_shape,
                const std::vector<size_t> &rhs_shape, cudaStream_t stream) {
    if (lhs_shape.size() != rhs_shape.size()) {
        throw std::runtime_error("Shape not equal");
    }
    for (int i = 0; i < lhs_shape.size(); i++) {
        if (i != axis && lhs_shape[i] != rhs_shape[i]) {
            throw std::runtime_error("Shape not equal");
        }
    }
    if (axis < 0) {
        axis = lhs_shape.size() + axis;
    }
    unsigned long num_chunks = 1;
    for (int i = 0; i < axis; i++) {
        num_chunks *= lhs_shape[i];
    }
    unsigned long lhs_chunk_size = 1;
    unsigned long rhs_chunk_size = 1;
    for (int i = axis; i < lhs_shape.size(); i++) {
        lhs_chunk_size *= lhs_shape[i];
    }
    for (int i = axis; i < rhs_shape.size(); i++) {
        rhs_chunk_size *= rhs_shape[i];
    }
    unsigned long whole_chunk_size = lhs_chunk_size + rhs_chunk_size;
    dim3 block_left(1024);
    dim3 block_right(1024);
    dim3 grid_left((lhs_chunk_size * num_chunks + block_left.x - 1) / block_left.x);
    dim3 grid_right((rhs_chunk_size * num_chunks + block_right.x - 1) / block_right.x);
    CudaConcatLeftKernel<<<grid_left, block_left, 0, stream>>>(dst, lhs, lhs_chunk_size, whole_chunk_size);
    CudaConcatRightKernel<<<grid_right, block_right, 0, stream>>>(dst, rhs, rhs_chunk_size, whole_chunk_size);
}

template <typename DataType>
void ConcatAsync(DataType *lhs, DataType *rhs, DataType *dst, long axis, const std::vector<size_t> &lhs_shape,
                 const std::vector<size_t> &rhs_shape, cudaStream_t stream) {
    CudaConcat(lhs, rhs, dst, axis, lhs_shape, rhs_shape, stream);
}

template <typename DataType>
void Concat(DataType *lhs, DataType *rhs, DataType *dst, long axis, const std::vector<size_t> &lhs_shape,
            const std::vector<size_t> &rhs_shape) {
    CudaConcat(lhs, rhs, dst, axis, lhs_shape, rhs_shape, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}
}  // namespace cudaop
}  // namespace smartmore

#endif