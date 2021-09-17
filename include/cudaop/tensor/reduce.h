/*******************************************************************************
 *  FILENAME:      reduce.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Thursday May 13th 2021
 *
 *  LAST MODIFIED: Thursday, July 22nd 2021, 10:17:26 am
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *
 *  REFERENCE:     https://developer.download.nvidia.cn/assets/cuda/files/reduction.pdf
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_TENSOR_REDUCE_H__
#define __SMARTMORE_CUDAOP_TENSOR_REDUCE_H__

#include <cudaop/common/utils.h>
#include <cudaop/generic/atomic.h>
#include <cudaop/generic/halfoperators.h>
#include <cudaop/type_traits/data_type_traits.h>
#include <cudaop/types.h>

namespace {
const int kBlockDim = 1024;
}

namespace smartmore {
namespace cudaop {
template <typename T, ReduceType reduce_type>
static inline __device__ auto CudaReduceOperate(T in1, T in2) -> enable_if_t<reduce_type == ReduceType::kSum, T> {
    return in1 + in2;
}

template <typename T, ReduceType reduce_type>
static inline __device__ auto CudaReduceOperate(T in1, T in2) -> enable_if_t<reduce_type == ReduceType::kMax, T> {
    return in1 > in2 ? in1 : in2;
}

template <typename T, ReduceType reduce_type>
static inline __device__ auto CudaReduceOperate(T in1, T in2) -> enable_if_t<reduce_type == ReduceType::kMin, T> {
    return in1 > in2 ? in2 : in1;
}

template <typename T, ReduceType reduce_type>
static inline __device__ auto CudaReduceAtomicOperate(T *addr, T value)
    -> enable_if_t<reduce_type == ReduceType::kSum, void> {
    atomicAdd(addr, value);
}

template <typename T, ReduceType reduce_type>
static inline __device__ auto CudaReduceAtomicOperate(T *addr, T value)
    -> enable_if_t<reduce_type == ReduceType::kMax, void> {
    atomicMax(addr, value);
}

template <typename T, ReduceType reduce_type>
static inline __device__ auto CudaReduceAtomicOperate(T *addr, T value)
    -> enable_if_t<reduce_type == ReduceType::kMin, void> {
    atomicMin(addr, value);
}

template <typename T, ReduceType reduce_type>
__global__ void CudaReduceKernel(void *input_data, int lenth, void *output_data) {
    __shared__ T sdata[kBlockDim];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lenth) {
        return;
    }
    sdata[threadIdx.x] = static_cast<T *>(input_data)[idx];
    __syncthreads();

    for (int s = 1; threadIdx.x + s < lenth && threadIdx.x + s < blockDim.x; s *= 2) {
        if (threadIdx.x % (2 * s) == 0) {
            sdata[threadIdx.x] = CudaReduceOperate<T, reduce_type>(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        CudaReduceAtomicOperate<T, reduce_type>(static_cast<T *>(output_data), sdata[0]);
    }
    return;
}

template <DataType data_type, ReduceType reduce_type>
void CudaReduce(void *input_data, int lenth, void *output_data, cudaStream_t stream) {
#if __CUDA_ARCH__ < 700
    static_assert(data_type != DataType::kHalf, "CudaReduce does not support kHalf with __CUDA_ARCH__ < 700");
#endif

    // forbid call for kInt8 kSum
    static_assert(data_type != DataType::kInt8 || reduce_type != ReduceType::kSum,
                  "CudaReduce does not support sum of int8");
    dim3 block(kBlockDim);
    dim3 grid((lenth + block.x - 1) / block.x);

    CudaReduceKernel<typename DataTypeTraits<data_type>::Type, reduce_type>
        <<<grid, block, 0, stream>>>(input_data, lenth, output_data);
}

template <DataType data_type, ReduceType reduce_type>
void Reduce(void *input_data, int lenth, void *output_data) {
    CudaReduce<data_type, reduce_type>(input_data, lenth, output_data, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}

template <DataType data_type, ReduceType reduce_type>
void ReduceAsync(void *input_data, int lenth, void *output_data, cudaStream_t stream) {
    CudaReduce<data_type, reduce_type>(input_data, lenth, output_data, stream);
}
}  // namespace cudaop
}  // namespace smartmore

#endif