/*******************************************************************************
 *  FILENAME:      threshold.h
 *
 *  AUTHORS:       Hou Yue    START DATE: Sunday August 8th 2021
 *
 *  LAST MODIFIED: Monday, August 9th 2021, 10:18:15 am
 *
 *  CONTACT:       yue.hou@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_TENSOR_THRESHOLD_H__
#define __SMARTMORE_CUDAOP_TENSOR_THRESHOLD_H__

#include <cudaop/common/utils.h>
#include <cudaop/type_traits/data_type_traits.h>
#include <cudaop/types.h>

namespace smartmore {
namespace cudaop {
template <typename T, ThreshType thresh_type>
static inline __device__ auto CudaThresholdKernalOperate(T *src, T *dst, double thresh, double maxval, int index)
    -> enable_if_t<thresh_type == ThreshType::kThresh_Binary, void> {
    if (src[index] > static_cast<T>(thresh)) {
        dst[index] = static_cast<T>(maxval);
    } else {
        dst[index] = 0;
    }
}

template <typename T, ThreshType thresh_type>
static inline __device__ auto CudaThresholdKernalOperate(T *src, T *dst, double thresh, double maxval, int index)
    -> enable_if_t<thresh_type == ThreshType::kThresh_Binary_INV, void> {
    if (src[index] > static_cast<T>(thresh)) {
        dst[index] = 0;
    } else {
        dst[index] = static_cast<T>(maxval);
    }
}

template <typename T, ThreshType thresh_type>
static inline __device__ auto CudaThresholdKernalOperate(T *src, T *dst, double thresh, double maxval, int index)
    -> enable_if_t<thresh_type == ThreshType::kThresh_Trunc, void> {
    if (src[index] > static_cast<T>(thresh)) {
        dst[index] = static_cast<T>(thresh);
    } else {
        dst[index] = src[index];
    }
}

template <typename T, ThreshType thresh_type>
static inline __device__ auto CudaThresholdKernalOperate(T *src, T *dst, double thresh, double maxval, int index)
    -> enable_if_t<thresh_type == ThreshType::kThresh_ToZero, void> {
    if (src[index] > static_cast<T>(thresh)) {
        dst[index] = src[index];
    } else {
        dst[index] = 0;
    }
}

template <typename T, ThreshType thresh_type>
static inline __device__ auto CudaThresholdKernalOperate(T *src, T *dst, double thresh, double maxval, int index)
    -> enable_if_t<thresh_type == ThreshType::kThresh_ToZero_INV, void> {
    if (src[index] > static_cast<T>(thresh)) {
        dst[index] = 0;
    } else {
        dst[index] = src[index];
    }
}

template <typename T, ThreshType thresh_type>
__global__ void CudaThresholdKernal(void *src, void *dst, unsigned int length, double thresh, double maxval) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= length) return;
    CudaThresholdKernalOperate<T, thresh_type>(static_cast<T *>(src), static_cast<T *>(dst), thresh, maxval, index);
}

template <DataType data_type, ThreshType thresh_type>
void CudaThreshold(void *src, void *dst, unsigned int length, double thresh, double maxval, cudaStream_t stream) {
    CUDAOP_ASSERT_TRUE(src != nullptr && dst != nullptr);
    dim3 block(1024);
    dim3 grid((length + block.x - 1) / block.x);
    CudaThresholdKernal<typename DataTypeTraits<data_type>::Type, thresh_type>
        <<<grid, block, 0, stream>>>(src, dst, length, thresh, maxval);
}

template <DataType data_type, ThreshType thresh_type>
void Threshold(void *src, void *dst, unsigned int length, double thresh, double maxval) {
    CudaThreshold<data_type, thresh_type>(src, dst, length, thresh, maxval, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
    return;
}

template <DataType data_type, ThreshType thresh_type>
void ThresholdAsync(void *src, void *dst, unsigned int length, double thresh, double maxval, cudaStream_t stream) {
    CudaThreshold<data_type, thresh_type>(src, dst, length, thresh, maxval, stream);
    return;
}
}  // namespace cudaop
}  // namespace smartmore

#endif