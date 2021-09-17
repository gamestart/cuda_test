/*******************************************************************************
 *  FILENAME:      meannormalization.h
 *
 *  AUTHORS:       Hou Yue    START DATE: Tuesday July 20th 2021
 *
 *  LAST MODIFIED: Monday, July 26th 2021, 2:29:07 pm
 *
 *  CONTACT:       yue.hou@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_MEANNORMALIZATION_H__
#define __SMARTMORE_CUDAOP_IMAGE_MEANNORMALIZATION_H__

#include <cuda_runtime_api.h>
#include <cudaop/common/utils.h>
#include <cudaop/image/generic/border.h>
#include <cudaop/types.h>

namespace smartmore {
namespace cudaop {
template <typename T>
__global__ void CudaMeanNormalizationKernal(T *src, T *dst, unsigned int length, const float mean,
                                            const float variance) {
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    if (id_x >= length) return;

    dst[id_x] = (src[id_x] - static_cast<T>(mean)) / static_cast<T>(variance);
}

template <typename T, DataType data_type>
void CudaMeanNormalization(void *src, void *dst, unsigned int length, const float mean, const float variance,
                           cudaStream_t stream) {
    static_assert(data_type != DataType::kInt8, "MeanNormalization does not supports DataType::kInt8");
    dim3 block(1024);
    dim3 grid((length + block.x - 1) / block.x);
    // normalization
    CudaMeanNormalizationKernal<T>
        <<<grid, block, 0, stream>>>(static_cast<T *>(src), static_cast<T *>(dst), length, mean, variance);
}

template <DataType data_type>
void MeanNormalization(void *src, void *dst, unsigned int length, const float mean, const float variance) {
    CudaMeanNormalization<typename DataTypeTraits<data_type>::Type, data_type>(src, dst, length, mean, variance, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}

template <DataType data_type>
void MeanNormalizationAsync(void *src, void *dst, unsigned int length, const float mean, const float variance,
                            cudaStream_t stream) {
    CudaMeanNormalization<data_type>(src, dst, length, mean, variance, stream);
}
}  // namespace cudaop
}  // namespace smartmore

#endif