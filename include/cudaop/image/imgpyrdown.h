/*******************************************************************************
 *  FILENAME:      imgpyrdown.h
 *
 *  AUTHORS:       Sun Yucheng    START DATE: Tuesday September 14th 2021
 *
 *  LAST MODIFIED: Friday, September 17th 2021, 10:10:39 am
 *
 *  CONTACT:       yucheng.sun@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMG_PYRDOWN_H__
#define __SMARTMORE_CUDAOP_IMG_PYRDOWN_H__

#include <cuda_runtime_api.h>
#include <cudaop/types.h>

#include <stdexcept>
#include <vector>

#include "cudaop/common/utils.h"
#include "cudaop/image/generic/border.h"

namespace smartmore {
namespace cudaop {
static inline __device__ int getKernel(int n) {
    int div_sum = 1;
#pragma unroll
    for (size_t i = 1; i <= n; i++) {
        div_sum *= i;
    }
#pragma unroll
    for (size_t i = 4 - n; i >= 1; i--) {
        div_sum *= i;
    }

    return (24 / div_sum);
}

template <ImageType image_type, DataType data_type, BorderType border_type>
static __global__ void PyrDownKernel(unsigned char *src, unsigned char *dst, int in_h, int in_w, int dst_h, int dst_w) {
    const int x_dst = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_dst = blockIdx.y * blockDim.y + threadIdx.y;

    extern __shared__ int shared_kel[];

    if (threadIdx.x < 5 && threadIdx.y == 0) {
        shared_kel[25 + threadIdx.x] = getKernel(threadIdx.x);
    }
    __syncthreads();

    if (threadIdx.x < 5 && threadIdx.y < 5) {
        shared_kel[threadIdx.y * 5 + threadIdx.x] = shared_kel[25 + threadIdx.x] * shared_kel[25 + threadIdx.y];
    }
    __syncthreads();

    if (x_dst < dst_w && y_dst < dst_h) {
        float result = 0;
        const int x_ori = 2 * x_dst;
        const int y_ori = 2 * y_dst;

        for (int i = -2; i < 3; i++) {
#pragma unroll
            for (int j = -2; j < 3; j++) {
                result += src[GetBorderIndex<border_type>(x_ori + i, in_w) +
                              GetBorderIndex<border_type>(y_ori + j, in_h) * in_w] *
                          shared_kel[i + 2 + (j + 2) * 5];
            }
        }

        dst[x_dst + y_dst * dst_w] = static_cast<int>(result / 256);
    }
}

template <ImageType image_type, DataType data_type, BorderType border_type>
void CudaImagePyrDown(unsigned char *src, unsigned char *dst, int in_h, int in_w, cudaStream_t stream) {
    static_assert(image_type == ImageType::kGRAY, "Pyramids down only supports ImageType::kGRAY");
    static_assert(data_type == DataType::kInt8, "Pyramids down only supports DataType::kInt8.");
    static_assert(border_type == BorderType::kReplicate, "Pyramids down only supports BorderType::kReplicate.");

    constexpr dim3 block(32, 16);

    const int dst_img_h = (in_h + 1) / 2;
    const int dst_img_w = (in_w + 1) / 2;
    const int grid_x = (dst_img_w + block.x - 1) / block.x;
    const int grid_y = (dst_img_h + block.y - 1) / block.y;
    const dim3 grid(grid_x, grid_y);

    PyrDownKernel<image_type, data_type, border_type>
        <<<grid, block, 30 * sizeof(int32_t), stream>>>(src, dst, in_h, in_w, dst_img_h, dst_img_w);

    return;
}

template <ImageType image_type, DataType data_type, BorderType border_type>
void ImagePyrDown(unsigned char *src, unsigned char *dst, int in_h, int in_w) {
    CudaImagePyrDown<image_type, data_type, border_type>(src, dst, in_h, in_w, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}

template <ImageType image_type, DataType data_type, BorderType border_type>
void ImagePyrDownAsync(unsigned char *src, unsigned char *dst, int in_h, int in_w, cudaStream_t stream) {
    CudaImagePyrDown<image_type, data_type, border_type>(src, dst, in_h, in_w, stream);
}
}  // namespace cudaop
}  // namespace smartmore

#endif  // __SMARTMORE_CUDAOP_IMG_PYRDOWN_H__