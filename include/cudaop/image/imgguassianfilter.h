/******************************************************************************
 * FILENAME:      imgguassianfilter.h
 *
 * AUTHORS:       Yutong Huang
 *
 * LAST MODIFIED: Wed 09 Jun 2021 04:31:47 PM CST
 *
 * CONTACT:       yutong.huang@smartmore.com
 ******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMG_GUASSIAN_FILTER_H__
#define __SMARTMORE_CUDAOP_IMG_GUASSIAN_FILTER_H__

#include <cuda_runtime_api.h>
#include <cudaop/types.h>

#include <stdexcept>
#include <vector>

#include "cudaop/common/utils.h"
#include "cudaop/image/generic/border.h"

namespace smartmore {
namespace cudaop {

template <int kernel_size>
static inline __device__ void GenerateGaussianKernel(float *kernel, float sigma, int i) {
    constexpr int radius = kernel_size / 2;
    constexpr float pi = 3.14159274101257324f;
    const float constant = 1.0 / (sigma * sqrt(pi * 2));

    kernel[i] = constant * (1 / exp(((i - radius) * (i - radius)) / (2 * sigma * sigma)));
}

template <ImageType image_type, DataType data_type, BorderType border_type, int kernel_h, int kernel_w, int segment_h,
          int segment_w, int segment_data_per_thread, int rows_per_thread>
static __global__ void GaussianFilterReplicateKernel(unsigned char *src, unsigned char *dst, int in_h, int in_w,
                                                     float sigma_h, float sigma_w) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;

    const int local_w = threadIdx.x;
    const int local_h = threadIdx.y;

    const int segment_start_h = blockIdx.y * blockDim.y - (kernel_h >> 1);
    const int segment_start_w = blockIdx.x * blockDim.x - (kernel_w >> 1);

    extern __shared__ char shared_mem[];
    float *hkernel = (float *)shared_mem;
    float *wkernel = hkernel + kernel_h;
    unsigned char *segment = (unsigned char *)(wkernel + kernel_w);
    float *rows = (float *)(segment + segment_h * segment_w);
    float *weights = rows + blockDim.x * segment_h;

    if (tid < kernel_h) {
        GenerateGaussianKernel<kernel_h>(hkernel, sigma_h, tid);
    } else if (tid < kernel_h + kernel_w) {
        GenerateGaussianKernel<kernel_w>(wkernel, sigma_w, tid - kernel_h);
    }

#pragma unroll
    for (int i = 0; i < segment_data_per_thread; i++) {
        const int segment_index = tid * segment_data_per_thread + i;
        if (segment_index >= segment_h * segment_w) {
            continue;
        }
        const int src_h = segment_index / segment_w + segment_start_h;
        const int src_w = segment_index % segment_w + segment_start_w;
        segment[segment_index] =
            src[GetBorderIndex<border_type>(src_h, in_h) * in_w + GetBorderIndex<border_type>(src_w, in_w)];
    }

    __syncthreads();

#pragma unroll
    for (int i = threadIdx.y * rows_per_thread; i < rows_per_thread * (1 + threadIdx.y); i++) {
        if (i < segment_h) {
            float result_x = 0;
            float weight_x = 0;
#pragma unroll
            for (int x = 0; x < kernel_w; x++) {
                result_x += segment[i * segment_w + (local_w + x)] * wkernel[x];
                weight_x += wkernel[x];
            }
            rows[i * blockDim.x + local_w] = result_x;
            weights[i * blockDim.x + local_w] = weight_x;
        }
    }

    __syncthreads();

    if (h >= in_h || w >= in_w) return;

    float result = 0;
    float weight = 0;

#pragma unroll
    for (int i = 0; i < kernel_h; i++) {
        result += hkernel[i] * rows[(i + local_h) * blockDim.x + local_w];
        weight += hkernel[i] * weights[(i + local_h) * blockDim.x + local_w];
    }

    dst[h * in_w + w] = result / weight + 0.5f;
}

template <ImageType image_type, DataType data_type, BorderType border_type, int kernel_h, int kernel_w>
void CudaImageGaussianFilter(unsigned char *src, unsigned char *dst, int in_h, int in_w, float sigma_h, float sigma_w,
                             cudaStream_t stream) {
    static_assert(image_type == ImageType::kGRAY, "Gaussian filter only supports ImageType::kGRAY");
    static_assert(data_type == DataType::kInt8, "Gaussian filter only supports DataType::kInt8.");
    static_assert(border_type == BorderType::kReplicate, "Gaussian filter only supports BorderType::kReplicate.");
    static_assert(kernel_h % 2 == 1 && kernel_h >= 3, "Kernel size must be an odd number larger than 1");
    static_assert(kernel_w % 2 == 1 && kernel_w >= 3, "Kernel size must be an odd number larger than 1");

    if (in_h <= (kernel_h >> 1) || in_w <= (kernel_w >> 1)) {
        throw std::runtime_error("Kernel size too large to fit image");
    }

    constexpr dim3 block(16, 16);

    static_assert(kernel_h + kernel_w <= block.x * block.y, "Kernel size too large");

    const int grid_x = (in_w + block.x - 1) / block.x;
    const int grid_y = (in_h + block.y - 1) / block.y;
    const dim3 grid(grid_x, grid_y);

    constexpr int segment_w = block.x + kernel_w - 1;
    constexpr int segment_h = block.y + kernel_h - 1;

    constexpr int segment_data_per_thread = (segment_h * segment_w + block.x * block.y - 1) / (block.x * block.y);

    constexpr int rows_per_thread = (segment_h + block.y - 1) / block.y;

    constexpr int shmem_size = sizeof(float) * (kernel_h + kernel_w + block.x * segment_h * 2) + segment_h * segment_w;

    GaussianFilterReplicateKernel<image_type, data_type, border_type, kernel_h, kernel_w, segment_h, segment_w,
                                  segment_data_per_thread, rows_per_thread>
        <<<grid, block, shmem_size, stream>>>(src, dst, in_h, in_w, sigma_h, sigma_w);
    return;
}

template <ImageType image_type, DataType data_type, BorderType border_type, int kernel_h, int kernel_w>
void ImageGaussianFilter(unsigned char *src, unsigned char *dst, int in_h, int in_w, float sigma_h, float sigma_w) {
    CudaImageGaussianFilter<image_type, data_type, border_type, kernel_h, kernel_w>(src, dst, in_h, in_w, sigma_h,
                                                                                    sigma_w, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}

template <ImageType image_type, DataType data_type, BorderType border_type, int kernel_h, int kernel_w>
void ImageGaussianFilterAsync(unsigned char *src, unsigned char *dst, int in_h, int in_w, float sigma_h, float sigma_w,
                              cudaStream_t stream) {
    CudaImageGaussianFilter<image_type, data_type, border_type, kernel_h, kernel_w>(src, dst, in_h, in_w, sigma_h,
                                                                                    sigma_w, stream);
}
}  // namespace cudaop
}  // namespace smartmore

#endif  // __SMARTMORE_CUDAOP_IMG_GUASSIAN_FILTER_H__
