/*******************************************************************************
 *  FILENAME:      imgmedianfilter.h
 *
 *  AUTHORS:       Wang Shengxiang    START DATE: Saturday May 29th 2021
 *
 *  LAST MODIFIED: Thursday, September 16th 2021, 7:42:29 pm
 *
 *  CONTACT:       shengxiang.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_MEDIAN_FILTER_H__
#define __SMARTMORE_CUDAOP_IMAGE_MEDIAN_FILTER_H__

#include <cudaop/common/utils.h>
#include <cudaop/image/generic/border.h>
#include <cudaop/image/generic/sort.h>
#include <cudaop/types.h>

#include <iostream>

namespace smartmore {
namespace cudaop {
// 5-28, first version, only support gray channel, int 8
template <ImageType image_type, DataType data_type, BorderType border_type, int kernel_size>
static __global__ void MedianFilterKernel(unsigned char *src, unsigned char *dst, int in_h, int in_w) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx >= in_w || idy >= in_h) return;

    int arr_len = kernel_size * kernel_size;
    int index = idy * in_w + idx;
    int sort_pos = 0;

    unsigned char sort_arr[kernel_size * kernel_size];

    // stride : span width of each element
    int stride = kernel_size >> 1;

    for (int j = idy - stride; j <= idy + stride; j++) {
        // j stands for logic ordinate, mj --- real ordinate
        int mj = GetBorderIndex<border_type>(j, in_h);
        for (int i = idx - stride; i <= idx + stride; i++) {
            int mi = GetBorderIndex<border_type>(i, in_w);
            sort_arr[sort_pos] = src[mj * in_w + mi];
            sort_pos++;
        }
    }

    dst[index] = SelectMaxK(sort_arr, arr_len, kernel_size * kernel_size / 2);

    return;
}

template <ImageType image_type, DataType data_type, BorderType border_type, int kernel_size>
void CudaMedianFilter(unsigned char *src, unsigned char *dst, int in_h, int in_w, cudaStream_t stream) {
    static_assert(data_type == DataType::kInt8, "Meidan fileter only supports DataType::kInt8.");
    static_assert(border_type == BorderType::kReplicate, "Meidan fileter only supports BorderType::kReplicate.");
    // kernel_size must be an odd and larger than 1
    static_assert((kernel_size & 1) != 0 || kernel_size >= 3,
                  "Median filter kernel size must be an odd and large than 1.");

    if (in_h <= (kernel_size >> 1) || in_w <= (kernel_size >> 1)) {
        throw std::runtime_error("Median filter size is too large or image size is too small.");
    }

    dim3 block(32, 16);
    int grid_x = (in_w + block.x - 1) / block.x;
    int grid_y = (in_h + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);

    MedianFilterKernel<image_type, data_type, border_type, kernel_size>
        <<<grid, block, 0, stream>>>(src, dst, in_h, in_w);
    return;
}

template <ImageType image_type, DataType data_type, BorderType border_type, int kernel_size>
void MedianFilter(void *src, void *dst, int in_h, int in_w) {
    CudaMedianFilter<image_type, data_type, border_type, kernel_size>(static_cast<unsigned char *>(src),
                                                                      static_cast<unsigned char *>(dst), in_h, in_w, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
    return;
}

template <ImageType image_type, DataType data_type, BorderType border_type, int kernel_size>
void MedianFilterAsync(unsigned char *src, unsigned char *dst, int in_h, int in_w, int ksize, cudaStream_t stream) {
    CudaMedianFilter<image_type, data_type, border_type, kernel_size>(src, dst, in_h, in_w, stream);
    return;
}
}  // namespace cudaop
}  // namespace smartmore

#endif
