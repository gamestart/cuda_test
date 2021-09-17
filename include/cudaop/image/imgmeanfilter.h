/*******************************************************************************
 *  FILENAME:      imgmeanfilter.h
 *
 *  AUTHORS:       Chen Fting    START DATE: Friday July 30th 2021
 *
 *  LAST MODIFIED: Thursday, September 16th 2021, 7:48:15 pm
 *
 *  CONTACT:       fting.chen@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_MEANFILTER_H__
#define __SMARTMORE_CUDAOP_IMAGE_MEANFILTER_H__

#include <cuda_runtime_api.h>
#include <cudaop/common/utils.h>
#include <cudaop/image/generic/border.h>
#include <cudaop/types.h>

namespace smartmore {
namespace cudaop {

template <BorderType border_type, int block_x, int block_y, int kernel_h, int kernel_w, int tile_x, int tile_y,
          int x_size, int y_size, int shmem_h, int shmem_w, int data_per_thread>
__global__ void MeanFilterKernelGrayInt8ReflectUseSharedmem(void *in, void *out, int in_h, int in_w) {
    const int r_w = kernel_w >> 1;
    const int r_h = kernel_h >> 1;
    int shmem_start_h = blockIdx.y * y_size - r_h;
    int shmem_start_w = blockIdx.x * x_size - r_w;

    int tid = threadIdx.y * block_x + threadIdx.x;
    extern __shared__ unsigned char shmem[];

#pragma unroll
    for (int i = 0; i < data_per_thread; i++) {
        int idx_shared = tid + i * block_x * block_y;
        int ori_x = idx_shared % shmem_w + shmem_start_w;
        int ori_y = idx_shared / shmem_w + shmem_start_h;
        shmem[idx_shared] = static_cast<unsigned char *>(
            in)[GetBorderIndex<border_type>(ori_y, in_h) * in_w + GetBorderIndex<border_type>(ori_x, in_w)];
    }
    __syncthreads();

    unsigned char *shmem_col = &shmem[shmem_h * shmem_w];

    const int seperate_x = shmem_w - kernel_w + 1;
    const int seperate_y = shmem_h;
    const int seperate_x_per_thread = (seperate_x + block_x - 1) / block_x;
    const int seperate_y_per_thread = (seperate_y + block_y - 1) / block_y;

    int j = threadIdx.y * seperate_y_per_thread;
#pragma unroll
    for (int tj = 0; tj < seperate_y_per_thread; tj++) {
        if (j < seperate_y) {
            int j_index = j * shmem_w;
            int sum_x = 0;

            int i = threadIdx.x * seperate_x_per_thread;
            if (i < seperate_x) {
#pragma unroll
                for (int k = 0; k < r_w; k++) {
                    sum_x += shmem[j_index + k + i] + shmem[j_index + kernel_w - 1 - k + i];
                }
                sum_x += shmem[j_index + r_w + i];
            }
            shmem_col[j * seperate_x + i] = int(float(sum_x) / kernel_w + 0.5f);

#pragma unroll
            for (int ti = 1; ti < seperate_x_per_thread; ti++) {
                i++;
                if (i < seperate_x) {
                    int shx = i + r_w;
                    sum_x += shmem[j_index + shx + r_w] - shmem[j_index + shx - r_w - 1];
                    shmem_col[j * seperate_x + i] = int(float(sum_x) / kernel_w + 0.5f);
                }
            }
        }
        j++;
    }
    __syncthreads();
    // 先列后行
    const int block_y_index = blockIdx.y * block_y + threadIdx.y;
    // int idx = (blockIdx.x * block_x + threadIdx.x) * tile_x;
    int thread_y_index = threadIdx.y * tile_y;
    int tidx = blockIdx.x * x_size;
    int border_x = in_w - tidx;
    int idx = threadIdx.x;
#pragma unroll
    for (int tmpidx = 0; tmpidx < tile_x; tmpidx++) {
        // int idx = threadIdx.x + tmpidx * block_x;
        if (idx < border_x) {
            int idy = block_y_index * tile_y;
            int sum_y = 0;

            if (idy < in_h) {
                for (int j = 0; j < r_h; j++) {
                    sum_y += shmem_col[(j + thread_y_index) * x_size + idx] +
                             shmem_col[(kernel_h - 1 - j + thread_y_index) * x_size + idx];
                }
                sum_y += shmem_col[(r_h + thread_y_index) * x_size + idx];
                static_cast<unsigned char *>(out)[idx + tidx + in_w * idy] = int(float(sum_y) / kernel_h + 0.5f);
            }
            idy++;

#pragma unroll
            for (int tidy = 1; tidy < tile_y; tidy++) {
                if (idy < in_h) {
                    sum_y += shmem_col[(thread_y_index + tidy + kernel_h - 1) * x_size + idx] -
                             shmem_col[(thread_y_index + tidy - 1) * x_size + idx];
                    static_cast<unsigned char *>(out)[idx + tidx + in_w * idy] = int(float(sum_y) / kernel_h + 0.5f);
                }
                idy++;
            }
        }
        idx += block_x;
    }
}

template <ImageType image_type, DataType data_type, BorderType border_type, int kernel_h, int kernel_w>
void CudaMeanFilter(void *in, void *out, int in_h, int in_w, cudaStream_t stream) {
    static_assert(image_type == ImageType::kGRAY && data_type == DataType::kInt8,
                  "MeanFilter only supports Gray int 8");
    const int block_x = 32;
    const int block_y = 16;
    const dim3 block(block_x, block_y);
    const int tile_x = 4;
    const int tile_y = 2;
    const int x_size = block_x * tile_x;
    const int y_size = block_y * tile_y;
    int grid_x = (in_w + x_size - 1) / x_size;
    int grid_y = (in_h + y_size - 1) / y_size;
    dim3 grid(grid_x, grid_y);

    {
        const int shmem_h = y_size + kernel_h - 1;
        const int shmem_w = x_size + kernel_w - 1;

        const int data_per_thread = (shmem_h * shmem_w + block_x * block_y - 1) / (block_x * block_y);

        MeanFilterKernelGrayInt8ReflectUseSharedmem<border_type, block_x, block_y, kernel_h, kernel_w, tile_x, tile_y,
                                                    x_size, y_size, shmem_h, shmem_w, data_per_thread>
            <<<grid, block, shmem_h * shmem_w + block_x * tile_x * shmem_h, stream>>>(in, out, in_h, in_w);
    }
}

template <ImageType image_type, DataType data_type, BorderType border_type, int kernel_h, int kernel_w>
void MeanFilter(void *in, void *out, int in_h, int in_w) {
    CudaMeanFilter<image_type, data_type, border_type, kernel_h, kernel_w>(in, out, in_h, in_w, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}

template <ImageType image_type, DataType data_type, BorderType border_type, int kernel_h, int kernel_w>
void MeanFilterAsync(void *in, void *out, int in_h, int in_w, cudaStream_t stream) {
    CudaMeanFilter<image_type, data_type, border_type, kernel_h, kernel_w>(in, out, in_h, in_w, stream);
}
}  // namespace cudaop
}  // namespace smartmore

#endif