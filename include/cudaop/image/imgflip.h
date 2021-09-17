/*******************************************************************************
 *  FILENAME:      flip.h
 *
 *  AUTHORS:       Hou Yue    START DATE: Thursday July 29th 2021
 *
 *  LAST MODIFIED: Tuesday, August 3rd 2021, 8:12:06 pm
 *
 *  CONTACT:       yue.hou@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_FLIP_H__
#define __SMARTMORE_CUDAOP_IMAGE_FLIP_H__

#include <cudaop/common/utils.h>
#include <cudaop/image/generic/loaddata.h>
#include <cudaop/image/generic/storedata.h>
#include <cudaop/types.h>

namespace smartmore {
namespace cudaop {
template <FlipType flip_type>
inline __device__ void ReflectIndex(int *index_x, int *index_y, const int idx, const int idy, int width, int height) {
    *index_x =
        (flip_type >= FlipType::kVert) ? (flip_type >= FlipType::kHor ? (width - idx - 1) : (idx)) : (width - idx - 1);
    *index_y = (flip_type >= FlipType::kVert) ? (flip_type >= FlipType::kHor ? (idy) : (height - idy - 1))
                                              : (height - idy - 1);
}

template <DataType data_type, ImageType image_type, FlipType flip_type>
__global__ void CudaFlipKernal(void *src, void *dst, const int width, const int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx >= width || idy >= height) return;

    auto in_data =
        LoadData<typename DataTypeTraits<data_type>::Type, image_type>(src, idx, idy, width, height, width * height);
    int index_x = idx, index_y = idy;
    ReflectIndex<flip_type>(&index_x, &index_y, idx, idy, width, height);
    StoreData<typename DataTypeTraits<data_type>::Type, image_type>(dst, index_x, index_y, width, height,
                                                                    width * height, in_data);
}

template <DataType data_type, ImageType image_type, FlipType flip_type>
void CudaImageFlip(void *src, void *dst, const Size src_size, cudaStream_t stream) {
    CUDAOP_ASSERT_TRUE(src != nullptr && dst != nullptr);
    constexpr dim3 block(32, 32);
    int grid_x = (src_size.width + block.x - 1) / block.x;
    int grid_y = (src_size.height + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);
    CudaFlipKernal<data_type, image_type, flip_type>
        <<<grid, block, 0, stream>>>(src, dst, src_size.width, src_size.height);
}

template <DataType data_type, ImageType image_type, FlipType flip_type>
void ImageFlip(void *src, void *dst, const Size src_size) {
    CudaImageFlip<data_type, image_type, flip_type>(src, dst, src_size, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
    return;
}
template <DataType data_type, ImageType image_type, FlipType flip_type>
void ImageFlipAsync(void *src, void *dst, const Size src_size, cudaStream_t stream) {
    CudaImageFlip<data_type, image_type, flip_type>(src, dst, src_size, stream);
    return;
}
}  // namespace cudaop
}  // namespace smartmore

#endif