/*******************************************************************************
 *  FILENAME:      imgcopy.h
 *
 *  AUTHORS:       Liang Jia    START DATE: Thursday May 13th 2021
 *
 *  LAST MODIFIED: Thursday, July 22nd 2021, 10:19:17 am
 *
 *  CONTACT:       jia.liang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_COPY_H__
#define __SMARTMORE_CUDAOP_IMAGE_COPY_H__
#include <cudaop/common/utils.h>
#include <cudaop/image/generic/loaddata.h>
#include <cudaop/image/generic/storedata.h>

namespace smartmore {
namespace cudaop {

template <ImageType image_type, DataType input_data_type>
__global__ void CudaImageCopyKernel(void *src, int src_w, int src_h, int src_start_x, int src_start_y, void *dst,
                                    int dst_w, int dst_h, int dst_start_x, int dst_start_y, int crop_w, int crop_h,
                                    int in_size) {
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (id_x >= crop_w || id_y >= crop_h) return;
    auto in_data = LoadData<typename DataTypeTraits<input_data_type>::Type, image_type>(
        src, id_x + src_start_x, id_y + src_start_y, src_w, src_h, in_size);
    StoreData<typename DataTypeTraits<input_data_type>::Type, image_type>(dst, id_x + dst_start_x, id_y + dst_start_y,
                                                                          dst_w, dst_h, dst_w * dst_h, in_data);
}

template <ImageType image_type, DataType input_data_type>
void CudaImageCopy(void *src, const Size &src_size, const Rect &copy_rect, void *dst, const Size &dst_size,
                   const Point &dst_rect_tl, cudaStream_t stream) {
    CUDAOP_ASSERT_TRUE(src != nullptr && dst != nullptr);
    CUDAOP_ASSERT_TRUE(src_size.height >= copy_rect.size.height + copy_rect.topleft.y);
    CUDAOP_ASSERT_TRUE(src_size.width >= copy_rect.size.width + copy_rect.topleft.x);

    CUDAOP_ASSERT_TRUE(dst_size.height >= copy_rect.size.height + dst_rect_tl.y);
    CUDAOP_ASSERT_TRUE(dst_size.width >= copy_rect.size.width + dst_rect_tl.x);
    dim3 block(32, 32);
    int grid_x = (copy_rect.size.width + block.x - 1) / block.x;
    int grid_y = (copy_rect.size.height + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);

    auto roi_w = copy_rect.size.width;
    auto roi_h = copy_rect.size.height;
    CudaImageCopyKernel<image_type, input_data_type><<<grid, block, 0, stream>>>(
        src, src_size.width, src_size.height, copy_rect.topleft.x, copy_rect.topleft.y, dst, dst_size.width,
        dst_size.height, dst_rect_tl.x, dst_rect_tl.y, roi_w, roi_h, roi_w * roi_h);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}

template <ImageType image_type, DataType input_data_type>
void ImageCopy(void *src, const Size &src_size, const Rect &copy_rect, void *dst, const Size &dst_size,
               const Point &dst_rect_tl) {
    CudaImageCopy<image_type, input_data_type>(src, src_size, copy_rect, dst, dst_size, dst_rect_tl, 0);
    cudaDeviceSynchronize();
    return;
}

template <ImageType image_type, DataType input_data_type>
void ImageCopyAsync(void *src, const Size &src_size, const Rect &copy_rect, void *dst, const Size &dst_size,
                    const Point &dst_rect_tl, cudaStream_t stream) {
    CudaImageCopy<image_type, input_data_type>(src, src_size, copy_rect, dst, dst_size, dst_rect_tl, stream);
    return;
}
}  // namespace cudaop
}  // namespace smartmore

#endif