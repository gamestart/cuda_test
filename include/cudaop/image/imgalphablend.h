/*******************************************************************************
 *  FILENAME:      imgalphablend.h
 *
 *  AUTHORS:       Liang Jia    START DATE: Saturday August 14th 2021
 *
 *  LAST MODIFIED: Monday, August 16th 2021, 11:14:06 am
 *
 *  CONTACT:       jia.liang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_IMGALPHABLEND_H__
#define __SMARTMORE_CUDAOP_IMAGE_IMGALPHABLEND_H__
#include <cudaop/common/utils.h>
#include <cudaop/image/generic/loaddata.h>
#include <cudaop/image/generic/storedata.h>

namespace smartmore {
namespace cudaop {

template <ImageType image_type, DataType input_data_type>
__global__ void CudaImageAlphaBlendKernel(void *foreground, void *background, float *alpha_mask, int bg_w, int bg_h,
                                          int top_x, int top_y, int fg_w, int fg_h, int fg_size, int bg_size) {
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (id_x >= fg_w || id_y >= fg_h) return;
    auto foreground_data = LoadData<typename DataTypeTraits<input_data_type>::Type, image_type>(foreground, id_x, id_y,
                                                                                                fg_w, fg_h, fg_size);
    auto background_data = LoadData<typename DataTypeTraits<input_data_type>::Type, image_type>(
        background, id_x + top_x, id_y + top_y, bg_w, bg_h, bg_size);

    float alpha = alpha_mask[id_y * fg_w + id_x];
    auto blend_data = foreground_data * alpha + background_data * (1.0f - alpha);
    StoreData<typename DataTypeTraits<input_data_type>::Type, image_type>(background, id_x + top_x, id_y + top_y, bg_w,
                                                                          bg_h, bg_size, blend_data);
}

template <ImageType image_type, DataType input_data_type>
void CudaImageAlphaBlend(void *foreground, void *background, float *alpha_mask, const Size &background_size,
                         const Rect &blend_roi, cudaStream_t stream) {
    CUDAOP_ASSERT_TRUE(input_data_type == DataType::kFloat32);
    CUDAOP_ASSERT_TRUE(foreground != nullptr && background != nullptr && alpha_mask != nullptr);
    CUDAOP_ASSERT_TRUE(blend_roi.topleft.x >= 0 && blend_roi.topleft.x + blend_roi.size.width <= background_size.width);
    CUDAOP_ASSERT_TRUE(blend_roi.topleft.y >= 0 &&
                       blend_roi.topleft.y + blend_roi.size.height <= background_size.height);

    dim3 block(32, 32);
    auto foregroud_size = blend_roi.size;
    auto roi_w = foregroud_size.width;
    auto roi_h = foregroud_size.height;
    auto bg_w = background_size.width;
    auto bg_h = background_size.height;

    int grid_x = (roi_w + block.x - 1) / block.x;
    int grid_y = (roi_h + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);

    CudaImageAlphaBlendKernel<image_type, input_data_type>
        <<<grid, block, 0, stream>>>(foreground, background, alpha_mask, bg_w, bg_h, blend_roi.topleft.x,
                                     blend_roi.topleft.y, roi_w, roi_h, roi_w * roi_h, bg_w * bg_h);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}

template <ImageType image_type, DataType input_data_type>
void ImageAlphaBlend(void *foreground, void *background, float *alpha_mask, const Size &background_size,
                     const Rect &blend_roi) {
    CudaImageAlphaBlend<image_type, input_data_type>(foreground, background, alpha_mask, background_size, blend_roi, 0);
    cudaDeviceSynchronize();
    return;
}

template <ImageType image_type, DataType input_data_type>
void ImageAlphaBlendAsync(void *foreground, void *background, float *alpha_mask, const Size &background_size,
                          const Rect &blend_roi, cudaStream_t stream) {
    CudaImageAlphaBlend<image_type, input_data_type>(foreground, background, alpha_mask, background_size, blend_roi,
                                                     stream);
    return;
}
}  // namespace cudaop
}  // namespace smartmore

#endif /* __SMARTMORE_CUDAOP_IMAGE_IMGALPHABLEND_H__ */
