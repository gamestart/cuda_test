/*******************************************************************************
 *  FILENAME:      imghorizonscan.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Friday March 12th 2021
 *
 *  LAST MODIFIED: Wednesday, May 19th 2021, 7:22:09 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_IMGHORIZON_SCAN_H__
#define __SMARTMORE_CUDAOP_IMAGE_IMGHORIZON_SCAN_H__

#include <cudaop/image/generic/loaddata.h>
#include <cudaop/image/generic/storedata.h>
#include <cudaop/image/generic/yuvformula.h>
#include <cudaop/type_traits/img_type_traits.h>

namespace smartmore {
namespace cudaop {
template <ImageType src_image_type, YUVFormula yuv_formula>
static inline __device__ auto GenWhite()
    -> enable_if_t<ImageTypeTraits<src_image_type>::yuv_type != YUVType::NOTYUV, float3> {
    return RGBToYUV<yuv_formula>(255.0, 255.0, 255.0);
}

template <ImageType src_image_type, YUVFormula yuv_formula>
static inline __device__ auto GenWhite()
    -> enable_if_t<ImageTypeTraits<src_image_type>::yuv_type == YUVType::NOTYUV &&
                       std::is_same<typename ImageTypeTraits<src_image_type>::PixelType, float3>::value,
                   float3> {
    return make_float3(255.0, 255.0, 255.0);
}

template <ImageType src_image_type, YUVFormula yuv_formula>
static inline __device__ auto GenWhite()
    -> enable_if_t<ImageTypeTraits<src_image_type>::yuv_type == YUVType::NOTYUV &&
                       std::is_same<typename ImageTypeTraits<src_image_type>::PixelType, float>::value,
                   float> {
    return 255.0;
}

template <ImageType input_image_type, DataType input_data_type, ImageType output_image_type, DataType output_data_type,
          YUVFormula yuv_formula>
__global__ void CudaImageHorizonScanSpecialEffectKernel(void *src_l, void *src_r, void *dst, int in_h, int in_w,
                                                        int in_size, int scan_x) {
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (id_x >= in_w || id_y >= in_h) return;

    float3 src_px;

    if (id_x < scan_x) {
        src_px = LoadDataAndUnifyTo255<input_data_type, input_image_type>(src_l, id_x, id_y, in_w, in_h, in_size);
    } else if (id_x > scan_x) {
        src_px = LoadDataAndUnifyTo255<input_data_type, input_image_type>(src_r, id_x, id_y, in_w, in_h, in_size);
    } else {
        src_px = GenWhite<input_image_type, yuv_formula>();
    }

    float3 dst_px = ConvertPixel<input_image_type, output_image_type, yuv_formula>(src_px.x, src_px.y, src_px.z);

    StoreDataUnifiedTo255<output_data_type, output_image_type>(dst, id_x, id_y, in_w, in_h, in_size, dst_px);
}

template <ImageType input_image_type, DataType input_data_type, ImageType output_image_type, DataType output_data_type,
          YUVFormula yuv_formula>
void CudaImageHorizonScanSpecialEffect(void *src_l, void *src_r, void *dst, int in_h, int in_w, int scan_x,
                                       cudaStream_t stream) {
    dim3 block(32, 32);
    int grid_x = (in_w + block.x - 1) / block.x;
    int grid_y = (in_h + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);
    CudaImageHorizonScanSpecialEffectKernel<input_image_type, input_data_type, output_image_type, output_data_type,
                                            yuv_formula>
        <<<grid, block, 0, stream>>>(src_l, src_r, dst, in_h, in_w, in_h * in_w, scan_x);
}

template <ImageType input_image_type, DataType input_data_type, ImageType output_image_type, DataType output_data_type,
          YUVFormula yuv_formula = YUVFormula::kBT601>
void ImageHorizonScanSpecialEffect(void *src_l, void *src_r, void *dst, int in_h, int in_w, int scan_x) {
    CudaImageHorizonScanSpecialEffect<input_image_type, input_data_type, output_image_type, output_data_type,
                                      yuv_formula>(src_l, src_r, dst, in_h, in_w, scan_x, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}

template <ImageType input_image_type, DataType input_data_type, ImageType output_image_type, DataType output_data_type,
          YUVFormula yuv_formula = YUVFormula::kBT601>
void ImageHorizonScanSpecialEffectAsync(void *src_l, void *src_r, void *dst, int in_h, int in_w, int scan_x,
                                        cudaStream_t stream) {
    CudaImageHorizonScanSpecialEffect<input_image_type, input_data_type, output_image_type, output_data_type,
                                      yuv_formula>(src_l, src_r, dst, in_h, in_w, scan_x, stream);
}
}  // namespace cudaop
}  // namespace smartmore
#endif