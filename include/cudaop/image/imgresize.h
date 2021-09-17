/*******************************************************************************
 *  FILENAME:      imgresize.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Saturday March 13th 2021
 *
 *  LAST MODIFIED: Thursday, May 20th 2021, 2:40:14 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_IMGRESIZE_H__
#define __SMARTMORE_CUDAOP_IMAGE_IMGRESIZE_H__

#include <cudaop/common/utils.h>
#include <cudaop/image/generic/interpolate.h>
#include <cudaop/image/generic/storedata.h>
#include <cudaop/image/generic/yuvformula.h>
#include <cudaop/type_traits/img_type_traits.h>

namespace smartmore {
namespace cudaop {
template <ImageType src_image_type, YUVFormula yuv_formula>
static inline __device__ auto GenBlack()
    -> enable_if_t<ImageTypeTraits<src_image_type>::yuv_type != YUVType::NOTYUV, float3> {
    return RGBToYUV<yuv_formula>(0.0, 0.0, 0.0);
}

template <ImageType src_image_type, YUVFormula yuv_formula>
static inline __device__ auto GenBlack()
    -> enable_if_t<ImageTypeTraits<src_image_type>::yuv_type == YUVType::NOTYUV &&
                       std::is_same<typename ImageTypeTraits<src_image_type>::PixelType, float3>::value,
                   float3> {
    return make_float3(0.0, 0.0, 0.0);
}

template <ImageType src_image_type, YUVFormula yuv_formula>
static inline __device__ auto GenBlack()
    -> enable_if_t<ImageTypeTraits<src_image_type>::yuv_type == YUVType::NOTYUV &&
                       std::is_same<typename ImageTypeTraits<src_image_type>::PixelType, float>::value,
                   float> {
    return 0.0;
}

template <ImageType src_image_type, DataType input_data_type, DataType output_data_type, ResizeAlgoType algo_type>
static __global__ void ImageResizeStretchKernel(void *src, void *dst, int in_h, int in_w, int in_size, int out_h,
                                                int out_w, int out_size, float fh, float fw) {
    int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x >= out_w || dst_y >= out_h) return;

    typename ImageTypeTraits<src_image_type>::PixelType out =
        Interpolate<src_image_type, input_data_type, algo_type>(src, in_h, in_w, in_size, dst_x, dst_y, fh, fw);

    StoreDataUnifiedTo255<output_data_type, src_image_type>(dst, dst_x, dst_y, out_w, out_h, out_size, out);
}

template <ImageType src_image_type, DataType input_data_type, DataType output_data_type, ResizeAlgoType algo_type>
static __global__ void ImageResizeSelfAdaptKernel(void *src, void *dst, int in_h, int in_w, int in_size, int out_h,
                                                  int out_w, int out_size, int frame_h, int frame_w, float fh, float fw,
                                                  int leftup_y, int leftup_x) {
    int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x >= out_w || dst_y >= out_h) return;

    typename ImageTypeTraits<src_image_type>::PixelType out;

    if (dst_x < leftup_x + frame_w && dst_x >= leftup_x && dst_y < leftup_y + frame_h && dst_y >= leftup_y) {
        out = Interpolate<src_image_type, input_data_type, algo_type>(src, in_h, in_w, in_size, dst_x - leftup_x,
                                                                      dst_y - leftup_y, fh, fw);
    } else {
        out = GenBlack<src_image_type, YUVFormula::kBT601>();
    }

    StoreDataUnifiedTo255<output_data_type, src_image_type>(dst, dst_x, dst_y, out_w, out_h, out_size, out);
}

template <ImageType src_image_type, DataType input_data_type, DataType output_data_type, ResizeScaleType scale_type,
          ResizeAlgoType algo_type>
auto CudaImageResize(void *src, void *dst, int in_h, int in_w, int out_h, int out_w, cudaStream_t stream)
    -> enable_if_t<scale_type == ResizeScaleType::kSelfAdapt, void> {
    dim3 block(32, 32);
    int grid_x = (out_w + block.x - 1) / block.x;
    int grid_y = (out_h + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);

    int frame_h, frame_w, leftup_y, leftup_x;

    if (out_h * in_w > out_w * in_h) {
        frame_w = out_w;
        frame_h = frame_w * in_h / in_w;
        leftup_x = 0;
        leftup_y = (out_h - frame_h) / 2;
    } else {
        frame_h = out_h;
        frame_w = frame_h * in_w / in_h;
        leftup_y = 0;
        leftup_x = (out_w - frame_w) / 2;
    }

    ImageResizeSelfAdaptKernel<src_image_type, input_data_type, output_data_type, algo_type>
        <<<grid, block, 0, stream>>>(src, dst, in_h, in_w, in_h * in_w, out_h, out_w, out_h * out_w, frame_h, frame_w,
                                     static_cast<float>(in_h) / frame_h, static_cast<float>(in_w) / frame_w, leftup_y,
                                     leftup_x);
    return;
}

template <ImageType src_image_type, DataType input_data_type, DataType output_data_type, ResizeScaleType scale_type,
          ResizeAlgoType algo_type>
auto CudaImageResize(void *src, void *dst, int in_h, int in_w, int out_h, int out_w, cudaStream_t stream)
    -> enable_if_t<scale_type == ResizeScaleType::kStretch, void> {
    dim3 block(32, 16);
    int grid_x = (out_w + block.x - 1) / block.x;
    int grid_y = (out_h + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);

    ImageResizeStretchKernel<src_image_type, input_data_type, output_data_type, algo_type>
        <<<grid, block, 0, stream>>>(src, dst, in_h, in_w, in_h * in_w, out_h, out_w, out_h * out_w,
                                     static_cast<float>(in_h) / out_h, static_cast<float>(in_w) / out_w);
    return;
}

template <ImageType src_image_type, DataType input_data_type, DataType output_data_type, ResizeScaleType scale_type,
          ResizeAlgoType algo_type>
void ImageResize(void *src, void *dst, int in_h, int in_w, int out_h, int out_w) {
    CudaImageResize<src_image_type, input_data_type, output_data_type, scale_type, algo_type>(src, dst, in_h, in_w,
                                                                                              out_h, out_w, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
    return;
}

template <ImageType src_image_type, DataType input_data_type, DataType output_data_type, ResizeScaleType scale_type,
          ResizeAlgoType algo_type>
void ImageResizeAsync(void *src, void *dst, int in_h, int in_w, int out_h, int out_w, cudaStream_t stream) {
    CudaImageResize<src_image_type, input_data_type, output_data_type, scale_type, algo_type>(src, dst, in_h, in_w,
                                                                                              out_h, out_w, stream);
    return;
}
}  // namespace cudaop
}  // namespace smartmore

#endif