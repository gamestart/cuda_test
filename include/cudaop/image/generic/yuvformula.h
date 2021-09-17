/*******************************************************************************
 *  FILENAME:      yuvformula.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Wednesday March 10th 2021
 *
 *  LAST MODIFIED: Thursday, August 26th 2021, 7:56:28 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_GENERIC_YUVFORMULA_H__
#define __SMARTMORE_CUDAOP_IMAGE_GENERIC_YUVFORMULA_H__

#include <cudaop/common/utils.h>
#include <cudaop/types.h>

namespace smartmore {
namespace cudaop {
static inline __device__ float Clamp255(float x) { return fminf(fmaxf(x, 0.0f), 255.0f); }

template <YUVFormula yuv_formula>
static inline __device__ auto YUVToRGB(float y, float u, float v)
    -> enable_if_t<yuv_formula == YUVFormula::kBT601, float3> {
    u -= 128.0f;
    v -= 128.0f;
    y -= 16.0f;

    return make_float3(Clamp255(1.164f * y + 1.596f * v), Clamp255(1.164f * y - 0.813f * v - 0.391f * u),
                       Clamp255(1.164f * y + 2.018f * u));
}

template <YUVFormula yuv_formula>
static inline __device__ auto RGBToYUV(float r, float g, float b)
    -> enable_if_t<yuv_formula == YUVFormula::kBT601, float3> {
    float Y = 0.257f * r + 0.504f * g + 0.098f * b + 16.0f;
    float U = -0.148f * r - 0.291f * g + 0.439f * b + 128.0f;
    float V = 0.439f * r - 0.368f * g - 0.071f * b + 128.0f;

    return make_float3(Clamp255(Y), Clamp255(U), Clamp255(V));
}

template <YUVFormula yuv_formula>
static inline __device__ auto YUVToRGB(float y, float u, float v)
    -> enable_if_t<yuv_formula == YUVFormula::kBT709, float3> {
    u -= 128.0f;
    v -= 128.0f;
    y -= 16.0f;

    return make_float3(Clamp255(1.164f * y + 1.793f * v), Clamp255(1.164f * y - 0.533f * v - 0.213f * u),
                       Clamp255(1.164f * y + 2.112f * u));
}

template <YUVFormula yuv_formula>
static inline __device__ auto RGBToYUV(float r, float g, float b)
    -> enable_if_t<yuv_formula == YUVFormula::kBT709, float3> {
    float Y = 0.183f * r + 0.614f * g + 0.062f * b + 16.0f;
    float U = -0.101f * r - 0.339f * g + 0.439f * b + 128.0f;
    float V = 0.439f * r - 0.399f * g - 0.04f * b + 128.0f;

    return make_float3(Clamp255(Y), Clamp255(U), Clamp255(V));
}

template <YUVFormula yuv_formula>
static inline __device__ auto YUVToRGB(float y, float u, float v)
    -> enable_if_t<yuv_formula == YUVFormula::kYCrCb, float3> {
    u -= 128.0f;
    v -= 128.0f;

    return make_float3(Clamp255(y + 1.403f * v), Clamp255(y - 0.714f * v - 0.344f * u), Clamp255(y + 1.733f * u));
}

template <YUVFormula yuv_formula>
static inline __device__ auto RGBToYUV(float r, float g, float b)
    -> enable_if_t<yuv_formula == YUVFormula::kYCrCb, float3> {
    float Y = 0.299f * r + 0.587f * g + 0.114f * b;
    float V = (r - Y) * 0.713f + 128.0f;
    float U = (b - Y) * 0.564f + 128.0f;

    return make_float3(Clamp255(Y), Clamp255(U), Clamp255(V));
}

template <ImageType input_image_type, ImageType output_image_type, YUVFormula yuv_formula>
static inline __device__ auto ConvertPixel(float x, float y, float z)
    -> enable_if_t<ImageTypeTraits<input_image_type>::yuv_type != YUVType::NOTYUV &&
                       ImageTypeTraits<output_image_type>::yuv_type == YUVType::NOTYUV,
                   float3> {
    return YUVToRGB<yuv_formula>(x, y, z);
}

template <ImageType input_image_type, ImageType output_image_type, YUVFormula yuv_formula>
static inline __device__ auto ConvertPixel(float x, float y, float z)
    -> enable_if_t<ImageTypeTraits<input_image_type>::yuv_type == YUVType::NOTYUV &&
                       ImageTypeTraits<output_image_type>::yuv_type != YUVType::NOTYUV,
                   float3> {
    return RGBToYUV<yuv_formula>(x, y, z);
}

template <ImageType input_image_type, ImageType output_image_type, YUVFormula yuv_formula>
static inline __device__ auto ConvertPixel(float x, float y, float z)
    -> enable_if_t<ImageTypeTraits<input_image_type>::yuv_type == YUVType::NOTYUV &&
                       ImageTypeTraits<output_image_type>::yuv_type == YUVType::NOTYUV,
                   float3> {
    return make_float3(x, y, z);
}

template <ImageType input_image_type, ImageType output_image_type, YUVFormula yuv_formula>
static inline __device__ auto ConvertPixel(float x, float y, float z)
    -> enable_if_t<ImageTypeTraits<input_image_type>::yuv_type != YUVType::NOTYUV &&
                       ImageTypeTraits<output_image_type>::yuv_type != YUVType::NOTYUV,
                   float3> {
    return make_float3(x, y, z);
}

}  // namespace cudaop
}  // namespace smartmore

#endif