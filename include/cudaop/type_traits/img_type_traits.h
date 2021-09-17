/*******************************************************************************
 *  FILENAME:      img_type_traits.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Friday March 12th 2021
 *
 *  LAST MODIFIED: Thursday, May 20th 2021, 7:37:30 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_TYPE_TRAITS_IMG_TYPE_TRAITS_H__
#define __SMARTMORE_CUDAOP_TYPE_TRAITS_IMG_TYPE_TRAITS_H__

#include <cudaop/types.h>

namespace smartmore {
namespace cudaop {
enum class YUVType {
    NOTYUV,
    YUV422,
    YUV420,
    YUV444,
};

template <ImageType image_type>
struct ImageTypeTraits {
    typedef float3 PixelType;
    static const YUVType yuv_type = YUVType::NOTYUV;
};

template <>
struct ImageTypeTraits<ImageType::kGRAY> {
    typedef float PixelType;
    static const YUVType yuv_type = YUVType::NOTYUV;
};

template <>
struct ImageTypeTraits<ImageType::kYUV_NV12> {
    typedef float3 PixelType;
    static const YUVType yuv_type = YUVType::YUV420;
};

template <>
struct ImageTypeTraits<ImageType::kYUV_UYVY> {
    typedef float3 PixelType;
    static const YUVType yuv_type = YUVType::YUV422;
};

template <>
struct ImageTypeTraits<ImageType::kYuv422p> {
    typedef float3 PixelType;
    static const YUVType yuv_type = YUVType::YUV422;
};

template <>
struct ImageTypeTraits<ImageType::kYUV_I420> {
    typedef float3 PixelType;
    static const YUVType yuv_type = YUVType::YUV420;
};
}  // namespace cudaop
}  // namespace smartmore

#endif