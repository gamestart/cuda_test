/*******************************************************************************
 *  FILENAME:      loaddata.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Wednesday March 10th 2021
 *
 *  LAST MODIFIED: Tuesday, July 13th 2021, 2:12:50 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_GENERIC_LOADDATA_H__
#define __SMARTMORE_CUDAOP_IMAGE_GENERIC_LOADDATA_H__

#include <cudaop/common/utils.h>
#include <cudaop/generic/float3operators.h>
#include <cudaop/type_traits/data_type_traits.h>
#include <cudaop/type_traits/img_type_traits.h>
#include <cudaop/types.h>

namespace smartmore {
namespace cudaop {
template <typename DataType, ImageType image_type>
static inline __device__ auto LoadData(void *src, int id_x, int id_y, int in_w, int in_h, int in_size)
    -> enable_if_t<image_type == ImageType::kBGRA_CHW, float3> {
    int id = id_y * in_w + id_x;
    return make_float3(((DataType *)src)[2 * in_size + id], ((DataType *)src)[in_size + id], ((DataType *)src)[id]);
}

template <typename DataType, ImageType image_type>
static inline __device__ auto LoadData(void *src, int id_x, int id_y, int in_w, int in_h, int in_size)
    -> enable_if_t<image_type == ImageType::kBGR_CHW, float3> {
    int id = id_y * in_w + id_x;
    return make_float3(((DataType *)src)[2 * in_size + id], ((DataType *)src)[in_size + id], ((DataType *)src)[id]);
}

template <typename DataType, ImageType image_type>
static inline __device__ auto LoadData(void *src, int id_x, int id_y, int in_w, int in_h, int in_size)
    -> enable_if_t<image_type == ImageType::kBGRA_HWC, float3> {
    int id = id_y * in_w * 4 + id_x * 4;
    return make_float3(((DataType *)src)[id + 2], ((DataType *)src)[id + 1], ((DataType *)src)[id]);
}

template <typename DataType, ImageType image_type>
static inline __device__ auto LoadData(void *src, int id_x, int id_y, int in_w, int in_h, int in_size)
    -> enable_if_t<image_type == ImageType::kBGR_HWC, float3> {
    int id = id_y * in_w * 3 + id_x * 3;
    return make_float3(((DataType *)src)[id + 2], ((DataType *)src)[id + 1], ((DataType *)src)[id]);
}

template <typename DataType, ImageType image_type>
static inline __device__ auto LoadData(void *src, int id_x, int id_y, int in_w, int in_h, int in_size)
    -> enable_if_t<image_type == ImageType::kRGBA_CHW, float3> {
    int id = id_y * in_w + id_x;
    return make_float3(((DataType *)src)[id], ((DataType *)src)[in_size + id], ((DataType *)src)[2 * in_size + id]);
}

template <typename DataType, ImageType image_type>
static inline __device__ auto LoadData(void *src, int id_x, int id_y, int in_w, int in_h, int in_size)
    -> enable_if_t<image_type == ImageType::kRGB_CHW, float3> {
    int id = id_y * in_w + id_x;
    return make_float3(((DataType *)src)[id], ((DataType *)src)[in_size + id], ((DataType *)src)[2 * in_size + id]);
}

template <typename DataType, ImageType image_type>
static inline __device__ auto LoadData(void *src, int id_x, int id_y, int in_w, int in_h, int in_size)
    -> enable_if_t<image_type == ImageType::kRGBA_HWC, float3> {
    int id = id_y * in_w * 4 + id_x * 4;
    return make_float3(((DataType *)src)[id], ((DataType *)src)[id + 1], ((DataType *)src)[id + 2]);
}

template <typename DataType, ImageType image_type>
static inline __device__ auto LoadData(void *src, int id_x, int id_y, int in_w, int in_h, int in_size)
    -> enable_if_t<image_type == ImageType::kRGB_HWC, float3> {
    int id = id_y * in_w * 3 + id_x * 3;
    return make_float3(((DataType *)src)[id], ((DataType *)src)[id + 1], ((DataType *)src)[id + 2]);
}

template <typename DataType, ImageType image_type>
static inline __device__ auto LoadData(void *src, int id_x, int id_y, int in_w, int in_h, int in_size)
    -> enable_if_t<image_type == ImageType::kYUV_NV12, float3> {
    int id_uv = id_y / 2 * in_w + id_x / 2 * 2;
    return make_float3(((DataType *)src)[id_y * in_w + id_x], ((DataType *)src)[in_size + id_uv],
                       ((DataType *)src)[in_size + id_uv + 1]);
}

template <typename DataType, ImageType image_type>
static inline __device__ auto LoadData(void *src, int id_x, int id_y, int in_w, int in_h, int in_size)
    -> enable_if_t<image_type == ImageType::kYUV_I420, float3> {
    int id_uv = id_y / 2 * in_w / 2 + id_x / 2;
    return make_float3(((DataType *)src)[id_y * in_w + id_x], ((DataType *)src)[in_size + id_uv],
                       ((DataType *)src)[in_size + id_uv + (in_w / 2) * (in_h / 2)]);
}

template <typename DataType, ImageType image_type>
static inline __device__ auto LoadData(void *src, int id_x, int id_y, int in_w, int in_h, int in_size)
    -> enable_if_t<image_type == ImageType::kGRAY, float> {
    int id = id_y * in_w + id_x;
    return float(((DataType *)src)[id]);
}

template <typename DataType, ImageType image_type>
static inline __device__ auto LoadData(void *src, int id_x, int id_y, int in_w, int in_h, int in_size)
    -> enable_if_t<image_type == ImageType::kYUV_UYVY, float3> {
    int id = id_y * in_w + id_x;
    if (id % 2 == 1) {
        return make_float3(((DataType *)src)[id * 2 + 1], ((DataType *)src)[id * 2 - 2], ((DataType *)src)[id * 2]);
    } else {
        return make_float3(((DataType *)src)[id * 2 + 1], ((DataType *)src)[id * 2], ((DataType *)src)[id * 2 + 2]);
    }
}

template <typename DataType, ImageType image_type>
static inline __device__ auto LoadData(void *src, int id_x, int id_y, int in_w, int in_h, int in_size)
    -> enable_if_t<image_type == ImageType::kYuv422p, float3> {
    int id = id_y * in_w + id_x;
    return make_float3(((DataType *)src)[id], ((DataType *)src)[id / 2 + in_size],
                       ((DataType *)src)[id / 2 + in_size * 3 / 2]);
}

template <DataType data_type, ImageType image_type>
static inline __device__ auto LoadDataAndUnifyTo255(void *src, int idx, int id_y, int in_w, int in_h, int in_size)
    -> enable_if_t<data_type == DataType::kInt8, typename ImageTypeTraits<image_type>::PixelType> {
    return LoadData<typename DataTypeTraits<data_type>::Type, image_type>(src, idx, id_y, in_w, in_h, in_size);
}

template <DataType data_type, ImageType image_type>
static inline __device__ auto LoadDataAndUnifyTo255(void *src, int idx, int id_y, int in_w, int in_h, int in_size)
    -> enable_if_t<data_type == DataType::kFloat32 || data_type == DataType::kHalf,
                   typename ImageTypeTraits<image_type>::PixelType> {
    auto tmpdata = LoadData<typename DataTypeTraits<data_type>::Type, image_type>(src, idx, id_y, in_w, in_h, in_size);
    return tmpdata * 255.0;
}
}  // namespace cudaop
}  // namespace smartmore

#endif