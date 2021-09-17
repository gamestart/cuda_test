/*******************************************************************************
 *  FILENAME:      storedata.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Wednesday March 10th 2021
 *
 *  LAST MODIFIED: Tuesday, July 13th 2021, 2:12:50 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_GENERIC_STOREDATA_H__
#define __SMARTMORE_CUDAOP_IMAGE_GENERIC_STOREDATA_H__

#include <cudaop/common/utils.h>
#include <cudaop/generic/float3operators.h>
#include <cudaop/type_traits/data_type_traits.h>
#include <cudaop/type_traits/img_type_traits.h>
#include <cudaop/types.h>

namespace smartmore {
namespace cudaop {
template <typename DataType, ImageType image_type>
static inline __device__ auto StoreData(void *dst, int id_x, int id_y, int in_w, int in_h, int in_size, float3 data)
    -> enable_if_t<image_type == ImageType::kBGRA_CHW, void> {
    int id = id_y * in_w + id_x;
    ((DataType *)dst)[id] = data.z;
    ((DataType *)dst)[in_size + id] = data.y;
    ((DataType *)dst)[2 * in_size + id] = data.x;
    ((DataType *)dst)[3 * in_size + id] = 255.0;
    return;
}

template <typename DataType, ImageType image_type>
static inline __device__ auto StoreData(void *dst, int id_x, int id_y, int in_w, int in_h, int in_size, float3 data)
    -> enable_if_t<image_type == ImageType::kBGR_CHW, void> {
    int id = id_y * in_w + id_x;
    ((DataType *)dst)[id] = data.z;
    ((DataType *)dst)[in_size + id] = data.y;
    ((DataType *)dst)[2 * in_size + id] = data.x;
    return;
}

template <typename DataType, ImageType image_type>
static inline __device__ auto StoreData(void *dst, int id_x, int id_y, int in_w, int in_h, int in_size, float3 data)
    -> enable_if_t<image_type == ImageType::kBGRA_HWC, void> {
    int id = id_y * in_w * 4 + id_x * 4;
    ((DataType *)dst)[id] = data.z;
    ((DataType *)dst)[id + 1] = data.y;
    ((DataType *)dst)[id + 2] = data.x;
    ((DataType *)dst)[id + 3] = 255.0;
    return;
}

template <typename DataType, ImageType image_type>
static inline __device__ auto StoreData(void *dst, int id_x, int id_y, int in_w, int in_h, int in_size, float3 data)
    -> enable_if_t<image_type == ImageType::kBGR_HWC, void> {
    int id = id_y * in_w * 3 + id_x * 3;
    ((DataType *)dst)[id] = data.z;
    ((DataType *)dst)[id + 1] = data.y;
    ((DataType *)dst)[id + 2] = data.x;
    return;
}

template <typename DataType, ImageType image_type>
static inline __device__ auto StoreData(void *dst, int id_x, int id_y, int in_w, int in_h, int in_size, float3 data)
    -> enable_if_t<image_type == ImageType::kRGBA_CHW, void> {
    int id = id_y * in_w + id_x;
    ((DataType *)dst)[id] = data.x;
    ((DataType *)dst)[in_size + id] = data.y;
    ((DataType *)dst)[2 * in_size + id] = data.z;
    ((DataType *)dst)[3 * in_size + id] = 255.0;
}

template <typename DataType, ImageType image_type>
static inline __device__ auto StoreData(void *dst, int id_x, int id_y, int in_w, int in_h, int in_size, float3 data)
    -> enable_if_t<image_type == ImageType::kRGB_CHW, void> {
    int id = id_y * in_w + id_x;
    ((DataType *)dst)[id] = data.x;
    ((DataType *)dst)[in_size + id] = data.y;
    ((DataType *)dst)[2 * in_size + id] = data.z;
    return;
}

template <typename DataType, ImageType image_type>
static inline __device__ auto StoreData(void *dst, int id_x, int id_y, int in_w, int in_h, int in_size, float3 data)
    -> enable_if_t<image_type == ImageType::kRGBA_HWC, void> {
    int id = id_y * in_w * 4 + id_x * 4;
    ((DataType *)dst)[id] = data.x;
    ((DataType *)dst)[id + 1] = data.y;
    ((DataType *)dst)[id + 2] = data.z;
    ((DataType *)dst)[id + 3] = 255.0;
    return;
}

template <typename DataType, ImageType image_type>
static inline __device__ auto StoreData(void *dst, int id_x, int id_y, int in_w, int in_h, int in_size, float3 data)
    -> enable_if_t<image_type == ImageType::kRGB_HWC, void> {
    int id = id_y * in_w * 3 + id_x * 3;
    ((DataType *)dst)[id] = data.x;
    ((DataType *)dst)[id + 1] = data.y;
    ((DataType *)dst)[id + 2] = data.z;
    return;
}

template <typename DataType, ImageType image_type>
static inline __device__ auto StoreData(void *dst, int id_x, int id_y, int in_w, int in_h, int in_size, float3 data)
    -> enable_if_t<image_type == ImageType::kYUV_NV12, void> {
    int id_uv = id_y / 2 * in_w + id_x / 2 * 2;
    ((DataType *)dst)[id_y * in_w + id_x] = static_cast<DataType>(data.x);
    ((DataType *)dst)[in_size + id_uv] = static_cast<DataType>(data.y);
    ((DataType *)dst)[in_size + id_uv + 1] = static_cast<DataType>(data.z);
    return;
}

template <typename DataType, ImageType image_type>
static inline __device__ auto StoreData(void *dst, int id_x, int id_y, int in_w, int in_h, int in_size, float3 data)
    -> enable_if_t<image_type == ImageType::kYUV_I420, void> {
    int id_uv = id_y / 2 * in_w / 2 + id_x / 2;
    ((DataType *)dst)[id_y * in_w + id_x] = static_cast<DataType>(data.x);
    ((DataType *)dst)[in_size + id_uv] = static_cast<DataType>(data.y);
    ((DataType *)dst)[in_size + id_uv + (in_w / 2) * (in_h / 2)] = static_cast<DataType>(data.z);
    return;
}

template <typename DataType, ImageType image_type>
static inline __device__ auto StoreData(void *dst, int id_x, int id_y, int in_w, int in_h, int in_size, float3 data)
    -> enable_if_t<image_type == ImageType::kYUV_UYVY, void> {
    int id = id_y * in_w + id_x;
    if (id % 2 == 1) {
        ((DataType *)dst)[id * 2] = data.z;
        ((DataType *)dst)[id * 2 + 1] = data.x;
    } else {
        ((DataType *)dst)[id * 2] = data.y;
        ((DataType *)dst)[id * 2 + 1] = data.x;
    }
    return;
}

template <typename DataType, ImageType image_type>
static inline __device__ auto StoreData(void *dst, int id_x, int id_y, int in_w, int in_h, int in_size, float3 data)
    -> enable_if_t<image_type == ImageType::kYuv422p, void> {
    int id = id_y * in_w + id_x;

    if (id % 2 == 1) {
        ((DataType *)dst)[id / 2 + in_h * in_w * 3 / 2] = data.z;
        ((DataType *)dst)[id] = data.x;
    } else {
        ((DataType *)dst)[id / 2 + in_h * in_w] = data.y;
        ((DataType *)dst)[id] = data.x;
    }

    return;
}

template <typename DataType, ImageType image_type>
static inline __device__ auto StoreData(void *dst, int id_x, int id_y, int in_w, int in_h, int in_size, float data)
    -> enable_if_t<image_type == ImageType::kGRAY, void> {
    int id = id_y * in_w + id_x;
    ((DataType *)dst)[id] = data;
    return;
}

template <DataType data_type, ImageType image_type>
static inline __device__ auto StoreDataUnifiedTo255(void *dst, int idx, int id_y, int in_w, int in_h, int in_size,
                                                    typename ImageTypeTraits<image_type>::PixelType data)
    -> enable_if_t<data_type == DataType::kInt8, void> {
    return StoreData<unsigned char, image_type>(dst, idx, id_y, in_w, in_h, in_size, data);
}

template <DataType data_type, ImageType image_type>
static inline __device__ auto StoreDataUnifiedTo255(void *dst, int idx, int id_y, int in_w, int in_h, int in_size,
                                                    typename ImageTypeTraits<image_type>::PixelType data)
    -> enable_if_t<data_type == DataType::kFloat32 || data_type == DataType::kHalf, void> {
    auto tmpdata = data / 255.0f;
    return StoreData<typename DataTypeTraits<data_type>::Type, image_type>(dst, idx, id_y, in_w, in_h, in_size,
                                                                           tmpdata);
}
}  // namespace cudaop
}  // namespace smartmore

#endif