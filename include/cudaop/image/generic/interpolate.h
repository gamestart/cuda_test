/*******************************************************************************
 *  FILENAME:      interpolate.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Thursday March 18th 2021
 *
 *  LAST MODIFIED: Wednesday, May 19th 2021, 7:05:04 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_GENERIC_INTERPOLATE_H__
#define __SMARTMORE_CUDAOP_IMAGE_GENERIC_INTERPOLATE_H__

#include <cudaop/common/utils.h>
#include <cudaop/generic/float3operators.h>
#include <cudaop/image/generic/loaddata.h>
#include <cudaop/types.h>

#define CLAMPD(x, a) ((x) < (a) ? (a) : (x))
#define CLAMPU(x, b) ((x) > (b) ? (b) : (x))
#define CLAMP(x, a, b) ((x) >= (a) ? ((x) <= (b) ? (x) : (b)) : (a))

namespace smartmore {
namespace cudaop {
template <typename T>
inline __device__ T GenZero();

template <>
inline __device__ float3 GenZero<float3>() {
    return make_float3(0, 0, 0);
}

template <>
inline __device__ float GenZero<float>() {
    return 0.0f;
}

// Bilinear Interpolation
// From jiawen.guan@smartmore.com
template <ImageType input_image_type, DataType input_data_type, ResizeAlgoType algo_type>
static __device__ auto Interpolate(void *src, int in_h, int in_w, int in_size, int dst_x, int dst_y, float fh, float fw)
    -> enable_if_t<algo_type == ResizeAlgoType::kBilinear, typename ImageTypeTraits<input_image_type>::PixelType> {
    const float src_x = (float(dst_x) + 0.5f) * fw - 0.5;
    const float src_y = (float(dst_y) + 0.5f) * fh - 0.5;

    // i, j
    int si = __float2int_rd(src_x);
    int sj = __float2int_rd(src_y);
    si = CLAMP(si, 0, in_w - 1);
    sj = CLAMP(sj, 0, in_h - 1);
    // i+1, j+1
    int sip1 = CLAMPU(si + 1, in_w - 1);
    int sjp1 = CLAMPU(sj + 1, in_h - 1);

    // u, v
    float su = src_x - float(si);
    float sv = src_y - float(sj);
    su = CLAMPD(su, 0.0f);
    sv = CLAMPD(sv, 0.0f);
    // 1-u, 1-v
    float ssu = 1.0f - su;
    float ssv = 1.0f - sv;

    // f(i, j)
    auto q11 = LoadDataAndUnifyTo255<input_data_type, input_image_type>(src, si, sj, in_w, in_h, in_size);
    // f(i+1, j)
    auto q12 = LoadDataAndUnifyTo255<input_data_type, input_image_type>(src, sip1, sj, in_w, in_h, in_size);
    // f(i, j+1)
    auto q21 = LoadDataAndUnifyTo255<input_data_type, input_image_type>(src, si, sjp1, in_w, in_h, in_size);
    // f(i+1, j+1)
    auto q22 = LoadDataAndUnifyTo255<input_data_type, input_image_type>(src, sip1, sjp1, in_w, in_h, in_size);

    auto out = ssu * ssv * q11 + su * ssv * q12 + ssu * sv * q21 + su * sv * q22;

    return out;
}

// Nearest Interpolation
template <ImageType input_image_type, DataType input_data_type, ResizeAlgoType algo_type>
static __device__ auto Interpolate(void *src, int in_h, int in_w, int in_size, int dst_x, int dst_y, float fh, float fw)
    -> enable_if_t<algo_type == ResizeAlgoType::kNearest, typename ImageTypeTraits<input_image_type>::PixelType> {
    const float src_x = float(dst_x) * fw;
    const float src_y = float(dst_y) * fh;

    // i, j
    int si = __float2int_rd(src_x);
    int sj = __float2int_rd(src_y);
    si = CLAMP(si, 0, in_w - 1);
    sj = CLAMP(sj, 0, in_h - 1);

    // f(i, j)
    auto q = LoadDataAndUnifyTo255<input_data_type, input_image_type>(src, si, sj, in_w, in_h, in_size);

    return q;
}

static __device__ inline float InterpolationCalculateBicubic(float x) {
    const float A = -0.75f;
    float abs_x = x >= 0 ? x : -x;
    const float x2 = x * x;
    const float x3 = abs_x * x2;

    if (abs_x <= 1) {
        return 1 - (A + 3) * x2 + (A + 2) * x3;
    } else if (abs_x <= 2) {
        return -4 * A + 8 * A * abs_x - 5 * A * x2 + A * x3;
    }

    return 0;
}

// Bicubic Interpolation
template <ImageType input_image_type, DataType input_data_type, ResizeAlgoType algo_type>
static __device__ auto Interpolate(void *src, int in_h, int in_w, int in_size, int dst_x, int dst_y, float fh, float fw)
    -> enable_if_t<algo_type == ResizeAlgoType::kBicubic, typename ImageTypeTraits<input_image_type>::PixelType> {
    using PixelType = typename ImageTypeTraits<input_image_type>::PixelType;
    const float src_x = (float(dst_x) + 0.5f) * fw - 0.5;
    const float src_y = (float(dst_y) + 0.5f) * fh - 0.5;

    // i, j
    int si = __float2int_rd(src_x);
    int sj = __float2int_rd(src_y);
    si = CLAMPU(si, in_w - 1);
    sj = CLAMPU(sj, in_h - 1);

    // 4*4 points coordinate around (src_x,src_y)
    int coord_x[4][4];
    int coord_y[4][4];
    for (int di = -1; di <= 2; di++) {
        for (int dj = -1; dj <= 2; dj++) {
            coord_x[di + 1][dj + 1] = CLAMP(si + di, 0, in_w - 1);
            coord_y[di + 1][dj + 1] = CLAMP(sj + dj, 0, in_h - 1);
        }
    }

    // 4*4 points value around (src_x,src_y)
    PixelType p[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            p[i][j] = LoadDataAndUnifyTo255<input_data_type, input_image_type>(src, coord_x[i][j], coord_y[i][j], in_w,
                                                                               in_h, in_size);
        }
    }

    // calculate (x^i)(y^j)
    float u = src_x - float(si);
    float v = src_y - float(sj);

    // calculate g(x)=sigma(0,3,sigma(0,3,x^i*y^j))
    PixelType out = GenZero<PixelType>();

    for (int i = -1; i <= 2; i++) {
        for (int j = -1; j <= 2; j++) {
            out = out + p[i + 1][j + 1] * InterpolationCalculateBicubic(i - u) * InterpolationCalculateBicubic(j - v);
        }
    }

    return out;
}
}  // namespace cudaop
}  // namespace smartmore

#endif
