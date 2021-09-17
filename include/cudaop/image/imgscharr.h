/*******************************************************************************
 *  FILENAME:      imgscharr.h
 *
 *  AUTHORS:       Wang Shengxiang    START DATE: Tuesday August 31st 2021
 *
 *  LAST MODIFIED: Thursday, September 16th 2021, 7:59:58 pm
 *
 *  CONTACT:       shengxiang.wang@smartmore.com
 *******************************************************************************/
#ifndef __SMARTMORE_CUDAOP_IMAGE_SCHARR_H__
#define __SMARTMORE_CUDAOP_IMAGE_SCHARR_H__

#include <cuda_runtime_api.h>
#include <cudaop/common/utils.h>
#include <cudaop/image/generic/border.h>
#include <cudaop/type_traits/data_type_traits.h>
#include <cudaop/types.h>

namespace smartmore {
namespace cudaop {
template <DataType data_type>
using DataType_t = typename DataTypeTraits<data_type>::Type;

template <BorderType border_type, DataType data_type, typename T = DataType_t<data_type>>
static inline auto __device__ GetValue(T *src, T *data, int idx, int idy, int in_h, int in_w)
    -> enable_if_t<border_type == BorderType::kReflect, void> {
    int index = 0;

#pragma unroll
    for (int j = -1; j < 2; j++) {
        int index_y = GetBorderIndex<border_type>(idy + j, in_h);
        for (int i = -1; i < 2; i++) {
            int index_x = GetBorderIndex<border_type>(idx + i, in_w);
            data[index] = src[index_y * in_w + index_x];
            index++;
        }
    }
    return;
}

template <BorderType border_type, DataType data_type, typename T = DataType_t<data_type>>
static inline auto __device__ GetValue(T *src, T *data, int idx, int idy, int in_h, int in_w)
    -> enable_if_t<border_type == BorderType::kReflectTotal, void> {
    int index = 0;

#pragma unroll
    for (int j = -1; j < 2; j++) {
        int index_y = GetBorderIndex<border_type>(idy + j, in_h);
        for (int i = -1; i < 2; i++) {
            int index_x = GetBorderIndex<border_type>(idx + i, in_w);
            data[index] = src[index_y * in_w + index_x];
            index++;
        }
    }
    return;
}

template <BorderType border_type, DataType data_type, typename T = DataType_t<data_type>>
static inline auto __device__ GetValue(T *src, T *data, int idx, int idy, int in_h, int in_w)
    -> enable_if_t<border_type == BorderType::kReplicate, void> {
    int index = 0;

#pragma unroll
    for (int j = -1; j < 2; j++) {
        int index_y = GetBorderIndex<border_type>(idy + j, in_h);
        for (int i = -1; i < 2; i++) {
            int index_x = GetBorderIndex<border_type>(idx + i, in_w);
            data[index] = src[index_y * in_w + index_x];
            index++;
        }
    }
    return;
}

template <BorderType border_type, DataType data_type, typename T = DataType_t<data_type>>
static inline auto __device__ GetValue(T *src, T *data, int idx, int idy, int in_h, int in_w)
    -> enable_if_t<border_type == BorderType::kConstant, void> {
    int index = 0;

#pragma unroll
    for (int j = -1; j < 2; j++) {
        int pos_y = idy + j;
        for (int i = -1; i < 2; i++) {
            int pos_x = idx + i;
            if (pos_x < 0 || pos_x >= in_w || pos_y < 0 || pos_y >= in_h) {
                data[index] = static_cast<T>(0);
            } else {
                data[index] = src[pos_y * in_w + pos_x];
            }
            index++;
        }
    }
    return;
}

// Scharr filter:
// [-3,  0, 3]
// [-10, 0, 10]
// [-3,  0, 3]
template <DataType input_data_type, DataType output_data_type, BorderType border_type>
static __global__ void ImageScharrKernel(void *src, void *dst, int in_h, int in_w, int dx, int dy, double scale,
                                         double delta) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx >= in_w || idy >= in_h) return;

    DataType_t<input_data_type> data[9];
    GetValue<border_type, input_data_type>(static_cast<DataType_t<input_data_type> *>(src),
                                           static_cast<DataType_t<input_data_type> *>(&data[0]), idx, idy, in_h, in_w);

    // deriv of x or y, according to the value of dy
    DataType_t<output_data_type> der = 3 * (data[2 + 4 * dy] - data[0]) + 10 * (data[5 + 2 * dy] - data[3 - 2 * dy]) +
                                       3 * (data[8] - data[6 - 4 * dy]);

    *(static_cast<DataType_t<output_data_type> *>(dst) + idy * in_w + idx) = scale * der + delta;

    return;
}

template <DataType input_data_type, DataType output_data_type, BorderType border_type>
void CudaImageScharr(void *src, void *dst, int in_h, int in_w, cudaStream_t stream, int dx, int dy, double scale,
                     double delta) {
    // if input data type is uchar, output must be fp32 or fp16 to prevent overflow
    // or, output data-type can also be uint16 etc.
    if (input_data_type == DataType::kInt8) {
        static_assert(output_data_type != DataType::kInt8);
    }

    if (!((dx == 1 && dy == 0) || (dx == 0 && dy == 1))) {
        throw std::runtime_error("Invalid deriv para.");
    }

    if (in_h < 1 || in_w < 1) {
        throw std::runtime_error("Image size is too small.");
    }

    dim3 block(32, 16);
    int grid_x = (in_w + block.x - 1) / block.x;
    int grid_y = (in_h + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);
    ImageScharrKernel<input_data_type, output_data_type, border_type>
        <<<grid, block, 0, stream>>>(src, dst, in_h, in_w, dx, dy, scale, delta);
    return;
}

template <DataType input_data_type, DataType output_data_type, BorderType border_type = BorderType::kReflect>
void ImageScharr(void *src, void *dst, int in_h, int in_w, int dx = 1, int dy = 0, double scale = 1.0,
                 double delta = 0.0) {
    CudaImageScharr<input_data_type, output_data_type, border_type>(src, dst, in_h, in_w, 0, dx, dy, scale, delta);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
    return;
}

template <DataType input_data_type, DataType output_data_type, BorderType border_type = BorderType::kReflect>
void ImageScharrAsync(void *src, void *dst, int in_h, int in_w, cudaStream_t stream, int dx = 1, int dy = 0,
                      double scale = 1.0, double delta = 0.0) {
    CudaImageScharr<input_data_type, output_data_type, border_type>(src, dst, in_h, in_w, stream, dx, dy, scale, delta);
    return;
}
}  // namespace cudaop
}  // namespace smartmore
#endif