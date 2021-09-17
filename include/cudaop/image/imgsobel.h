/*******************************************************************************
 *  FILENAME:      imgsobel.h
 *
 *  AUTHORS:       Wang Shengxiang    START DATE: Monday September 6th 2021
 *
 *  LAST MODIFIED: Monday, September 6th 2021, 5:30:15 pm
 *
 *  CONTACT:       shengxiang.wang@smartmore.com
 *******************************************************************************/
#ifndef __SMARTMORE_CUDAOP_IMAGE_SOBEL_H__
#define __SMARTMORE_CUDAOP_IMAGE_SOBEL_H__

#include <cuda_runtime_api.h>
#include <cudaop/common/utils.h>
#include <cudaop/image/generic/border.h>
#include <cudaop/image/imgscharr.h>
#include <cudaop/type_traits/data_type_traits.h>
#include <cudaop/types.h>

namespace smartmore {
namespace cudaop {
template <DataType data_type>
using DataType_t = typename DataTypeTraits<data_type>::Type;

template <DataType input_data_type, DataType output_data_type, int ksize, BorderType border_type>
static __global__ void ImageSobelKernel(void *src, void *dst, int in_h, int in_w, int dx, int dy, double scale,
                                        double delta) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx >= in_w || idy >= in_h) return;

    DataType_t<input_data_type> data[9];
    GetValue<border_type, input_data_type>(static_cast<DataType_t<input_data_type> *>(src),
                                           static_cast<DataType_t<input_data_type> *>(&data[0]), idx, idy, in_h, in_w);

    DataType_t<output_data_type> der =
        data[2 + 4 * dy] - data[0] + 2 * (data[5 + 2 * dy] - data[3 - 2 * dy]) + data[8] - data[6 - 4 * dy];

    *(static_cast<DataType_t<output_data_type> *>(dst) + idy * in_w + idx) = der * scale + delta;

    return;
}

template <DataType input_data_type, DataType output_data_type, int ksize, BorderType border_type>
void CudaImageSobel(void *src, void *dst, int in_h, int in_w, cudaStream_t stream, int dx, int dy, double scale,
                    double delta) {
    // TODO: Allow for other kernel sizes, waiting for update from OpenCV
    static_assert(ksize == 3 || ksize == -1, "Sobel kernel size has to be -1(Scharr filter) or 3}");
    // if input data type is uchar, output must be fp32 or fp16 to prevent overflow
    // or, output data-type can also be uint16 etc.
    if (input_data_type == DataType::kInt8) {
        static_assert(output_data_type != DataType::kInt8);
    }

    if (!((dx == 1 && dy == 0) || (dx == 0 && dy == 1))) {
        throw std::runtime_error("Now, only the first derivative is supported.");
    }

    if (in_h < 1 || in_w < 1) {
        throw std::runtime_error("Image size is too small.");
    }

    dim3 block(32, 16);
    int grid_x = (in_w + block.x - 1) / block.x;
    int grid_y = (in_h + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);
    ImageSobelKernel<input_data_type, output_data_type, ksize, border_type>
        <<<grid, block, 0, stream>>>(src, dst, in_h, in_w, dx, dy, scale, delta);
    return;
}

template <DataType input_data_type, DataType output_data_type, int ksize = 3,
          BorderType border_type = BorderType::kReflect>
void ImageSobel(void *src, void *dst, int in_h, int in_w, int dx = 1, int dy = 0, double scale = 1.0,
                double delta = 0.0) {
    // -1 stands for Scharr filter:
    if (ksize == -1) {
        ImageScharr<input_data_type, output_data_type, border_type>(src, dst, in_h, in_w, dx, dy, scale, delta);
        return;
    }

    CudaImageSobel<input_data_type, output_data_type, ksize, border_type>(src, dst, in_h, in_w, 0, dx, dy, scale,
                                                                          delta);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}

template <DataType input_data_type, DataType output_data_type, int ksize = 3,
          BorderType border_type = BorderType::kReflect>
void ImageSobelAsync(void *src, void *dst, int in_h, int in_w, cudaStream_t stream, int dx = 1, int dy = 0,
                     double scale = 1.0, double delta = 0.0) {
    // -1 stands for Scharr filter:
    if (ksize == -1) {
        ImageScharrAsync<input_data_type, output_data_type, border_type>(src, dst, in_h, in_w, stream, dx, dy, scale,
                                                                         delta);
        return;
    }

    CudaImageSobel<input_data_type, output_data_type, ksize, border_type>(src, dst, in_h, in_w, stream, dx, dy, scale,
                                                                          delta);
    return;
}
}  // namespace cudaop
}  // namespace smartmore
#endif