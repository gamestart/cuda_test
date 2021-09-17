/*******************************************************************************
 *  FILENAME:      imgconvert.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Wednesday March 10th 2021
 *
 *  LAST MODIFIED: Saturday, May 22nd 2021, 7:49:30 am
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_IMGCONVERT_H__
#define __SMARTMORE_CUDAOP_IMAGE_IMGCONVERT_H__

#include <cudaop/image/generic/loaddata.h>
#include <cudaop/image/generic/storedata.h>
#include <cudaop/image/generic/yuvformula.h>
#include <cudaop/types.h>

namespace smartmore {
namespace cudaop {
template <ImageType input_image_type, DataType input_data_type, ImageType output_image_type, DataType output_data_type,
          YUVFormula yuv_formula>
__global__ void CudaImageConvertKernel(void *src, void *dst, int in_h, int in_w, int in_size) {
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (id_x >= in_w || id_y >= in_h) return;

    float3 src_px = LoadDataAndUnifyTo255<input_data_type, input_image_type>(src, id_x, id_y, in_w, in_h, in_size);

    float3 dst_px = ConvertPixel<input_image_type, output_image_type, yuv_formula>(src_px.x, src_px.y, src_px.z);

    StoreDataUnifiedTo255<output_data_type, output_image_type>(dst, id_x, id_y, in_w, in_h, in_size, dst_px);
}

template <ImageType input_image_type, DataType input_data_type, ImageType output_image_type, DataType output_data_type,
          YUVFormula yuv_formula>
void CudaImageConvert(void *src, void *dst, int in_h, int in_w, cudaStream_t stream) {
    static_assert(input_image_type != ImageType::kGRAY && output_image_type != ImageType::kGRAY,
                  "ImageConvert does not support ImageType::kGRAY");
    dim3 block(32, 32);
    int grid_x = (in_w + block.x - 1) / block.x;
    int grid_y = (in_h + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);
    CudaImageConvertKernel<input_image_type, input_data_type, output_image_type, output_data_type, yuv_formula>
        <<<grid, block>>>(src, dst, in_h, in_w, in_h * in_w);
}

template <ImageType input_image_type, DataType input_data_type, ImageType output_image_type, DataType output_data_type,
          YUVFormula yuv_formula = YUVFormula::kBT601>
void ImageConvert(void *src, void *dst, int in_h, int in_w) {
    CudaImageConvert<input_image_type, input_data_type, output_image_type, output_data_type, yuv_formula>(
        src, dst, in_h, in_w, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}

template <ImageType input_image_type, DataType input_data_type, ImageType output_image_type, DataType output_data_type,
          YUVFormula yuv_formula = YUVFormula::kBT601>
void ImageConvertAsync(void *src, void *dst, int in_h, int in_w, cudaStream_t stream) {
    CudaImageConvert<input_image_type, input_data_type, output_image_type, output_data_type, yuv_formula>(
        src, dst, in_h, in_w, stream);
}
}  // namespace cudaop
}  // namespace smartmore

#endif