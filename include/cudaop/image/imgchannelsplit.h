/*******************************************************************************
 *  FILENAME:      imgchannelsplit.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Thursday March 11th 2021
 *
 *  LAST MODIFIED: Thursday, September 16th 2021, 4:26:31 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_IMGCHANNELSPLIT_H__
#define __SMARTMORE_CUDAOP_IMAGE_IMGCHANNELSPLIT_H__

#include <cudaop/common/utils.h>
#include <cudaop/image/generic/loaddata.h>
#include <cudaop/type_traits/data_type_traits.h>
#include <cudaop/type_traits/img_type_traits.h>
#include <cudaop/types.h>

#include <vector>

namespace smartmore {
namespace cudaop {
template <typename output_type, ImageType image_type>
static inline __device__ auto CudaImageChannelSplitKernelStoreData(void *dst_yr, void *dst_ug, void *dst_vb, int id_x,
                                                                   int id_y, int in_w, int in_h, float3 data)
    -> enable_if_t<ImageTypeTraits<image_type>::yuv_type == YUVType::NOTYUV ||
                       ImageTypeTraits<image_type>::yuv_type == YUVType::YUV444,
                   void> {
    int id = id_y * in_w + id_x;
    ((output_type *)dst_yr)[id] = data.x;
    ((output_type *)dst_ug)[id] = data.y;
    ((output_type *)dst_vb)[id] = data.z;
}

template <typename output_type, ImageType image_type>
static inline __device__ auto CudaImageChannelSplitKernelStoreData(void *dst_yr, void *dst_ug, void *dst_vb, int id_x,
                                                                   int id_y, int in_w, int in_h, float3 data)
    -> enable_if_t<ImageTypeTraits<image_type>::yuv_type == YUVType::YUV420, void> {
    int id_uv = id_y / 2 * in_w / 2 + id_x / 2;
    ((output_type *)dst_yr)[id_y * in_w + id_x] = data.x;
    ((output_type *)dst_ug)[id_uv] = data.y;
    ((output_type *)dst_vb)[id_uv] = data.z;
}

template <typename output_type, ImageType image_type>
static inline __device__ auto CudaImageChannelSplitKernelStoreData(void *dst_yr, void *dst_ug, void *dst_vb, int id_x,
                                                                   int id_y, int in_w, int in_h, float3 data)
    -> enable_if_t<ImageTypeTraits<image_type>::yuv_type == YUVType::YUV422, void> {
    int id_uv = id_y * in_w / 2 + id_x / 2;
    ((output_type *)dst_yr)[id_y * in_w + id_x] = data.x;
    ((output_type *)dst_ug)[id_uv] = data.y;
    ((output_type *)dst_vb)[id_uv] = data.z;
}

template <DataType output_data_type, ImageType image_type>
static inline __device__ auto CudaImageChannelSplitKernelStoreDataUnifiedTo255(void *dst_yr, void *dst_ug, void *dst_vb,
                                                                               int id_x, int id_y, int in_w, int in_h,
                                                                               float3 data)
    -> enable_if_t<output_data_type == DataType::kInt8, void> {
    return CudaImageChannelSplitKernelStoreData<unsigned char, image_type>(dst_yr, dst_ug, dst_vb, id_x, id_y, in_w,
                                                                           in_h, data);
}

template <DataType output_data_type, ImageType image_type>
static inline __device__ auto CudaImageChannelSplitKernelStoreDataUnifiedTo255(void *dst_yr, void *dst_ug, void *dst_vb,
                                                                               int id_x, int id_y, int in_w, int in_h,
                                                                               float3 data)
    -> enable_if_t<output_data_type == DataType::kFloat32 || output_data_type == DataType::kHalf, void> {
    float3 tmpdata = make_float3(data.x / 255.0, data.y / 255.0, data.z / 255.0);
    return CudaImageChannelSplitKernelStoreData<DataTypeTraits<output_data_type>::Type, image_type>(
        dst_yr, dst_ug, dst_vb, id_x, id_y, in_w, in_h, tmpdata);
}

template <ImageType image_type, DataType input_data_type, DataType output_data_type>
__global__ void CudaImageChannelSplitKernel(void *src, void *dst_yr, void *dst_ug, void *dst_vb, int in_w, int in_h,
                                            int in_size) {
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (id_x >= in_w || id_y >= in_h) return;

    float3 px = LoadDataAndUnifyTo255<input_data_type, image_type>(src, id_x, id_y, in_w, in_h, in_size);

    CudaImageChannelSplitKernelStoreDataUnifiedTo255<output_data_type, image_type>(dst_yr, dst_ug, dst_vb, id_x, id_y,
                                                                                   in_w, in_h, px);

    return;
}

template <ImageType image_type, DataType input_data_type, DataType output_data_type>
void CudaImageChannelSplit(void *src, void *dst_yr, void *dst_ug, void *dst_vb, int in_h, int in_w,
                           cudaStream_t stream) {
    dim3 block(32, 32);
    int grid_x = (in_w + block.x - 1) / block.x;
    int grid_y = (in_h + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);
    CudaImageChannelSplitKernel<image_type, input_data_type, output_data_type>
        <<<grid, block, 0, stream>>>(src, dst_yr, dst_ug, dst_vb, in_w, in_h, in_h * in_w);
}

template <ImageType image_type, DataType input_data_type, DataType output_data_type>
void ImageChannelSplit(void *src, const std::vector<void *> &dsts, const int in_h, const int in_w) {
    CUDAOP_ASSERT_TRUE(dsts.size() == 3);
    CudaImageChannelSplit<image_type, input_data_type, output_data_type>(src, dsts[0], dsts[1], dsts[2], in_h, in_w, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}

template <ImageType image_type, DataType input_data_type, DataType output_data_type>
void ImageChannelSplitAsync(void *src, const std::vector<void *> &dsts, const int in_h, const int in_w,
                            cudaStream_t stream) {
    CUDAOP_ASSERT_TRUE(dsts.size() == 3);
    CudaImageChannelSplit<image_type, input_data_type, output_data_type>(src, dsts[0], dsts[1], dsts[2], in_h, in_w,
                                                                         stream);
}
}  // namespace cudaop
}  // namespace smartmore

#endif