/*******************************************************************************
 *  FILENAME:      imgchannelmerge.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Friday March 12th 2021
 *
 *  LAST MODIFIED: Wednesday, May 19th 2021, 7:22:09 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_IMGCHANNELMERGE_H__
#define __SMARTMORE_CUDAOP_IMAGE_IMGCHANNELMERGE_H__

#include <cudaop/common/utils.h>
#include <cudaop/image/generic/storedata.h>
#include <cudaop/type_traits/data_type_traits.h>
#include <cudaop/type_traits/img_type_traits.h>
#include <cudaop/types.h>

#include <vector>

namespace smartmore {
namespace cudaop {
template <typename input_type, ImageType image_type>
static inline __device__ auto CudaImageChannelMergeKernelLoadData(void *src_yr, void *src_ug, void *src_vb, int id_x,
                                                                  int id_y, int in_w, int in_h)
    -> enable_if_t<ImageTypeTraits<image_type>::yuv_type == YUVType::NOTYUV ||
                       ImageTypeTraits<image_type>::yuv_type == YUVType::YUV444,
                   float3> {
    int id = id_y * in_w + id_x;
    return make_float3(((input_type *)src_yr)[id], ((input_type *)src_ug)[id], ((input_type *)src_vb)[id]);
}

template <typename input_type, ImageType image_type>
static inline __device__ auto CudaImageChannelMergeKernelLoadData(void *src_yr, void *src_ug, void *src_vb, int id_x,
                                                                  int id_y, int in_w, int in_h)
    -> enable_if_t<ImageTypeTraits<image_type>::yuv_type == YUVType::YUV420, float3> {
    int id_uv = id_y / 2 * in_w / 2 + id_x / 2;
    return make_float3(((input_type *)src_yr)[id_y * in_w + id_x], ((input_type *)src_ug)[id_uv],
                       ((input_type *)src_vb)[id_uv]);
}

template <typename input_type, ImageType image_type>
static inline __device__ auto CudaImageChannelMergeKernelLoadData(void *src_yr, void *src_ug, void *src_vb, int id_x,
                                                                  int id_y, int in_w, int in_h)
    -> enable_if_t<ImageTypeTraits<image_type>::yuv_type == YUVType::YUV422, float3> {
    int id_uv = id_y * in_w / 2 + id_x / 2;
    return make_float3(((input_type *)src_yr)[id_y * in_w + id_x], ((input_type *)src_ug)[id_uv],
                       ((input_type *)src_vb)[id_uv]);
}

template <DataType output_type, ImageType image_type>
static inline __device__ auto CudaImageChannelMergeKernelLoadDataUnifyTo255(void *src_yr, void *src_ug, void *src_vb,
                                                                            int id_x, int id_y, int in_w, int in_h)
    -> enable_if_t<output_type == DataType::kInt8, float3> {
    return CudaImageChannelMergeKernelLoadData<unsigned char, image_type>(src_yr, src_ug, src_vb, id_x, id_y, in_w,
                                                                          in_h);
}

template <DataType output_type, ImageType image_type>
static inline __device__ auto CudaImageChannelMergeKernelLoadDataUnifyTo255(void *src_yr, void *src_ug, void *src_vb,
                                                                            int id_x, int id_y, int in_w, int in_h)
    -> enable_if_t<output_type == DataType::kFloat32 || output_type == DataType::kHalf, float3> {
    float3 tmpdata = CudaImageChannelMergeKernelLoadData<typename DataTypeTraits<output_type>::Type, image_type>(
        src_yr, src_ug, src_vb, id_x, id_y, in_w, in_h);
    return make_float3(tmpdata.x * 255.0, tmpdata.y * 255.0, tmpdata.z * 255.0);
}

template <ImageType image_type, DataType input_data_type, DataType output_data_type>
__global__ void CudaImageChannelMergeKernel(void *src_yr, void *src_ug, void *src_vb, void *dst, int in_w, int in_h,
                                            int in_size) {
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (id_x >= in_w || id_y >= in_h) return;

    float3 px = CudaImageChannelMergeKernelLoadDataUnifyTo255<input_data_type, image_type>(src_yr, src_ug, src_vb, id_x,
                                                                                           id_y, in_w, in_h);

    return StoreDataUnifiedTo255<output_data_type, image_type>(dst, id_x, id_y, in_w, in_h, in_w * in_h, px);
}

template <ImageType image_type, DataType input_data_type, DataType output_data_type>
void CudaImageChannelMerge(void *src_yr, void *src_ug, void *src_vb, void *dst, int in_h, int in_w,
                           cudaStream_t stream) {
    dim3 block(32, 32);
    int grid_x = (in_w + block.x - 1) / block.x;
    int grid_y = (in_h + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);
    CudaImageChannelMergeKernel<image_type, input_data_type, output_data_type>
        <<<grid, block, 0, stream>>>(src_yr, src_ug, src_vb, dst, in_w, in_h, in_h * in_w);
}

template <ImageType image_type, DataType input_data_type, DataType output_data_type>
void ImageChannelMerge(const std::vector<void *> &srcs, void *dst, const int in_h, const int in_w) {
    CUDAOP_ASSERT_TRUE(srcs.size() == 3);
    CudaImageChannelMerge<image_type, input_data_type, output_data_type>(srcs[0], srcs[1], srcs[2], dst, in_h, in_w, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
    return;
}

template <ImageType image_type, DataType input_data_type, DataType output_data_type>
void ImageChannelMergeAsync(const std::vector<void *> &srcs, void *dst, const int in_h, const int in_w,
                            cudaStream_t stream) {
    CUDAOP_ASSERT_TRUE(srcs.size() == 3);
    CudaImageChannelMerge<image_type, input_data_type, output_data_type>(srcs[0], srcs[1], srcs[2], dst, in_h, in_w,
                                                                         stream);
    return;
}
}  // namespace cudaop
}  // namespace smartmore

#endif
