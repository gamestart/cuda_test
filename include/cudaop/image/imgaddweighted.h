/*******************************************************************************
 *  FILENAME:      imgaddweighted.h
 *
 *  AUTHORS:       Liang Jia    START DATE: Saturday August 14th 2021
 *
 *  LAST MODIFIED: Saturday, August 14th 2021, 2:24:11 pm
 *
 *  CONTACT:       jia.liang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_IMGADDWEIGHTED_H__
#define __SMARTMORE_CUDAOP_IMAGE_IMGADDWEIGHTED_H__
#include <cudaop/common/utils.h>
#include <cudaop/image/generic/loaddata.h>
#include <cudaop/image/generic/storedata.h>

namespace smartmore {
namespace cudaop {

template <ImageType image_type, DataType input_data_type>
__global__ void CudaImageAddWeightedKernel(void *src1, float alpha, void *src2, float beta, float gamma, int src_w,
                                           int src_h, void *dst, int in_size) {
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (id_x >= src_w || id_y >= src_h) return;
    auto src1_data =
        LoadData<typename DataTypeTraits<input_data_type>::Type, image_type>(src1, id_x, id_y, src_w, src_h, in_size);
    auto src2_data =
        LoadData<typename DataTypeTraits<input_data_type>::Type, image_type>(src2, id_x, id_y, src_w, src_h, in_size);
    auto dst_data = src1_data * alpha + src2_data * beta + make_float3(gamma, gamma, gamma);
    StoreData<typename DataTypeTraits<input_data_type>::Type, image_type>(dst, id_x, id_y, src_w, src_h, in_size,
                                                                          dst_data);
}

template <ImageType image_type, DataType input_data_type>
void CudaImageAddWeighted(void *src1, float alpha, void *src2, float beta, float gamma, const Size &src_size, void *dst,
                          cudaStream_t stream) {
    CUDAOP_ASSERT_TRUE(src1 != nullptr && src2 != nullptr && dst != nullptr);

    dim3 block(32, 32);
    int grid_x = (src_size.width + block.x - 1) / block.x;
    int grid_y = (src_size.height + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);

    auto roi_w = src_size.width;
    auto roi_h = src_size.height;
    CudaImageAddWeightedKernel<image_type, input_data_type><<<grid, block, 0, stream>>>(
        src1, alpha, src2, beta, gamma, src_size.width, src_size.height, dst, roi_w * roi_h);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}

template <ImageType image_type, DataType input_data_type>
void ImageAddWeighted(void *src1, float alpha, void *src2, float beta, float gamma, const Size &src_size, void *dst) {
    CudaImageAddWeighted<image_type, input_data_type>(src1, alpha, src2, beta, gamma, src_size, dst, 0);
    cudaDeviceSynchronize();
    return;
}

template <ImageType image_type, DataType input_data_type>
void ImageAddWeightedAsync(void *src1, float alpha, void *src2, float beta, float gamma, const Size &src_size,
                           void *dst, cudaStream_t stream) {
    CudaImageAddWeighted<image_type, input_data_type>(src1, alpha, src2, beta, gamma, src_size, dst, stream);
    return;
}
}  // namespace cudaop
}  // namespace smartmore

#endif /* __SMARTMORE_CUDAOP_IMAGE_IMGADDWEIGHTED_H__ */
