/*******************************************************************************
 *  FILENAME:      imgcopymakeborder.h
 *
 *  AUTHORS:       Hou Yue    START DATE: Wednesday August 11th 2021
 *
 *  LAST MODIFIED: Sunday, August 15th 2021, 6:40:08 pm
 *
 *  CONTACT:       yue.hou@smartmore.com
 *******************************************************************************/

#include <cudaop/common/utils.h>
#include <cudaop/image/generic/loaddata.h>
#include <cudaop/image/generic/storedata.h>
#include <cudaop/types.h>

namespace smartmore {
namespace cudaop {
template <DataType data_type, ImageType image_type>
static inline __device__ auto LoadConstantValue(float value) -> enable_if_t<image_type == ImageType::kGRAY, float> {
    return value;
}

template <DataType data_type, ImageType image_type>
static inline __device__ auto LoadConstantValue(float value) -> enable_if_t<image_type != ImageType::kGRAY, float3> {
    return make_float3(value, value, value);
}

template <CopyBorderType copyborder_type>
static inline __device__ auto IndexMapping(int lower, int upper, int size, int ori_index)
    -> enable_if_t<copyborder_type == CopyBorderType::kBorder_Replicate, int> {
    int dst_index = (ori_index > lower) ? (ori_index > (lower + size - 1) ? (size - 1) : (ori_index - lower)) : 0;
    return dst_index;
}

template <CopyBorderType copyborder_type>
static inline __device__ auto IndexMapping(int lower, int upper, int size, int ori_index)
    -> enable_if_t<copyborder_type == CopyBorderType::kBorder_Reflect, int> {
    int lower_val = (lower - ori_index - 1) % (size * 2) < size ? (lower - ori_index - 1) % (size * 2)
                                                                : (2 * size - (lower - ori_index - 1) % (size * 2) - 1);
    int upper_val = (ori_index - lower - size) % (size * 2) < size ? size - (ori_index - lower - size) % (size * 2) - 1
                                                                   : (ori_index - lower - size) % (size * 2) - size;

    int dst_index = ori_index < lower ? lower_val : (ori_index >= lower + size ? upper_val : ori_index - lower);
    return dst_index;
}

template <CopyBorderType copyborder_type>
static inline __device__ auto IndexMapping(int lower, int upper, int size, int ori_index)
    -> enable_if_t<copyborder_type == CopyBorderType::kBorder_Reflect_101, int> {
    int lower_val = (lower - ori_index - 1) % (size * 2 - 2) < size - 1
                        ? (lower - ori_index - 1) % (size * 2 - 2) + 1
                        : (size - 1) * 2 - ((lower - ori_index - 1) % (size * 2 - 2)) - 1;
    int upper_val = (ori_index - lower - size) % (size * 2 - 2) < size - 1
                        ? size - (ori_index - lower - size) % (size * 2 - 2) - 2
                        : (ori_index - lower - size) % (size * 2 - 2) - size + 2;

    int dst_index = ori_index < lower ? lower_val : (ori_index >= lower + size ? upper_val : ori_index - lower);
    return dst_index;
}

template <CopyBorderType copyborder_type>
static inline __device__ auto IndexMapping(int lower, int upper, int size, int ori_index)
    -> enable_if_t<copyborder_type == CopyBorderType::kBorder_Warp, int> {
    int lk = ((lower + size - 1) - ori_index) / size, uk = (ori_index - lower) / size;
    int dst_index = ori_index < lower
                        ? (lk * size - lower + ori_index)
                        : (ori_index >= lower + size ? (ori_index - lower - uk * size) : ori_index - lower);
    return dst_index;
}

template <DataType data_type, ImageType image_type, CopyBorderType copyborder_type>
static inline __device__ auto CudaImageCopyMakeBorderKernelOperate(void *src, void *dst, unsigned int height,
                                                                   unsigned int width, int out_w, int out_h, int top,
                                                                   int bottom, int left, int right, int idx, int idy,
                                                                   float value)
    -> enable_if_t<copyborder_type == CopyBorderType::kBorder_Constant, void> {
    if (idx < left || idx > (left + width - 1) || idy < top || idy > (top + height - 1)) {
        auto data = LoadConstantValue<data_type, image_type>(value);
        StoreData<typename DataTypeTraits<data_type>::Type, image_type>(dst, idx, idy, out_w, out_h, out_w * out_h,
                                                                        data);
    } else {
        auto in_data = LoadData<typename DataTypeTraits<data_type>::Type, image_type>(src, idx - left, idy - top, width,
                                                                                      height, width * height);
        StoreData<typename DataTypeTraits<data_type>::Type, image_type>(dst, idx, idy, out_w, out_h, out_w * out_h,
                                                                        in_data);
    }
}

template <DataType data_type, ImageType image_type, CopyBorderType copyborder_type>
static inline __device__ auto CudaImageCopyMakeBorderKernelOperate(void *src, void *dst, unsigned int height,
                                                                   unsigned int width, int out_w, int out_h, int top,
                                                                   int bottom, int left, int right, int idx, int idy,
                                                                   float value)
    -> enable_if_t<copyborder_type != CopyBorderType::kBorder_Constant, void> {
    int id_x = IndexMapping<copyborder_type>(left, right, width, idx);
    int id_y = IndexMapping<copyborder_type>(top, bottom, height, idy);
    auto in_data =
        LoadData<typename DataTypeTraits<data_type>::Type, image_type>(src, id_x, id_y, width, height, width * height);
    StoreData<typename DataTypeTraits<data_type>::Type, image_type>(dst, idx, idy, out_w, out_h, out_w * out_h,
                                                                    in_data);
}

template <DataType data_type, ImageType image_type, CopyBorderType copyborder_type>
__global__ void CudaImageCopyMakeBorderKernel(void *src, void *dst, unsigned int height, unsigned int width, int out_w,
                                              int out_h, int top, int bottom, int left, int right, float value) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx >= out_w || idy >= out_h) return;
    CudaImageCopyMakeBorderKernelOperate<data_type, image_type, copyborder_type>(
        src, dst, height, width, out_w, out_h, top, bottom, left, right, idx, idy, value);
}

template <DataType data_type, ImageType image_type, CopyBorderType copyborder_type>
void CudaImageCopyMakeBorder(void *src, void *dst, unsigned int height, unsigned int width, int top, int bottom,
                             int left, int right, float value, cudaStream_t stream) {
    CUDAOP_ASSERT_TRUE(src != nullptr && dst != nullptr);
    dim3 block(32, 32);
    int out_w = left + width + right;
    int out_h = top + height + bottom;
    int grid_x = (out_w + block.x - 1) / block.x;
    int grid_y = (out_h + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);
    CudaImageCopyMakeBorderKernel<data_type, image_type, copyborder_type>
        <<<grid, block, 0, stream>>>(src, dst, height, width, out_w, out_h, top, bottom, left, right, value);
}

template <DataType data_type, ImageType image_type, CopyBorderType copyborder_type>
void ImageCopyMakeBorder(void *src, void *dst, unsigned int height, unsigned int width, int top, int bottom, int left,
                         int right, float value = 0.f) {
    CudaImageCopyMakeBorder<data_type, image_type, copyborder_type>(src, dst, height, width, top, bottom, left, right,
                                                                    value, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
    return;
}

template <DataType data_type, ImageType image_type, CopyBorderType copyborder_type>
void ImageCopyMakeBorderAsync(void *src, void *dst, unsigned int height, unsigned int width, int top, int bottom,
                              int left, int right, float value = 0.f, cudaStream_t stream = 0) {
    CudaImageCopyMakeBorder<data_type, image_type, copyborder_type>(src, dst, height, width, top, bottom, left, right,
                                                                    value, stream);
    return;
}
}  // namespace cudaop
}  // namespace smartmore