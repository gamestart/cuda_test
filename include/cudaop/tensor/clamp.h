#ifndef __SMARTMORE_CUDAOP_TENSOR_CLAMP_H__
#define __SMARTMORE_CUDAOP_TENSOR_CLAMP_H__

#include <cudaop/common/utils.h>
#include <cudaop/type_traits/data_type_traits.h>
#include <cudaop/types.h>

namespace {
template <typename T>
inline __device__ T Clamp(const T x, float a, float b) {
    float tmpf = float(x);
    return T(tmpf >= a ? tmpf <= b ? tmpf : b : a);
}
}  // namespace

namespace smartmore {
namespace cudaop {
template <typename T, DataType data_type>
__global__ void CudaClampKernel(void *src, unsigned int length, float lower_bound, float upper_bound) {
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;

    if (id_x >= length) return;

    *(static_cast<T *>(src) + id_x) = Clamp(*(static_cast<T *>(src) + id_x), lower_bound, upper_bound);
}

template <DataType data_type>
void CudaClamp(void *src, unsigned int length, float lower_bound, float upper_bound, cudaStream_t stream) {
    dim3 block(1024);
    dim3 grid((length + block.x - 1) / block.x);
    CudaClampKernel<typename DataTypeTraits<data_type>::Type, data_type>
        <<<grid, block, 0, stream>>>(src, length, lower_bound, upper_bound);
}

template <DataType data_type>
void Clamp(void *src, unsigned int length, float lower_bound, float upper_bound) {
    CudaClamp<data_type>(src, length, lower_bound, upper_bound, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}

template <DataType data_type>
void ClampAsync(void *src, unsigned int length, float lower_bound, float upper_bound, cudaStream_t stream) {
    // assign stream by user
    CudaClamp<data_type>(src, length, lower_bound, upper_bound, stream);
}
}  // namespace cudaop
}  // namespace smartmore

#endif