/*******************************************************************************
 *  FILENAME:      arc_length.h
 *
 *  AUTHORS:       Wang Shengxiang    START DATE: Tuesday September 7th 2021
 *
 *  LAST MODIFIED: Monday, September 13th 2021, 2:30:44 pm
 *
 *  CONTACT:       shengxiang.wang@smartmore.com
 *******************************************************************************/
#ifndef __SMARTMORE_CUDAOP_IMAGE_ARC_LENGTH_H__
#define __SMARTMORE_CUDAOP_IMAGE_ARC_LENGTH_H__

#include <cuda_runtime_api.h>
#include <cudaop/common/utils.h>

namespace smartmore {
namespace cudaop {
// compute last sub-arc length
template <typename T, bool is_closed>
static __device__ inline auto LastSubArcLength(void *curve, int num_point) -> enable_if_t<is_closed == false, float> {
    float dx = *(static_cast<T *>(curve));
    float dy = *(static_cast<T *>(curve) + 1);
    return sqrt(dx * dx + dy * dy);
}

template <typename T, bool is_closed>
static __device__ inline auto LastSubArcLength(void *curve, int num_point) -> enable_if_t<is_closed, float> {
    float dx0 = *(static_cast<T *>(curve));
    float dy0 = *(static_cast<T *>(curve) + 1);

    float dx1 = *(static_cast<T *>(curve) + num_point * 2 - 2);
    float dy1 = *(static_cast<T *>(curve) + num_point * 2 - 1);
    return sqrt(dx0 * dx0 + dy0 * dy0) + sqrt(dx1 * dx1 + dy1 * dy1);
}

template <typename T, bool is_closed>
static __global__ void ArcLengthKernel(void *curve, int num_point, float *arc_length) {
    int index = (blockDim.x * blockIdx.x + threadIdx.x) * 2;

    if (index >= num_point * 2 - 2) return;

    extern __shared__ float edge[];

    float dx = *(static_cast<T *>(curve) + index + 2) - *(static_cast<T *>(curve) + index);
    float dy = *(static_cast<T *>(curve) + index + 3) - *(static_cast<T *>(curve) + index + 1);

    edge[threadIdx.x] = sqrt(dx * dx + dy * dy);
    __syncthreads();

    for (int s = 1; threadIdx.x + s < num_point - 1 && threadIdx.x + s < blockDim.x; s *= 2) {
        if (threadIdx.x % (2 * s) == 0) {
            edge[threadIdx.x] = edge[threadIdx.x] + edge[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(arc_length, edge[0]);
    }
    // add last sub-arc length, according to is closed
    if (index == 0) {
        float last_length = LastSubArcLength<T, is_closed>(curve, num_point);
        atomicAdd(arc_length, last_length);
    }

    return;
}

template <typename T, bool is_closed>
void CudaArcLength(void *curve, int num_point, float *arc_length, cudaStream_t stream) {
    // input data type must be int or float, consistent with OpencV
    static_assert(std::is_same<T, float>::value || std::is_same<T, int>::value);

    float init_arc_length = 0.0f;
    cudaMemcpy(arc_length, &init_arc_length, sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(128);
    dim3 grid((num_point + block.x - 1) / block.x);
    ArcLengthKernel<T, is_closed><<<grid, block, block.x * sizeof(float), stream>>>(curve, num_point, arc_length);
    return;
}

template <typename T, bool is_closed>
void ArcLength(void *curve, int num_point, float *arc_length) {
    CudaArcLength<T, is_closed>(curve, num_point, arc_length, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
    return;
}

template <typename T, bool is_closed>
void ArcLengthAsync(void *curve, int num_point, float *arc_length, cudaStream_t stream) {
    CudaArcLength<T, is_closed>(curve, num_point, arc_length, stream);
    return;
}
}  // namespace cudaop
}  // namespace smartmore
#endif