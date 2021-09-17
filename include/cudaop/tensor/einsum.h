/*******************************************************************************
 *  FILENAME:      einsum.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Thursday March 11th 2021
 *
 *  LAST MODIFIED: Tuesday, June 15th 2021, 11:27:49 am
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_TENSOR_EINSUM_H__
#define __SMARTMORE_CUDAOP_TENSOR_EINSUM_H__

#include <cudaop/common/utils.h>
#include <cudaop/types.h>

#include <functional>
#include <map>
#include <stdexcept>
#include <string>

namespace smartmore {
namespace cudaop {
__global__ void CudaEinsum_kBNHW_BNCHW_To_BCHW_Kernel(float *src_bnhw, float *src_bnchw, float *dst, const int in_b,
                                                      const int in_n, const int in_c, const int in_h, const int in_w,
                                                      const int bnhw_step, const int bnchw_step) {
    const int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (id_x >= in_w || id_y >= in_h) return;

    for (int i = 0; i < in_b; i++) {
        const int bnhw_id = i * in_n * in_h * in_w + id_y * in_w + id_x;
        for (int j = 0; j < in_c; j++) {
            const int bnchw_id = i * in_n * in_c * in_h * in_w + j * in_h * in_w + id_y * in_w + id_x;
            const int rst_id = i * in_c * in_h * in_w + j * in_h * in_w + id_y * in_w + id_x;
            dst[rst_id] = 0;

            for (int k = 0; k < in_n; k++) {
                dst[rst_id] += src_bnhw[bnhw_id + k * bnhw_step] * src_bnchw[bnchw_id + k * bnchw_step];
            }
        }
    }
}

void CudaEinsum_kBNHW_BNCHW_To_BCHW(float *src_bnhw, float *src_bnchw, float *dst, const int in_b, const int in_n,
                                    const int in_c, const int in_h, const int in_w, cudaStream_t stream) {
    dim3 block(32, 32);
    int grid_x = (in_w + block.x - 1) / block.x;
    int grid_y = (in_h + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);
    CudaEinsum_kBNHW_BNCHW_To_BCHW_Kernel<<<grid, block, 0, stream>>>(src_bnhw, src_bnchw, dst, in_b, in_n, in_c, in_h,
                                                                      in_w, in_h * in_w, in_c * in_h * in_w);
}

__global__ void CudaEinsum_kBCHW_BCHW_To_BHW_Kernel(float *src1, float *src2, float *dst, const int in_b,
                                                    const int in_c, const int in_h, const int in_w) {
    const int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t id_by = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= in_w || id_by >= in_h * in_b) return;
    const int id_y = id_by % in_h;
    const int batch_id = id_by / in_h;
    const size_t out_pannel = in_h * in_w;
    const size_t in_pannel = in_h * in_w * in_c;
    size_t out_idx = out_idx = id_y * in_w + id_x + batch_id * out_pannel;
    dst[out_idx] = 0;
    size_t in_idx;
    for (int c = 0; c < in_c; c++) {
        in_idx = c * in_h * in_w + id_y * in_w + id_x + batch_id * in_pannel;
        dst[out_idx] += (src1[in_idx] * src2[in_idx]);
    }
}

void CudaEinsum_kBCHW_BCHW_To_BHW(float *src1, float *src2, float *dst, const int in_b, const int in_c, const int in_h,
                                  const int in_w, cudaStream_t stream) {
    dim3 block(32, 32);
    int grid_x = (in_w + block.x - 1) / block.x;
    int grid_y = (in_h * in_b + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);
    CudaEinsum_kBCHW_BCHW_To_BHW_Kernel<<<grid, block, 0, stream>>>(src1, src2, dst, in_b, in_c, in_h, in_w);
}

template <EinsumType einsum_type>
void CudaEinsum(float *src1, float *src2, float *dst, const std::vector<size_t> &dims, cudaStream_t stream) {
    switch (einsum_type) {
        case EinsumType::kBNHW_BNCHW_To_BCHW:
            CudaEinsum_kBNHW_BNCHW_To_BCHW(src1, src2, dst, dims[0], dims[1], dims[2], dims[3], dims[4], stream);
            break;
        case EinsumType::kBCHW_BCHW_To_BHW:
            CudaEinsum_kBCHW_BCHW_To_BHW(src1, src2, dst, dims[0], dims[1], dims[2], dims[3], stream);
            break;
        default:
            throw std::runtime_error("unsupportted einsum_type");
    }
}

template <EinsumType einsum_type>
void Einsum(float *src1, float *src2, float *dst, const std::vector<size_t> &dims) {
    CudaEinsum<einsum_type>(src1, src2, dst, dims, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}

template <EinsumType einsum_type>
void EinsumAsync(float *src1, float *src2, float *dst, const std::vector<size_t> &dims, cudaStream_t stream) {
    CudaEinsum<einsum_type>(src1, src2, dst, dims, stream);
}
}  // namespace cudaop
}  // namespace smartmore

#endif