/*******************************************************************************
 *  FILENAME:      minAreaRect.h
 *
 *  AUTHORS:       Chen Fting    START DATE: Tuesday August 17th 2021
 *
 *  LAST MODIFIED: Tuesday, August 17th 2021, 8:43:46 pm
 *
 *  CONTACT:       fting.chen@smartmore.com
 *******************************************************************************/
#ifndef __SMARTMORE_CUDAOP_IMG_MINAREARECT_FILTER_H__
#define __SMARTMORE_CUDAOP_IMG_MINAREARECT_FILTER_H__

#include <cuda_runtime_api.h>
#include <cudaop/common/utils.h>
#include <cudaop/types.h>

namespace smartmore {
namespace cudaop {
__device__ int count = 0;
template <int block_size>
__global__ void MinAreaRectKernel(void *in, void *out, void *g_data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //直线方程Ax+By+C=0
    // shm保存11个信息，外接矩形面积，maxDis点坐标，minDis点坐标，maxLen点坐标，minLen点坐标，当前连线的A和B
    __shared__ float shm[block_size][11];
    shm[threadIdx.x][0] = 3.4e38;
    __syncthreads();

    if (2 * idx >= n * (n - 1)) {
        return;
    }
    int left = floor((2 * n - 1 - sqrtf(4 * n * n - 4 * n + 1 - 8 * idx)) / 2);
    int right = idx + 1 - left * (2 * n - 1 - left) / 2 + left;
    float maxDis = -1e20;
    float minDis = 3.4e38;
    float maxLen = -1e20;
    float minLen = 3.4e38;
    float vec_x = static_cast<float *>(in)[right * 2] - static_cast<float *>(in)[left * 2];
    float vec_y = static_cast<float *>(in)[right * 2 + 1] - static_cast<float *>(in)[left * 2 + 1];
    float A = static_cast<float *>(in)[right * 2 + 1] - static_cast<float *>(in)[left * 2 + 1];
    float B = static_cast<float *>(in)[left * 2] - static_cast<float *>(in)[right * 2];
    float C = static_cast<float *>(in)[right * 2] * static_cast<float *>(in)[left * 2 + 1] -
              static_cast<float *>(in)[left * 2] * static_cast<float *>(in)[right * 2 + 1];
    float sq = sqrtf(A * A + B * B);

    shm[threadIdx.x][9] = A;
    shm[threadIdx.x][10] = B;
    for (int i = 0; i < n; i++) {
        float dis = (static_cast<float *>(in)[i * 2] * A + static_cast<float *>(in)[i * 2 + 1] * B + C) / sq;
        if (dis > maxDis) {
            // maxDis
            shm[threadIdx.x][1] = static_cast<float *>(in)[i * 2];
            shm[threadIdx.x][2] = static_cast<float *>(in)[i * 2 + 1];
            maxDis = dis;
        }
        if (dis < minDis) {
            // minDis
            shm[threadIdx.x][3] = static_cast<float *>(in)[i * 2];
            shm[threadIdx.x][4] = static_cast<float *>(in)[i * 2 + 1];
            minDis = dis;
        }

        float len = (static_cast<float *>(in)[i * 2] * vec_x + static_cast<float *>(in)[i * 2 + 1] * vec_y) /
                    sqrtf(vec_x * vec_x + vec_y * vec_y);
        if (len > maxLen) {
            // maxlen
            shm[threadIdx.x][5] = static_cast<float *>(in)[i * 2];
            shm[threadIdx.x][6] = static_cast<float *>(in)[i * 2 + 1];
            maxLen = len;
        }
        if (len < minLen) {
            // minlen
            shm[threadIdx.x][7] = static_cast<float *>(in)[i * 2];
            shm[threadIdx.x][8] = static_cast<float *>(in)[i * 2 + 1];
            minLen = len;
        }
    }
    float S = fabs((maxLen - minLen) * (maxDis - minDis));
    shm[threadIdx.x][0] = S;
    __syncthreads();

    int tid = threadIdx.x;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shm[tid][0] > shm[tid + stride][0]) {
                for (int t = 0; t < 11; t++) {
                    float temp = shm[tid][t];
                    shm[tid][t] = shm[tid + stride][t];
                    shm[tid + stride][t] = temp;
                }
            }
        }
        __syncthreads();
    }
    __shared__ int isLast;
    const int bid = blockIdx.x;
    if (tid == 0) {
        for (int t = 0; t < 11; t++) {
            static_cast<float *>(g_data)[bid * 11 + t] = shm[tid][t];
        }
        __threadfence();
        int ticket = atomicAdd(&count, 1);
        isLast = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    if (isLast) {
        int data_per_thread = (gridDim.x + blockDim.x - 1) / blockDim.x;
        for (int i = 1; i < data_per_thread; i++) {
            int index = tid + blockDim.x * i;
            if (index < gridDim.x) {
                if (static_cast<float *>(g_data)[tid * 11] > static_cast<float *>(g_data)[index * 11]) {
                    for (int t = 0; t < 11; t++) {
                        float temp = static_cast<float *>(g_data)[tid * 11 + t];
                        static_cast<float *>(g_data)[tid * 11 + t] = static_cast<float *>(g_data)[index * 11 + t];
                        static_cast<float *>(g_data)[index * 11 + t] = temp;
                    }
                }
            }
            __syncthreads();
        }
        int border = blockDim.x * 2 > gridDim.x ? gridDim.x : blockDim.x;
        for (int i = border / 2; i > 0; i >>= 1) {
            if (tid < i) {
                if (static_cast<float *>(g_data)[tid * 11] > static_cast<float *>(g_data)[(tid + i) * 11]) {
                    for (int t = 0; t < 11; t++) {
                        float temp = static_cast<float *>(g_data)[tid * 11 + t];
                        static_cast<float *>(g_data)[tid * 11 + t] = static_cast<float *>(g_data)[(tid + i) * 11 + t];
                        static_cast<float *>(g_data)[(tid + i) * 11 + t] = temp;
                    }
                }
            }
            __syncthreads();
        }
        // gridDim.x为奇数的情况
        if (border == gridDim.x && tid == 0) {
            if (static_cast<float *>(g_data)[0] > static_cast<float *>(g_data)[(gridDim.x - 1) * 11]) {
                for (int t = 0; t < 11; t++) {
                    float temp = static_cast<float *>(g_data)[t];
                    static_cast<float *>(g_data)[t] = static_cast<float *>(g_data)[(gridDim.x - 1) * 11 + t];
                    static_cast<float *>(g_data)[(gridDim.x - 1) * 11 + t] = temp;
                }
            }
        }
    }

    if (threadIdx.x == 0 && isLast) {
        float A1 = static_cast<float *>(g_data)[9];
        float B1 = static_cast<float *>(g_data)[10];
        // minDis
        float C1 = -1.0f * (A1 * static_cast<float *>(g_data)[3] + B1 * static_cast<float *>(g_data)[4]);

        // maxDis
        float C2 = -1.0f * (A1 * static_cast<float *>(g_data)[1] + B1 * static_cast<float *>(g_data)[2]);

        // minlen
        float C3 = (B1 * static_cast<float *>(g_data)[7] - A1 * static_cast<float *>(g_data)[8]);

        // maxlen
        float C4 = (B1 * static_cast<float *>(g_data)[5] - A1 * static_cast<float *>(g_data)[6]);
        float x0 = (B1 * C3 - A1 * C1) / (A1 * A1 + B1 * B1);
        float y0 = -1.0f * (C1 * B1 + A1 * C3) / (A1 * A1 + B1 * B1);
        static_cast<float *>(out)[0] = x0 == 0 ? 0 : x0;
        static_cast<float *>(out)[1] = y0 == 0 ? 0 : y0;

        float x1 = (B1 * C3 - A1 * C2) / (A1 * A1 + B1 * B1);
        float y1 = -1.0f * (C2 * B1 + A1 * C3) / (A1 * A1 + B1 * B1);
        static_cast<float *>(out)[2] = x1 == 0 ? 0 : x1;
        static_cast<float *>(out)[3] = y1 == 0 ? 0 : y1;

        float x2 = (B1 * C4 - A1 * C2) / (A1 * A1 + B1 * B1);
        float y2 = -1.0f * (C2 * B1 + A1 * C4) / (A1 * A1 + B1 * B1);
        static_cast<float *>(out)[4] = x2 == 0 ? 0 : x2;
        static_cast<float *>(out)[5] = y2 == 0 ? 0 : y2;

        float x3 = (B1 * C4 - A1 * C1) / (A1 * A1 + B1 * B1);
        float y3 = -1.0f * (C1 * B1 + A1 * C4) / (A1 * A1 + B1 * B1);
        static_cast<float *>(out)[6] = x3 == 0 ? 0 : x3;
        static_cast<float *>(out)[7] = y3 == 0 ? 0 : y3;
    }
}

template <DataType data_type>
void CudaMinAreaRect(void *in, void *out, void *g_data, int n, cudaStream_t stream) {
    const int thread_per_block = 512;
    const int size = n * (n - 1) / 2;
    int grid_x = (size + thread_per_block - 1) / thread_per_block;
    MinAreaRectKernel<thread_per_block><<<grid_x, thread_per_block, 0, stream>>>(in, out, g_data, n);
}

template <DataType data_type>
void MinAreaRect(void *in, void *out, void *g_data, int n) {
    CudaMinAreaRect<data_type>(in, out, g_data, n, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}

template <DataType data_type>
void MinAreaRectAsync(void *in, void *out, void *g_data, int n, cudaStream_t stream) {
    CudaMinAreaRect<data_type>(in, out, g_data, n, stream);
}
}  // namespace cudaop
}  // namespace smartmore
#endif