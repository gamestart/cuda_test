/*******************************************************************************
 *  FILENAME:      halfoperators.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Wednesday May 19th 2021
 *
 *  LAST MODIFIED: Thursday, May 20th 2021, 5:19:55 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_GENERIC_HALF_OPERATORS_H__
#define __SMARTMORE_CUDAOP_GENERIC_HALF_OPERATORS_H__

#include <cuda_fp16.h>

namespace smartmore {
namespace cudaop {
#if __CUDA_ARCH__ <= 50
inline __device__ bool operator<(half &a, half &b) { return __hlt(a, b); }

inline __device__ bool operator>(half &a, half &b) { return __hgt(a, b); }

inline __device__ bool operator<=(half &a, half &b) { return __hle(a, b); }

inline __device__ bool operator>=(half &a, half &b) { return __hge(a, b); }
#else
inline __device__ bool operator<(half &a, half &b) { return float(a) < float(b); }

inline __device__ bool operator>(half &a, half &b) { return float(a) > float(b); }

inline __device__ bool operator<=(half &a, half &b) { return float(a) <= float(b); }

inline __device__ bool operator>=(half &a, half &b) { return float(a) >= float(b); }
#endif

}  // namespace cudaop
}  // namespace smartmore
#endif