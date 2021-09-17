/*******************************************************************************
 *  FILENAME:      float3operators.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Friday March 19th 2021
 *
 *  LAST MODIFIED: Wednesday, May 19th 2021, 7:04:29 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_GENERIC_FLOAT3_OPERATORS_H__
#define __SMARTMORE_CUDAOP_GENERIC_FLOAT3_OPERATORS_H__

namespace smartmore {
namespace cudaop {
inline __device__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 operator/(const float3 &a, const float &b) { return make_float3(a.x / b, a.y / b, a.z / b); }

inline __device__ float3 operator*(const float &b, const float3 &a) { return make_float3(a.x * b, a.y * b, a.z * b); }

inline __device__ float3 operator*(const float3 &a, const float &b) { return make_float3(a.x * b, a.y * b, a.z * b); }
}  // namespace cudaop
}  // namespace smartmore

#endif