/*******************************************************************************
 *  FILENAME:      border.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Saturday May 29th 2021
 *
 *  LAST MODIFIED: Thursday, September 16th 2021, 8:50:21 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_IMAGE_GENERIC_BORDER_H__
#define __SMARTMORE_CUDAOP_IMAGE_GENERIC_BORDER_H__

#include <type_traits>

#include "cudaop/types.h"

namespace smartmore {
namespace cudaop {

template <BorderType bordertype, int value>
__device__ int GetBorderIndex(int actual, int size);

// gfedcb | abcdefgh | gfedcba
template <BorderType bordertype, std::enable_if_t<bordertype == BorderType::kReflect, int> = 0>
__device__ int GetBorderIndex(int actual, int size) {
    int rst = actual >= 0 ? actual : ((-actual) % size + size) % size;
    rst = rst < size ? rst : ((2 * size - 2 - actual) % size + size) % size;
    return rst;
}

// fedcba | abcdefgh | hgfedcb
template <BorderType bordertype, std::enable_if_t<bordertype == BorderType::kReflectTotal, int> = 0>
__device__ int GetBorderIndex(int actual, int size) {
    int rst = actual >= 0 ? actual : ((-actual - 1) % size + size) % size;
    rst = rst < size ? rst : ((2 * size - 1 - actual) % size + size) % size;
    return rst;
}

// aaaaaa | abcdefgh | hhhhhhh
template <BorderType bordertype, std::enable_if_t<bordertype == BorderType::kReplicate, int> = 0>
__device__ int GetBorderIndex(int actual, int size) {
    int rst = actual >= 0 ? (actual < size ? actual : size - 1) : 0;
    return rst;
}

}  // namespace cudaop
}  // namespace smartmore

#endif