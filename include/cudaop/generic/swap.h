/*******************************************************************************
 *  FILENAME:      swap.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Tuesday June 1st 2021
 *
 *  LAST MODIFIED: Tuesday, June 1st 2021, 12:30:23 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_GENERIC_SWAP_H__
#define __SMARTMORE_CUDAOP_GENERIC_SWAP_H__

namespace smartmore {
namespace cudaop {
template <typename T>
__device__ void Swap(T &x, T &y) {
    T z(x);
    x = y;
    y = z;
}
}  // namespace cudaop
}  // namespace smartmore

#endif