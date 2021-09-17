/*******************************************************************************
 *  FILENAME:      utils.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Thursday March 11th 2021
 *
 *  LAST MODIFIED: Saturday, May 22nd 2021, 3:43:43 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_COMMON_UTILS_H__
#define __SMARTMORE_CUDAOP_COMMON_UTILS_H__

#include <stdexcept>
#include <string>
#include <type_traits>

namespace smartmore {
namespace cudaop {
template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;
}
}  // namespace smartmore

#define CUDAOP_CHECK_CUDA_SATUS(status)                                                            \
    do {                                                                                           \
        auto rst = status;                                                                         \
        if ((rst) != cudaSuccess) {                                                                \
            throw std::runtime_error("cuda err: " + std::to_string(static_cast<int>(rst)) + " (" + \
                                     cudaGetErrorString(rst) + ")" + " at " + __FILE__ + ":" +     \
                                     std::to_string(__LINE__));                                    \
        }                                                                                          \
    } while (0)

#define CUDAOP_ASSERT_TRUE(condition)                                                                              \
    do {                                                                                                           \
        if (!(condition)) {                                                                                        \
            throw std::runtime_error("assert faild: " + std::string(#condition) + " at " + std::string(__FILE__) + \
                                     ":" + std::to_string(__LINE__));                                              \
        }                                                                                                          \
    } while (0)

#endif
