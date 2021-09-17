/*******************************************************************************
 *  FILENAME:      data_type_traits.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Friday May 14th 2021
 *
 *  LAST MODIFIED: Wednesday, May 19th 2021, 7:08:17 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_TYPE_TRAITS_DATA_TYPE_TRAITS_H__
#define __SMARTMORE_CUDAOP_TYPE_TRAITS_DATA_TYPE_TRAITS_H__

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudaop/types.h>

namespace smartmore {
namespace cudaop {
template <DataType data_type>
struct DataTypeTraits;

template <>
struct DataTypeTraits<DataType::kInt8> {
    typedef unsigned char Type;
};

template <>
struct DataTypeTraits<DataType::kFloat32> {
    typedef float Type;
};

template <>
struct DataTypeTraits<DataType::kHalf> {
    typedef half Type;
};
}  // namespace cudaop
}  // namespace smartmore

#endif