/*******************************************************************************
 *  FILENAME:      datatypeconvert.h
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Wednesday May 19th 2021
 *
 *  LAST MODIFIED: Thursday, May 20th 2021, 3:43:46 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/

#ifndef __SMARTMORE_CUDAOP_TENSOR_DATATYPECONVERT_H__
#define __SMARTMORE_CUDAOP_TENSOR_DATATYPECONVERT_H__

#include <cudaop/generic/halfoperators.h>
#include <cudaop/type_traits/data_type_traits.h>
#include <cudaop/types.h>

namespace smartmore {
namespace cudaop {
template <typename T1, typename T2>
__global__ void CudaDataTypeConvertKernel(void *input_data, void *output_data, int lenth) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= lenth) return;
    static_cast<T2 *>(output_data)[id] = float(static_cast<T1 *>(input_data)[id]);
}

template <DataType input_data_type, DataType output_data_type>
void CudaDataTypeConvert(void *input_data, void *output_data, int lenth, cudaStream_t stream) {
    dim3 block(1024);
    dim3 grid((lenth + block.x - 1) / block.x);
    CudaDataTypeConvertKernel<typename DataTypeTraits<input_data_type>::Type,
                              typename DataTypeTraits<output_data_type>::Type>
        <<<grid, block, 0, stream>>>(input_data, output_data, lenth);
}

template <DataType input_data_type, DataType output_data_type>
void DataTypeConvert(void *input_data, void *output_data, int lenth) {
    CudaDataTypeConvert<input_data_type, output_data_type>(input_data, output_data, lenth, 0);
    cudaStreamSynchronize(0);
    CUDAOP_CHECK_CUDA_SATUS(cudaGetLastError());
}

template <DataType input_data_type, DataType output_data_type>
void DataTypeConvertAsync(void *input_data, void *output_data, int lenth, cudaStream_t stream) {
    CudaDataTypeConvert<input_data_type, output_data_type>(input_data, output_data, lenth, stream);
}
}  // namespace cudaop
}  // namespace smartmore

#endif
