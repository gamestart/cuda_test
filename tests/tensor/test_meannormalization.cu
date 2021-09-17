/*******************************************************************************
 *  FILENAME:      test_meannormalization.cu
 *
 *  AUTHORS:       Hou Yue    START DATE: Wednesday July 21st 2021
 *
 *  LAST MODIFIED: Monday, July 26th 2021, 2:02:25 pm
 *
 *  CONTACT:       yue.hou@smartmore.com
 *******************************************************************************/

#define CATCH_CONFIG_MAIN

#include <cuda_runtime.h>
#include <cudaop/cudaop.h>
#include <macro.h>
#include <stdio.h>
#include <utils.h>

#include <catch2/catch.hpp>

TEST_CASE("MeanNormalization Float32", "[meannormalization]") {
    const int src_len = 10;
    std::vector<float> input_data(src_len);
    smartmore::RandomFloatVector(input_data);

    void *input_device = nullptr, *output_device = nullptr;
    std::vector<float> output_actual(src_len), output_expect(src_len);

    CUDA_CHECK(cudaMalloc(&input_device, input_data.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output_device, input_data.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(input_device, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    float mean = 0.f, variance = 0.f;
    // mean
    mean = std::accumulate(input_data.begin(), input_data.end(), 0.f, std::plus<float>()) / src_len;
    // variance
    for (size_t i = 0; i < input_data.size(); ++i) {
        variance += pow(input_data[i] - mean, 2) / src_len;
    }

    MeanNormalization<DataType::kFloat32>(input_device, output_device, src_len, mean, variance);
    // actual
    CUDA_CHECK(cudaMemcpy(&output_actual[0], output_device, src_len * sizeof(float), cudaMemcpyDeviceToHost));

    // expect
    for (int i = 0; i < input_data.size(); i++) {
        output_expect[i] = (input_data[i] - mean) / variance;

        float diff = fabs(output_actual[i] - output_expect[i]);
        REQUIRE(diff / output_expect[i] < 0.001);
    }

    CUDA_CHECK_AND_FREE(input_device);
    CUDA_CHECK_AND_FREE(output_device);
}

#if __CUDA_ARCH__ >= 700
TEST_CASE("MeanNormalization Half", "[meannormalization]") {
    const int src_len = 10;
    std::vector<float> input_data(src_len);
    smartmore::RandomFloatVector(input_data);

    void *input_float = nullptr, *input_half = nullptr, *output_float = nullptr, *output_half = nullptr;

    std::vector<float> output_actual(src_len), output_expect(src_len);

    CUDA_CHECK(cudaMalloc(&input_float, input_data.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&input_half, input_data.size() * sizeof(float) / 2));
    CUDA_CHECK(cudaMalloc(&output_float, input_data.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output_half, input_data.size() * sizeof(float) / 2));
    CUDA_CHECK(cudaMemcpy(input_float, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    DataTypeConvert<DataType::kFloat32, DataType::kHalf>(input_float, input_half, input_data.size());
    DataTypeConvert<DataType::kFloat32, DataType::kHalf>(output_float, output_half, input_data.size());

    float mean = 0.f, variance = 0.f;
    // mean
    mean = std::accumulate(input_data.begin(), input_data.end(), 0.f, std::plus<float>()) / src_len;
    // variance
    for (size_t i = 0; i < input_data.size(); ++i) {
        variance += pow(input_data[i] - mean, 2) / src_len;
    }

    MeanNormalization<DataType::kHalf>(input_half, output_half, src_len, mean, variance);

    DataTypeConvert<DataType::kHalf, DataType::kFloat32>(output_half, output_float, input_data.size());
    CUDA_CHECK(cudaMemcpy(&output_actual[0], output_float, src_len * sizeof(float), cudaMemcpyDeviceToHost));

    // expect
    for (int i = 0; i < input_data.size(); i++) {
        output_expect[i] = (input_data[i] - mean) / variance;

        float diff = fabs(output_actual[i] - output_expect[i]);
        REQUIRE(diff / output_expect[i] < 0.01);
    }

    CUDA_CHECK_AND_FREE(input_float);
    CUDA_CHECK_AND_FREE(input_half);
    CUDA_CHECK_AND_FREE(output_float);
    CUDA_CHECK_AND_FREE(output_half);
}
#endif