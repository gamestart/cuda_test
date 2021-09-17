/*******************************************************************************
 *  FILENAME:      test_reduce.cu
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Thursday May 13th 2021
 *
 *  LAST MODIFIED: Thursday, May 20th 2021, 5:10:55 pm
 *
 *  CONTACT:       xiaofei.wang@smartmore.com
 *******************************************************************************/
#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>
#include <cudaop/cudaop.h>
#include <macro.h>
#include <utils.h>

#include <catch2/catch.hpp>
#include <opencv2/opencv.hpp>

TEST_CASE("Reduce Sum Float32", "[reduce]") {
    const int src_len = 10000;
    std::vector<float> input_data(src_len);
    smartmore::RandomFloatVector(input_data);

    void *input_device = nullptr, *output_device = nullptr;
    float output_actual = 0.0, output_expect = 0.0;

    CUDA_CHECK(cudaMalloc(&input_device, input_data.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output_device, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(input_device, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(output_device, &output_actual, sizeof(float), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    Reduce<DataType::kFloat32, ReduceType::kSum>(input_device, src_len, output_device);
    CUDA_CHECK(cudaMemcpy(&output_actual, output_device, sizeof(float), cudaMemcpyDeviceToHost));

    output_expect = std::accumulate(input_data.begin(), input_data.end(), 0.f, std::plus<float>());

    float diff = fabs(output_expect - output_actual);

    // relative accuracy
    REQUIRE(diff / output_expect < 0.001);

    CUDA_CHECK_AND_FREE(input_device);
    CUDA_CHECK_AND_FREE(output_device);
}

TEST_CASE("Reduce Max Float32", "[reduce]") {
    const int src_len = 10000;
    std::vector<float> input_data(src_len);
    smartmore::RandomFloatVector(input_data);

    void *input_device = nullptr, *output_device = nullptr;
    float output_actual = 0.0, output_expect = 0.0;

    CUDA_CHECK(cudaMalloc(&input_device, input_data.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output_device, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(input_device, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(output_device, &output_actual, sizeof(float), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    Reduce<DataType::kFloat32, ReduceType::kMax>(input_device, src_len, output_device);
    CUDA_CHECK(cudaMemcpy(&output_actual, output_device, sizeof(float), cudaMemcpyDeviceToHost));

    for (auto f : input_data) {
        output_expect = output_expect > f ? output_expect : f;
    }
    float diff = fabs(output_expect - output_actual);

    REQUIRE(diff < 0.0001);

    CUDA_CHECK_AND_FREE(input_device);
    CUDA_CHECK_AND_FREE(output_device);
}

TEST_CASE("Reduce Min Float32", "[reduce]") {
    const int src_len = 10;
    std::vector<float> input_data(src_len);
    smartmore::RandomFloatVector(input_data);

    void *input_device = nullptr, *output_device = nullptr;
    float output_actual = 1.0, output_expect = 1.0;

    CUDA_CHECK(cudaMalloc(&input_device, input_data.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output_device, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(input_device, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(output_device, &output_actual, sizeof(float), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    Reduce<DataType::kFloat32, ReduceType::kMin>(input_device, src_len, output_device);
    CUDA_CHECK(cudaMemcpy(&output_actual, output_device, sizeof(float), cudaMemcpyDeviceToHost));

    for (auto f : input_data) {
        output_expect = output_expect > f ? f : output_expect;
    }
    float diff = fabs(output_expect - output_actual);

    REQUIRE(diff < 0.0001);

    CUDA_CHECK_AND_FREE(input_device);
    CUDA_CHECK_AND_FREE(output_device);
}

TEST_CASE("Reduce Min Int8", "[reduce]") {
    const int src_len = 10;
    std::vector<float> input_data_f(src_len);
    std::vector<unsigned char> input_data(src_len);
    smartmore::RandomFloatVector(input_data_f);
    for (int i = 0; i < input_data_f.size(); i++) {
        input_data[i] = input_data_f[i] * 255.0f;
    }

    void *input_device = nullptr, *output_device = nullptr;
    unsigned char output_actual = 255, output_expect = 255;

    CUDA_CHECK(cudaMalloc(&input_device, input_data.size() * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&output_device, sizeof(unsigned char)));
    CUDA_CHECK(
        cudaMemcpy(input_device, input_data.data(), input_data.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(output_device, &output_actual, sizeof(unsigned char), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    Reduce<DataType::kInt8, ReduceType::kMin>(input_device, src_len, output_device);
    CUDA_CHECK(cudaMemcpy(&output_actual, output_device, sizeof(unsigned char), cudaMemcpyDeviceToHost));

    for (auto f : input_data) {
        output_expect = output_expect > f ? f : output_expect;
    }
    unsigned char diff = fabs(output_expect - output_actual);

    REQUIRE(diff < 1);

    CUDA_CHECK_AND_FREE(input_device);
    CUDA_CHECK_AND_FREE(output_device);
}

TEST_CASE("Reduce Max Int8", "[reduce]") {
    const int src_len = 10;
    std::vector<float> input_data_f(src_len);
    std::vector<unsigned char> input_data(src_len);
    smartmore::RandomFloatVector(input_data_f);
    for (int i = 0; i < input_data_f.size(); i++) {
        input_data[i] = input_data_f[i] * 255.0f;
    }

    void *input_device = nullptr, *output_device = nullptr;
    unsigned char output_actual = 0, output_expect = 0;

    CUDA_CHECK(cudaMalloc(&input_device, input_data.size() * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&output_device, sizeof(unsigned char)));
    CUDA_CHECK(
        cudaMemcpy(input_device, input_data.data(), input_data.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(output_device, &output_actual, sizeof(unsigned char), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    Reduce<DataType::kInt8, ReduceType::kMax>(input_device, src_len, output_device);

    CUDA_CHECK(cudaMemcpy(&output_actual, output_device, sizeof(unsigned char), cudaMemcpyDeviceToHost));

    for (auto f : input_data) {
        output_expect = output_expect < f ? f : output_expect;
    }
    unsigned char diff = fabs(output_expect - output_actual);

    REQUIRE(diff < 1);

    CUDA_CHECK_AND_FREE(input_device);
    CUDA_CHECK_AND_FREE(output_device);
}

#if __CUDA_ARCH__ >= 700
TEST_CASE("Reduce Sum Half", "[reduce]") {
    const int src_len = 10;
    std::vector<float> input_data(src_len);
    smartmore::RandomFloatVector(input_data);

    void *input_float = nullptr, *input_half = nullptr, *output_half = nullptr, *output_float = nullptr;
    float output_actual = 0.0, output_expect = 0.0;

    CUDA_CHECK(cudaMalloc(&input_float, input_data.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&input_half, input_data.size() * sizeof(float) / 2));
    CUDA_CHECK(cudaMalloc(&output_half, sizeof(float) / 2));
    CUDA_CHECK(cudaMalloc(&output_float, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(input_float, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(output_float, &output_actual, sizeof(float), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    DataTypeConvert<DataType::kFloat32, DataType::kHalf>(input_float, input_half, input_data.size());
    DataTypeConvert<DataType::kFloat32, DataType::kHalf>(output_float, output_half, input_data.size());
    Reduce<DataType::kHalf, ReduceType::kSum>(input_half, src_len, output_half);
    DataTypeConvert<DataType::kHalf, DataType::kFloat32>(output_half, output_float, 1);
    CUDA_CHECK(cudaMemcpy(&output_actual, output_float, sizeof(float), cudaMemcpyDeviceToHost));

    for (auto f : input_data) {
        output_expect += f;
    }

    float diff = fabs(output_expect - output_actual);
    REQUIRE(diff < 0.01);

    CUDA_CHECK_AND_FREE(input_float);
    CUDA_CHECK_AND_FREE(input_half);
    CUDA_CHECK_AND_FREE(output_half);
    CUDA_CHECK_AND_FREE(output_float);
}

TEST_CASE("Reduce Min Half", "[reduce]") {
    const int src_len = 10;
    std::vector<float> input_data(src_len);
    smartmore::RandomFloatVector(input_data);

    void *input_float = nullptr, *input_half = nullptr, *output_half = nullptr, *output_float = nullptr;
    float output_actual = 1.0, output_expect = 1.0;

    CUDA_CHECK(cudaMalloc(&input_float, input_data.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&input_half, input_data.size() * sizeof(float) / 2));
    CUDA_CHECK(cudaMalloc(&output_half, sizeof(float) / 2));
    CUDA_CHECK(cudaMalloc(&output_float, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(input_float, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(output_float, &output_actual, sizeof(float), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    DataTypeConvert<DataType::kFloat32, DataType::kHalf>(input_float, input_half, input_data.size());
    DataTypeConvert<DataType::kFloat32, DataType::kHalf>(output_float, output_half, input_data.size());
    Reduce<DataType::kHalf, ReduceType::kMin>(input_half, src_len, output_half);
    DataTypeConvert<DataType::kHalf, DataType::kFloat32>(output_half, output_float, 1);
    CUDA_CHECK(cudaMemcpy(&output_actual, output_float, sizeof(float), cudaMemcpyDeviceToHost));

    for (auto f : input_data) {
        output_expect = output_expect > f ? f : output_expect;
    }

    float diff = fabs(output_expect - output_actual);
    REQUIRE(diff < 0.001);

    CUDA_CHECK_AND_FREE(input_float);
    CUDA_CHECK_AND_FREE(input_half);
    CUDA_CHECK_AND_FREE(output_half);
    CUDA_CHECK_AND_FREE(output_float);
}

TEST_CASE("Reduce Max Half", "[reduce]") {
    const int src_len = 10;
    std::vector<float> input_data(src_len);
    smartmore::RandomFloatVector(input_data);

    void *input_float = nullptr, *input_half = nullptr, *output_half = nullptr, *output_float = nullptr;
    float output_actual = 0.0, output_expect = 0.0;

    CUDA_CHECK(cudaMalloc(&input_float, input_data.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&input_half, input_data.size() * sizeof(float) / 2));
    CUDA_CHECK(cudaMalloc(&output_half, sizeof(float) / 2));
    CUDA_CHECK(cudaMalloc(&output_float, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(input_float, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(output_float, &output_actual, sizeof(float), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    DataTypeConvert<DataType::kFloat32, DataType::kHalf>(input_float, input_half, input_data.size());
    DataTypeConvert<DataType::kFloat32, DataType::kHalf>(output_float, output_half, input_data.size());
    Reduce<DataType::kHalf, ReduceType::kMax>(input_half, src_len, output_half);
    DataTypeConvert<DataType::kHalf, DataType::kFloat32>(output_half, output_float, 1);
    CUDA_CHECK(cudaMemcpy(&output_actual, output_float, sizeof(float), cudaMemcpyDeviceToHost));

    for (auto f : input_data) {
        output_expect = output_expect < f ? f : output_expect;
    }

    float diff = fabs(output_expect - output_actual);
    REQUIRE(diff < 0.001);

    CUDA_CHECK_AND_FREE(input_float);
    CUDA_CHECK_AND_FREE(input_half);
    CUDA_CHECK_AND_FREE(output_half);
    CUDA_CHECK_AND_FREE(output_float);
}
#endif