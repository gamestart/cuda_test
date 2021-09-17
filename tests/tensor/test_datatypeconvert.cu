/*******************************************************************************
 *  FILENAME:      test_datatypeconvert.cu
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Wednesday May 19th 2021
 *
 *  LAST MODIFIED: Wednesday, May 19th 2021, 9:31:50 pm
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

TEST_CASE("int8 to float", "[DataTypeConvert]") {
    void *src = nullptr, *dst = nullptr;

    std::vector<unsigned char> input{1, 4, 3, 5, 6};

    CUDA_CHECK(cudaMalloc(&src, input.size() * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&dst, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(src, input.data(), input.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    DataTypeConvert<DataType::kInt8, DataType::kFloat32>(src, dst, input.size());

    std::vector<float> actual(input.size()), expect(input.size());

    for (int i = 0; i < input.size(); i++) {
        expect[i] = input[i];
    }

    CUDA_CHECK(cudaMemcpy(&actual[0], dst, sizeof(float) * actual.size(), cudaMemcpyDeviceToHost));

    for (int i = 0; i < actual.size(); i++) {
        REQUIRE(actual[i] == expect[i]);
    }
}

TEST_CASE("float to int8", "[DataTypeConvert]") {
    void *src = nullptr, *dst = nullptr;

    std::vector<float> input{1, 4, 3, 5, 6};

    CUDA_CHECK(cudaMalloc(&src, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst, input.size() * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(src, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    DataTypeConvert<DataType::kFloat32, DataType::kInt8>(src, dst, input.size());

    std::vector<unsigned char> actual(input.size()), expect(input.size());

    for (int i = 0; i < input.size(); i++) {
        expect[i] = input[i];
    }

    CUDA_CHECK(cudaMemcpy(&actual[0], dst, sizeof(unsigned char) * actual.size(), cudaMemcpyDeviceToHost));

    for (int i = 0; i < actual.size(); i++) {
        REQUIRE(actual[i] == expect[i]);
    }
}

TEST_CASE("int8 to half and half to int8", "[DataTypeConvert]") {
    void *src = nullptr, *dst_half = nullptr, *dst_int8;

    std::vector<unsigned char> input{1, 4, 3, 5, 6};

    CUDA_CHECK(cudaMalloc(&src, input.size() * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&dst_half, input.size() * sizeof(float) / 2));
    CUDA_CHECK(cudaMalloc(&dst_int8, input.size() * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(src, input.data(), input.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    DataTypeConvert<DataType::kInt8, DataType::kHalf>(src, dst_half, input.size());
    DataTypeConvert<DataType::kHalf, DataType::kInt8>(dst_half, dst_int8, input.size());

    std::vector<unsigned char> actual(input.size()), expect(input.size());

    for (int i = 0; i < input.size(); i++) {
        expect[i] = input[i];
    }

    CUDA_CHECK(cudaMemcpy(&actual[0], dst_int8, sizeof(unsigned char) * actual.size(), cudaMemcpyDeviceToHost));

    for (int i = 0; i < actual.size(); i++) {
        REQUIRE(actual[i] == expect[i]);
    }
}

TEST_CASE("float to half and half to float", "[DataTypeConvert]") {
    void *src = nullptr, *dst_half = nullptr, *dst_int8;

    std::vector<float> input{1, 4, 3, 5, 6};

    CUDA_CHECK(cudaMalloc(&src, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst_half, input.size() * sizeof(float) / 2));
    CUDA_CHECK(cudaMalloc(&dst_int8, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(src, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    DataTypeConvert<DataType::kFloat32, DataType::kHalf>(src, dst_half, input.size());
    DataTypeConvert<DataType::kHalf, DataType::kFloat32>(dst_half, dst_int8, input.size());

    std::vector<float> actual(input.size()), expect(input.size());

    for (int i = 0; i < input.size(); i++) {
        expect[i] = input[i];
    }

    CUDA_CHECK(cudaMemcpy(&actual[0], dst_int8, sizeof(float) * actual.size(), cudaMemcpyDeviceToHost));

    for (int i = 0; i < actual.size(); i++) {
        REQUIRE(actual[i] == expect[i]);
    }
}
