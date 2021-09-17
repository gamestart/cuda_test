/*******************************************************************************
 *  FILENAME:      test_clamp.cu
 *
 *  AUTHORS:       Wang Shengxiang    START DATE: Tuesday September 7th 2021
 *
 *  LAST MODIFIED: Tuesday, September 7th 2021, 5:26:33 pm
 *
 *  CONTACT:       shengxiang.wang@smartmore.com
 *******************************************************************************/
#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>
#include <cudaop/cudaop.h>
#include <macro.h>
#include <utils.h>

#include <catch2/catch.hpp>
#include <opencv2/opencv.hpp>

TEST_CASE("Clamp [0,100] INT8", "[clamp]") {
    // Clamp [50,100] INT8
    void *src_dev = nullptr;
    // 3 * 4 * 3 image, set some number larger than 100
    std::vector<unsigned char> input{105, 150, 92,  205, 124, 230, 72,  162, 246, 112, 152, 168,
                                     130, 238, 189, 74,  154, 68,  166, 58,  29,  155, 255, 187,
                                     220, 213, 132, 80,  212, 216, 33,  242, 244, 6,   139, 170};
    CUDA_CHECK(cudaMalloc(&src_dev, input.size() * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(src_dev, input.data(), input.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));

    smartmore::cudaop::Clamp<smartmore::cudaop::DataType::kInt8>(src_dev, input.size(), 50, 100);
    // convert in cpu
    std::vector<unsigned char> expect(input.size()), actual(input.size());

    CUDA_CHECK(cudaMemcpy(&actual[0], src_dev, actual.size() * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    for (int i = 0; i < input.size(); i++) {
        expect[i] = (input[i] > 100) ? 100 : ((input[i] < 50) ? 50 : input[i]);
    }

    // compare cuda with opencv
    for (int i = 0; i < input.size(); i++) {
        REQUIRE(fabs(actual[i] - expect[i]) < 1);
    }

    CUDA_CHECK_AND_FREE(src_dev);
}

TEST_CASE("Clamp [0,1] FLOAT32", "[clamp]") {
    // Clamp [0,1] FLOAT32
    void *src_dev = nullptr;
    // 2 * 4 image, set some number larger than 100
    std::vector<float> input{0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5};

    CUDA_CHECK(cudaMalloc(&src_dev, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(src_dev, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    smartmore::cudaop::Clamp<smartmore::cudaop::DataType::kFloat32>(src_dev, input.size(), 0, 1);

    std::vector<float> result_cuda(input.size());

    CUDA_CHECK(cudaMemcpy(result_cuda.data(), src_dev, input.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // convert in cpu
    std::vector<float> result_cpu(input.size());
    for (int i = 0; i < input.size(); i++) {
        result_cpu[i] = (input[i] > 1) ? 1 : ((input[i] < 0) ? 0 : input[i]);
    }

    // compare cuda with opencv
    for (int i = 0; i < input.size(); i++) {
        float result = result_cuda[i];
        float expect = result_cpu[i];
        REQUIRE(fabs(result - expect) < 0.01);
    }

    CUDA_CHECK_AND_FREE(src_dev);
}

TEST_CASE("Clamp [0,1] half", "[clamp]") {
    void *src_float = nullptr, *src_half = nullptr, *dst_float = nullptr;

    std::vector<float> input{0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5};

    CUDA_CHECK(cudaMalloc(&src_float, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&src_half, input.size() * sizeof(float) / 2));
    CUDA_CHECK(cudaMalloc(&dst_float, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(src_float, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    DataTypeConvert<DataType::kFloat32, DataType::kHalf>(src_float, src_half, input.size());
    Clamp<DataType::kHalf>(src_half, input.size(), 0, 1);
    DataTypeConvert<DataType::kHalf, DataType::kFloat32>(src_half, dst_float, input.size());

    std::vector<float> expect(input.size()), actual(input.size());
    CUDA_CHECK(cudaMemcpy(&actual[0], dst_float, input.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < input.size(); i++) {
        expect[i] = (input[i] > 1) ? 1 : ((input[i] < 0) ? 0 : input[i]);
    }

    for (int i = 0; i < input.size(); i++) {
        REQUIRE(fabs(actual[i] - expect[i]) < 0.01);
    }

    CUDA_CHECK_AND_FREE(src_float);
    CUDA_CHECK_AND_FREE(src_half);
    CUDA_CHECK_AND_FREE(dst_float);
}