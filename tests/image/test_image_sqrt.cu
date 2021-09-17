/*******************************************************************************
 *  FILENAME:      test_image_sqrt.cu
 *
 *  AUTHORS:       Chen Fting    START DATE: Thursday August 5th 2021
 *
 *  LAST MODIFIED: Friday, August 6th 2021, 8:16:31 pm
 *
 *  CONTACT:       fting.chen@smartmore.com
 *******************************************************************************/
#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>
#include <cudaop/cudaop.h>
#include <macro.h>
#include <utils.h>

#include <catch2/catch.hpp>
#include <opencv2/opencv.hpp>

TEST_CASE("Sqrt", "[sqrt]") {
    int height = 2000, width = 2000;

    std::vector<float> input_data(height * width);
    smartmore::RandomFloatVector(input_data);
    cv::Mat src_f(height, width, CV_32FC1, input_data.data()), expect_mat;

    cv::sqrt(src_f, expect_mat);
    std::vector<float> expect(expect_mat.cols * expect_mat.rows);
    memcpy(&expect[0], expect_mat.data, expect_mat.cols * expect_mat.rows * sizeof(float));

    std::vector<float> actual(input_data.size());
    void *input_device = nullptr, *output_device = nullptr;
    CUDA_CHECK(cudaMalloc(&input_device, input_data.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output_device, input_data.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(input_device, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("Sqrt");
        Sqrt<DataType::kFloat32>(input_device, output_device, height * width);
    }

    CUDA_CHECK(cudaMemcpy(&actual[0], output_device, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < actual.size(); i++) {
        REQUIRE(fabs(actual[i] - expect[i]) <= 0.001f);
    }

    CUDA_CHECK_AND_FREE(input_device);
    CUDA_CHECK_AND_FREE(output_device);
}