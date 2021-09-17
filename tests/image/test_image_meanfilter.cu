/*******************************************************************************
 *  FILENAME:      test_image_meanfilter.cu
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Friday May 28th 2021
 *
 *  LAST MODIFIED: Tuesday, June 8th 2021, 5:17:14 pm
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

TEST_CASE("Meanfilter", "[meanfilter]") {
    int height = 2000, width = 2000;
    const int kernel_h = 7, kernel_w = 7;

    // mean (x-kx/2,x+(kx-1)/2)
    std::vector<unsigned char> input_gray(height * width);
    smartmore::RandomInt8Vector(input_gray);

    cv::Mat src_gray(height, width, CV_8UC1, input_gray.data()), expect_mat;
    cv::blur(src_gray, expect_mat, cv::Size(kernel_w, kernel_h));
    std::vector<unsigned char> expect(expect_mat.cols * expect_mat.rows);
    memcpy(&expect[0], expect_mat.data, expect_mat.cols * expect_mat.rows * sizeof(unsigned char));

    std::vector<unsigned char> actual(expect.size());
    void *input_device = nullptr, *output_device = nullptr;
    CUDA_CHECK(cudaMalloc(&input_device, input_gray.size()));
    CUDA_CHECK(cudaMalloc(&output_device, input_gray.size()));
    CUDA_CHECK(cudaMemcpy(input_device, input_gray.data(), input_gray.size(), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("MeanFilter");
        MeanFilter<ImageType::kGRAY, DataType::kInt8, BorderType::kReflect, kernel_h, kernel_w>(
            input_device, output_device, height, width);
    }

    CUDA_CHECK(cudaMemcpy(&actual[0], output_device, actual.size(), cudaMemcpyDeviceToHost));

    for (int i = 0; i < actual.size(); i++) {
        REQUIRE(fabs(actual[i] - expect[i]) <= 1.0f);
    }

    CUDA_CHECK_AND_FREE(input_device);
    CUDA_CHECK_AND_FREE(output_device);
}