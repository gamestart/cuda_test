/*******************************************************************************
 *  FILENAME:      imgmedianfilter.h
 *
 *  AUTHORS:       Wang Shengxiang    START DATE: Saturday May 29th 2021
 *
 *  LAST MODIFIED: Wednesday, June 2nd 2021, 12:50:33 pm
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

TEST_CASE("Medianfilter-int8", "[medianfilter]") {
    int height = 4000, width = 2000;
    const int ksize = 7;

    // median [x-ksize/2, x+ksize/2]
    std::vector<unsigned char> input_gray(height * width);
    smartmore::RandomInt8Vector(input_gray);

    cv::Mat src_gray(height, width, CV_8UC1, input_gray.data()), expect_mat;

    {
        smartmore::Clock clk("MedianFilter-int8-opencv: ");
        cv::medianBlur(src_gray, expect_mat, ksize);
    }

    std::vector<unsigned char> expect(expect_mat.cols * expect_mat.rows);
    memcpy(&expect[0], expect_mat.data, expect_mat.cols * expect_mat.rows * sizeof(char));

    std::vector<unsigned char> actual(expect.size());
    void *input_device = nullptr, *output_device = nullptr;
    CUDA_CHECK(cudaMalloc(&input_device, input_gray.size()));
    CUDA_CHECK(cudaMalloc(&output_device, input_gray.size()));
    CUDA_CHECK(cudaMemcpy(input_device, input_gray.data(), input_gray.size(), cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("MedianFilter-int8-cuda: ");
        MedianFilter<ImageType::kGRAY, DataType::kInt8, BorderType::kReplicate, ksize>(input_device, output_device,
                                                                                       height, width);
    }

    CUDA_CHECK(cudaMemcpy(&actual[0], output_device, actual.size(), cudaMemcpyDeviceToHost));

    for (int i = 0; i < actual.size(); i++) {
        REQUIRE(fabs(actual[i] - expect[i]) <= 1.0f);
    }

    CUDA_CHECK_AND_FREE(input_device);
    CUDA_CHECK_AND_FREE(output_device);
}
