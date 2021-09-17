/*******************************************************************************
 *  FILENAME:      test_image_addweighted.cu
 *
 *  AUTHORS:       Liang Jia    START DATE: Saturday August 14th 2021
 *
 *  LAST MODIFIED: Saturday, August 14th 2021, 3:26:16 pm
 *
 *  CONTACT:       jia.liang@smartmore.com
 *******************************************************************************/
#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>
#include <cudaop/cudaop.h>
#include <macro.h>
#include <utils.h>

#include <catch2/catch.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace smartmore::cudaop;

TEST_CASE("ImageAddWeighted", "[image_addweighted]") {
    const int height = 1080, width = 1920, channels = 3;
    const float alpha = 0.6f;
    const float beta = 1 - alpha;
    const float gamma = 0.0f;
    float *src1_device, *src2_device;
    std::vector<float> src1_data(height * width * channels);
    smartmore::RandomFloatVector(src1_data);

    std::vector<float> src2_data(height * width * channels);
    smartmore::RandomFloatVector(src2_data);

    cv::Mat src1_f = cv::Mat(height, width, CV_32FC3, src1_data.data());
    cv::Mat src2_f = cv::Mat(height, width, CV_32FC3, src2_data.data());

    cv::Mat actual_mat = src1_f.clone();
    cv::Mat expet_mat = src1_f.clone();
    size_t data_size = sizeof(float) * src1_f.cols * src1_f.rows * src1_f.channels();
    CUDA_CHECK(cudaMalloc(&src1_device, data_size));
    CUDA_CHECK(cudaMemcpy(src1_device, src1_f.data, data_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&src2_device, data_size));
    CUDA_CHECK(cudaMemcpy(src2_device, src2_f.data, data_size, cudaMemcpyHostToDevice));

    std::vector<Rect> sliced_rects;
    ImageAddWeighted<ImageType::kBGR_HWC, DataType::kFloat32>(src1_device, alpha, src2_device, beta, gamma,
                                                              Size{width, height}, src1_device);

    CUDA_CHECK(cudaMemcpy(actual_mat.data, src1_device, data_size, cudaMemcpyDeviceToHost));
    cv::addWeighted(src1_f, alpha, src2_f, beta, gamma, expet_mat);
    float max_diff = smartmore::CVMatMaxDiff(actual_mat, expet_mat);
    REQUIRE(max_diff < 0.0001);

    CUDA_CHECK_AND_FREE(src1_device);
    CUDA_CHECK_AND_FREE(src2_device);
}