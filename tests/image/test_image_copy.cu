/*******************************************************************************
 *  FILENAME:      test_image_copy.cu
 *
 *  AUTHORS:       Liang Jia    START DATE: Wednesday July 21st 2021
 *
 *  LAST MODIFIED: Friday, August 6th 2021, 10:29:01 am
 *
 *  CONTACT:       jia.liang@smartmore.com
 *******************************************************************************/

#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>
#include <cudaop/cudaop.h>
#include <macro.h>
#include <utils.h>

#include <catch2/catch.hpp>
#include <opencv2/opencv.hpp>

TEST_CASE("ImageCopy", "[image_copy]") {
    const int src_h = 1080, src_w = 1920;
    const int dst_h = 720, dst_w = 1280;
    auto src_rect = cv::Rect(0, 0, 640, 480);
    auto dst_rect = cv::Rect(100, 100, 640, 480);

    void *src_device, *dst_device;

    cv::Mat src_f = cv::Mat(src_h, src_w, CV_32FC3, cv::Scalar(1, 1, 1));
    cv::Mat dst_f = cv::Mat::zeros(dst_h, dst_w, CV_32FC3);
    cv::Mat actual(dst_h, dst_w, CV_32FC3);
    cv::Mat expect = dst_f.clone();
    CUDA_CHECK(cudaMalloc(&src_device, sizeof(float) * src_f.cols * src_f.rows * src_f.channels()));
    CUDA_CHECK(cudaMalloc(&dst_device, sizeof(float) * dst_f.cols * dst_f.rows * dst_f.channels()));

    CUDA_CHECK(cudaMemcpy(src_device, src_f.data, sizeof(float) * src_f.cols * src_f.rows * src_f.channels(),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dst_device, dst_f.data, sizeof(float) * dst_f.cols * dst_f.rows * dst_f.channels(),
                          cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    ImageCopy<ImageType::kBGR_HWC, DataType::kFloat32>(
        src_device, Size{src_w, src_h}, Rect{Point{src_rect.x, src_rect.y}, Size{src_rect.width, src_rect.height}},
        dst_device, Size{dst_w, dst_h}, Point{dst_rect.x, dst_rect.y});

    CUDA_CHECK(cudaMemcpy(actual.data, dst_device, sizeof(float) * actual.cols * actual.rows * actual.channels(),
                          cudaMemcpyDeviceToHost));

    src_f(src_rect).copyTo(expect(dst_rect));

    float max_diff = smartmore::CVMatMaxDiff(actual, expect);

    REQUIRE(max_diff < 0.0001);
    CUDA_CHECK_AND_FREE(src_device);
    CUDA_CHECK_AND_FREE(dst_device);
}