/******************************************************************************
 * FILENAME:      test_image_gaussianfilter.cu
 *
 * AUTHORS:       Yutong Huang
 *
 * LAST MODIFIED: Thu 10 Jun 2021 10:29:37 AM CST
 *
 * CONTACT:       yutong.huang@smartmore.com
 ******************************************************************************/

#define CATCH_CONFIG_MAIN
#include <cudaop/cudaop.h>

#include <catch2/catch.hpp>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "macro.h"
#include "utils.h"

TEST_CASE("GaussianFilter", "[gaussianfilter]") {
    using namespace smartmore::cudaop;

    const int img_h = 2000;
    const int img_w = 2000;
    const int kernel_h = 11;
    const int kernel_w = 11;
    const float sigma_x = 1.f;
    const float sigma_y = 1.f;

    std::vector<unsigned char> pixels(img_h * img_w);
    smartmore::RandomInt8Vector(pixels);

    cv::Mat mat(img_h, img_w, CV_8UC1, pixels.data());
    unsigned char *src = nullptr;
    unsigned char *dst = nullptr;

    CUDA_CHECK(cudaMalloc(&src, img_h * img_w));
    CUDA_CHECK(cudaMalloc(&dst, img_h * img_w));
    CUDA_CHECK(cudaMemcpy(src, mat.data, img_h * img_w, cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("GaussianFilter");
        ImageGaussianFilter<ImageType::kGRAY, DataType::kInt8, BorderType::kReplicate, kernel_h, kernel_w>(
            src, dst, img_h, img_w, sigma_x, sigma_y);
    }

    cv::Mat out(img_h, img_w, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(out.data, dst, img_h * img_w, cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_FREE(src);
    CUDA_CHECK_AND_FREE(dst);

    cv::Mat reference;
    cv::GaussianBlur(mat, reference, {kernel_h, kernel_w}, sigma_x, sigma_y, cv::BORDER_REPLICATE);

    cv::Mat diff = reference - out;
    int max_diff = 0;
    for (int i = 0; i < img_h * img_w; i++) {
        max_diff = max_diff < diff.data[i] ? diff.data[i] : max_diff;
    }

    REQUIRE(max_diff <= 1);
}
