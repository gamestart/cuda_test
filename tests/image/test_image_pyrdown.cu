/*******************************************************************************
 *  FILENAME:      test_image_pyrdown.cu
 *
 *  AUTHORS:       Sun Yucheng    START DATE: Tuesday September 14th 2021
 *
 *  LAST MODIFIED: Friday, September 17th 2021, 10:11:13 am
 *
 *  CONTACT:       yucheng.sun@smartmore.com
 *******************************************************************************/

#define CATCH_CONFIG_MAIN
#include <cudaop/cudaop.h>

#include <catch2/catch.hpp>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "macro.h"
#include "utils.h"

TEST_CASE("PyrDown", "[pyrdown]") {
    using namespace smartmore::cudaop;

    const int img_h = 3100;
    const int img_w = 2600;
    constexpr int dst_img_h = (img_h + 1) / 2;
    constexpr int dst_img_w = (img_w + 1) / 2;

    std::vector<unsigned char> pixels(img_h * img_w);
    smartmore::RandomInt8Vector(pixels);

    cv::Mat mat(img_h, img_w, CV_8UC1, pixels.data());
    unsigned char *src = nullptr;
    unsigned char *dst = nullptr;

    CUDA_CHECK(cudaMalloc(&src, img_h * img_w));
    CUDA_CHECK(cudaMalloc(&dst, dst_img_h * dst_img_w));
    CUDA_CHECK(cudaMemcpy(src, mat.data, img_h * img_w, cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("Pyrdown-kInt8-cuda: ");
        ImagePyrDown<ImageType::kGRAY, DataType::kInt8, BorderType::kReplicate>(src, dst, img_h, img_w);
    }

    cv::Mat out(dst_img_h, dst_img_w, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(out.data, dst, dst_img_h * dst_img_w, cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_FREE(src);
    CUDA_CHECK_AND_FREE(dst);

    cv::Mat reference;
    for (int i = 0; i < 10; i++) {
        smartmore::Clock clk("Pyrdown-kInt8-OpenCV: ");
        cv::pyrDown(mat, reference, cv::Size(dst_img_w, dst_img_h), cv::BORDER_REPLICATE);
    }

    cv::Mat diff = reference - out;
    int max_diff = 0;
    for (int i = 0; i < dst_img_h * dst_img_w; i++) {
        max_diff = max_diff < diff.data[i] ? diff.data[i] : max_diff;
    }

    REQUIRE(max_diff <= 1);
}
