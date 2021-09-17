/*******************************************************************************
 *  FILENAME:      test_image_resize.cu
 *
 *  AUTHORS:       Wang Xiaofei    START DATE: Tuesday April 27th 2021
 *
 *  LAST MODIFIED: Wednesday, May 19th 2021, 2:49:55 pm
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

TEST_CASE("ImageResize Bilinear", "[image_resize]") {
    const int src_h = 560, src_w = 675, src_c = 3;
    const int dst_h = 1024, dst_w = 400;
    void *src_device, *dst_device;

    std::vector<float> input_data(src_h * src_w * src_c);
    smartmore::RandomFloatVector(input_data);
    cv::Mat src_f(src_h, src_w, CV_32FC3, input_data.data());
    cv::Mat actual(dst_h, dst_w, CV_32FC3), expect(dst_h, dst_w, CV_32FC3);

    CUDA_CHECK(cudaMalloc(&src_device, sizeof(float) * src_f.cols * src_f.rows * src_f.channels()));
    CUDA_CHECK(cudaMalloc(&dst_device, sizeof(float) * expect.cols * expect.rows * expect.channels()));
    CUDA_CHECK(cudaMemcpy(src_device, src_f.data, sizeof(float) * src_f.cols * src_f.rows * src_f.channels(),
                          cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    ImageResize<ImageType::kBGR_HWC, DataType::kFloat32, DataType::kFloat32, ResizeScaleType::kStretch,
                ResizeAlgoType::kBilinear>(src_device, dst_device, src_f.rows, src_f.cols, expect.rows, expect.cols);

    CUDA_CHECK(cudaMemcpy(actual.data, dst_device, sizeof(float) * actual.cols * actual.rows * actual.channels(),
                          cudaMemcpyDeviceToHost));
    cv::resize(src_f, expect, expect.size(), 0, 0, cv::INTER_LINEAR);

    float max_diff = smartmore::CVMatMaxDiff(actual, expect);

    REQUIRE(max_diff < 0.0001);
    CUDA_CHECK_AND_FREE(src_device);
    CUDA_CHECK_AND_FREE(dst_device);
}

TEST_CASE("ImageResize Nearest", "[image_resize]") {
    const int src_h = 560, src_w = 675, src_c = 3;
    const int dst_h = 1024, dst_w = 400;
    void *src_device, *dst_device;

    std::vector<float> input_data(src_h * src_w * src_c);
    smartmore::RandomFloatVector(input_data);
    cv::Mat src_f(src_h, src_w, CV_32FC3, input_data.data());
    cv::Mat actual(dst_h, dst_w, CV_32FC3), expect(dst_h, dst_w, CV_32FC3);

    CUDA_CHECK(cudaMalloc(&src_device, sizeof(float) * src_f.cols * src_f.rows * src_f.channels()));
    CUDA_CHECK(cudaMalloc(&dst_device, sizeof(float) * expect.cols * expect.rows * expect.channels()));
    CUDA_CHECK(cudaMemcpy(src_device, src_f.data, sizeof(float) * src_f.cols * src_f.rows * src_f.channels(),
                          cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    ImageResize<ImageType::kBGR_HWC, DataType::kFloat32, DataType::kFloat32, ResizeScaleType::kStretch,
                ResizeAlgoType::kNearest>(src_device, dst_device, src_f.rows, src_f.cols, expect.rows, expect.cols);

    CUDA_CHECK(cudaMemcpy(actual.data, dst_device, sizeof(float) * actual.cols * actual.rows * actual.channels(),
                          cudaMemcpyDeviceToHost));
    cv::resize(src_f, expect, expect.size(), 0, 0, cv::INTER_NEAREST);

    float max_diff = smartmore::CVMatMaxDiff(actual, expect);

    REQUIRE(max_diff < 0.0001);
    CUDA_CHECK_AND_FREE(src_device);
    CUDA_CHECK_AND_FREE(dst_device);
}

TEST_CASE("ImageResize Bicubic", "[image_resize]") {
    const int src_h = 560, src_w = 675, src_c = 3;
    const int dst_h = 1024, dst_w = 400;
    void *src_device, *dst_device;

    std::vector<float> input_data(src_h * src_w * src_c);
    smartmore::RandomFloatVector(input_data);
    cv::Mat src_f(src_h, src_w, CV_32FC3, input_data.data());
    cv::Mat actual(dst_h, dst_w, CV_32FC3), expect(dst_h, dst_w, CV_32FC3);

    CUDA_CHECK(cudaMalloc(&src_device, sizeof(float) * src_f.cols * src_f.rows * src_f.channels()));
    CUDA_CHECK(cudaMalloc(&dst_device, sizeof(float) * expect.cols * expect.rows * expect.channels()));
    CUDA_CHECK(cudaMemcpy(src_device, src_f.data, sizeof(float) * src_f.cols * src_f.rows * src_f.channels(),
                          cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    ImageResize<ImageType::kBGR_HWC, DataType::kFloat32, DataType::kFloat32, ResizeScaleType::kStretch,
                ResizeAlgoType::kBicubic>(src_device, dst_device, src_f.rows, src_f.cols, expect.rows, expect.cols);

    CUDA_CHECK(cudaMemcpy(actual.data, dst_device, sizeof(float) * actual.cols * actual.rows * actual.channels(),
                          cudaMemcpyDeviceToHost));
    cv::resize(src_f, expect, expect.size(), 0, 0, cv::INTER_CUBIC);

    float max_diff = smartmore::CVMatMaxDiff(actual, expect);

    REQUIRE(max_diff < 0.0001);
    CUDA_CHECK_AND_FREE(src_device);
    CUDA_CHECK_AND_FREE(dst_device);
}
