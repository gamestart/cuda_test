/*******************************************************************************
 *  FILENAME:      test_image_ copymakeborder.cu
 *
 *  AUTHORS:       Hou Yue    START DATE: Thursday August 12th 2021
 *
 *  LAST MODIFIED: Saturday, August 14th 2021, 5:45:51 pm
 *
 *  CONTACT:       yue.hou@smartmore.com
 *******************************************************************************/

#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>
#include <cudaop/cudaop.h>
#include <macro.h>
#include <utils.h>

#include <catch2/catch.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

TEST_CASE("ImageCopyMakeBorder to kGRAY", "[ImageCopyMakeBorder]") {
    const int height = 1080, width = 1920;
    int top = 100, bottom = 100, left = 100, right = 100;
    int out_w = width + left + right;
    int out_h = height + top + bottom;
    void *d_src, *d_dst;

    std::vector<uchar> input_gray(width * height * 3);
    smartmore::RandomInt8Vector(input_gray);

    cv::Mat src_f = cv::Mat(height, width, CV_8UC1, input_gray.data());
    cv::Mat dst_f = cv::Mat::zeros(out_h, out_w, CV_8UC1);
    cv::Mat actual(out_h, out_w, CV_8UC1);
    cv::Mat expect(out_h, out_w, CV_8UC1);

    CUDA_CHECK(cudaMalloc(&d_src, sizeof(unsigned char) * src_f.cols * src_f.rows * src_f.channels()));
    CUDA_CHECK(cudaMalloc(&d_dst, sizeof(unsigned char) * dst_f.cols * dst_f.rows * dst_f.channels()));
    CUDA_CHECK(cudaMemcpy(d_src, src_f.data, sizeof(unsigned char) * src_f.cols * src_f.rows * src_f.channels(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dst, dst_f.data, sizeof(unsigned char) * dst_f.cols * dst_f.rows * dst_f.channels(),
                          cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    SECTION("kBorder_Replicate to gray") {
        ImageCopyMakeBorder<DataType::kInt8, ImageType::kGRAY, CopyBorderType::kBorder_Replicate>(
            d_src, d_dst, height, width, top, bottom, left, right);
        cv::copyMakeBorder(src_f, expect, top, bottom, left, right, cv::BorderTypes::BORDER_REPLICATE);
    }
    SECTION("kBorder_Constant to gray") {
        ImageCopyMakeBorder<DataType::kInt8, ImageType::kGRAY, CopyBorderType::kBorder_Constant>(
            d_src, d_dst, height, width, top, bottom, left, right, 255);
        cv::copyMakeBorder(src_f, expect, top, bottom, left, right, cv::BorderTypes::BORDER_CONSTANT, 255);
    }
    SECTION("kBorder_Reflect to gray") {
        ImageCopyMakeBorder<DataType::kInt8, ImageType::kGRAY, CopyBorderType::kBorder_Reflect>(
            d_src, d_dst, height, width, top, bottom, left, right);
        cv::copyMakeBorder(src_f, expect, top, bottom, left, right, cv::BorderTypes::BORDER_REFLECT);
    }
    SECTION("kBorder_Reflect_101 to gray") {
        ImageCopyMakeBorder<DataType::kInt8, ImageType::kGRAY, CopyBorderType::kBorder_Reflect_101>(
            d_src, d_dst, height, width, top, bottom, left, right);
        cv::copyMakeBorder(src_f, expect, top, bottom, left, right, cv::BorderTypes::BORDER_REFLECT_101);
    }
    SECTION("kBorder_Warp to gray") {
        ImageCopyMakeBorder<DataType::kInt8, ImageType::kGRAY, CopyBorderType::kBorder_Warp>(
            d_src, d_dst, height, width, top, bottom, left, right);
        cv::copyMakeBorder(src_f, expect, top, bottom, left, right, cv::BorderTypes::BORDER_WRAP);
    }

    CUDA_CHECK(cudaMemcpy(actual.data, d_dst, sizeof(unsigned char) * actual.cols * actual.rows * actual.channels(),
                          cudaMemcpyDeviceToHost));

    int maxDiff = 0;
    for (int i = 0; i < actual.rows; i++) {
        for (int j = 0; j < actual.cols; j++) {
            int diff = abs(actual.at<unsigned char>(i, j) - expect.at<unsigned char>(i, j));
            if (diff > maxDiff) maxDiff = diff;
        }
    }
    REQUIRE(maxDiff < 1);
    CUDA_CHECK_AND_FREE(d_src);
    CUDA_CHECK_AND_FREE(d_dst);
}

TEST_CASE("ImageCopyMakeBorder to kBGR_HWC", "[ImageCopyMakeBorder]") {
    void *d_src, *d_dst;
    const int height = 1080, width = 1920;
    int top = 2000, bottom = 2000, left = 3000, right = 3000;
    int out_w = width + left + right;
    int out_h = height + top + bottom;

    std::vector<float> input_bgr(width * height * 3);
    smartmore::RandomFloatVector(input_bgr);

    cv::Mat src_f = cv::Mat(height, width, CV_32FC3, input_bgr.data());
    cv::Mat dst_f = cv::Mat::zeros(out_h, out_w, CV_32FC3);
    cv::Mat actual(out_h, out_w, CV_32FC3);
    cv::Mat expect(out_h, out_w, CV_32FC3);

    CUDA_CHECK(cudaMalloc(&d_src, sizeof(float) * src_f.cols * src_f.rows * src_f.channels()));
    CUDA_CHECK(cudaMalloc(&d_dst, sizeof(float) * dst_f.cols * dst_f.rows * dst_f.channels()));
    CUDA_CHECK(cudaMemcpy(d_src, src_f.data, sizeof(float) * src_f.cols * src_f.rows * src_f.channels(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dst, dst_f.data, sizeof(float) * dst_f.cols * dst_f.rows * dst_f.channels(),
                          cudaMemcpyHostToDevice));
    using namespace smartmore::cudaop;
    SECTION("kBorder_Replicate to kBGR_HWC") {
        ImageCopyMakeBorder<DataType::kFloat32, ImageType::kBGR_HWC, CopyBorderType::kBorder_Replicate>(
            d_src, d_dst, height, width, top, bottom, left, right);
        cv::copyMakeBorder(src_f, expect, top, bottom, left, right, cv::BorderTypes::BORDER_REPLICATE);
    }
    SECTION("kBorder_Constant to kBGR_HWC") {
        ImageCopyMakeBorder<DataType::kFloat32, ImageType::kBGR_HWC, CopyBorderType::kBorder_Constant>(
            d_src, d_dst, height, width, top, bottom, left, right, 255);
        cv::copyMakeBorder(src_f, expect, top, bottom, left, right, cv::BorderTypes::BORDER_CONSTANT,
                           cv::Scalar(255, 255, 255));
    }
    SECTION("kBorder_Reflect to kBGR_HWC") {
        ImageCopyMakeBorder<DataType::kFloat32, ImageType::kBGR_HWC, CopyBorderType::kBorder_Reflect>(
            d_src, d_dst, height, width, top, bottom, left, right);
        cv::copyMakeBorder(src_f, expect, top, bottom, left, right, cv::BorderTypes::BORDER_REFLECT);
    }
    SECTION("kBorder_Reflect_101 to kBGR_HWC") {
        ImageCopyMakeBorder<DataType::kFloat32, ImageType::kBGR_HWC, CopyBorderType::kBorder_Reflect_101>(
            d_src, d_dst, height, width, top, bottom, left, right);
        cv::copyMakeBorder(src_f, expect, top, bottom, left, right, cv::BorderTypes::BORDER_REFLECT_101);
    }
    SECTION("kBorder_Warp to kBGR_HWC") {
        ImageCopyMakeBorder<DataType::kFloat32, ImageType::kBGR_HWC, CopyBorderType::kBorder_Warp>(
            d_src, d_dst, height, width, top, bottom, left, right);
        cv::copyMakeBorder(src_f, expect, top, bottom, left, right, cv::BorderTypes::BORDER_WRAP);
    }

    CUDA_CHECK(cudaMemcpy(actual.data, d_dst, sizeof(float) * actual.cols * actual.rows * actual.channels(),
                          cudaMemcpyDeviceToHost));

    float max_diff = smartmore::CVMatMaxDiff(actual, expect);
    REQUIRE(max_diff < 0.0001);
    CUDA_CHECK_AND_FREE(d_src);
    CUDA_CHECK_AND_FREE(d_dst);
}

TEST_CASE("kHalf ImageCopyMakeBorder to kBGR_HWC", "[ImageCopyMakeBorder]") {
    void *src_float = nullptr, *dst_float = nullptr;
    void *src_half = nullptr, *dst_half = nullptr;
    const int height = 1080, width = 1920;
    int top = 100, bottom = 100, left = 100, right = 100;
    int out_w = width + left + right;
    int out_h = height + top + bottom;

    std::vector<float> input_bgr(width * height * 3);
    smartmore::RandomFloatVector(input_bgr);

    cv::Mat src_f = cv::Mat(height, width, CV_32FC3, input_bgr.data());
    cv::Mat dst_f = cv::Mat::zeros(out_h, out_w, CV_32FC3);
    cv::Mat actual(out_h, out_w, CV_32FC3);
    cv::Mat expect(out_h, out_w, CV_32FC3);

    CUDA_CHECK(cudaMalloc(&src_float, sizeof(float) * src_f.cols * src_f.rows * src_f.channels()));
    CUDA_CHECK(cudaMalloc(&src_half, sizeof(float) / 2 * src_f.cols * src_f.rows * src_f.channels()));
    CUDA_CHECK(cudaMalloc(&dst_float, sizeof(float) * dst_f.cols * dst_f.rows * dst_f.channels()));
    CUDA_CHECK(cudaMalloc(&dst_half, sizeof(float) / 2 * dst_f.cols * dst_f.rows * dst_f.channels()));
    CUDA_CHECK(cudaMemcpy(src_float, src_f.data, sizeof(float) * src_f.cols * src_f.rows * src_f.channels(),
                          cudaMemcpyHostToDevice));

    using namespace smartmore::cudaop;
    DataTypeConvert<DataType::kFloat32, DataType::kHalf>(src_float, src_half, height * width * src_f.channels());
    SECTION("kHalf kBorder_Replicate to kBGR_HWC") {
        ImageCopyMakeBorder<DataType::kHalf, ImageType::kBGR_HWC, CopyBorderType::kBorder_Replicate>(
            src_half, dst_half, height, width, top, bottom, left, right);
        cv::copyMakeBorder(src_f, expect, top, bottom, left, right, cv::BorderTypes::BORDER_REPLICATE);
    }
    SECTION("kHalf kBorder_Constant to kBGR_HWC") {
        ImageCopyMakeBorder<DataType::kHalf, ImageType::kBGR_HWC, CopyBorderType::kBorder_Constant>(
            src_half, dst_half, height, width, top, bottom, left, right, 255);
        cv::copyMakeBorder(src_f, expect, top, bottom, left, right, cv::BorderTypes::BORDER_CONSTANT,
                           cv::Scalar(255, 255, 255));
    }
    SECTION("kHalf kBorder_Reflect to kBGR_HWC") {
        ImageCopyMakeBorder<DataType::kHalf, ImageType::kBGR_HWC, CopyBorderType::kBorder_Reflect>(
            src_half, dst_half, height, width, top, bottom, left, right);
        cv::copyMakeBorder(src_f, expect, top, bottom, left, right, cv::BorderTypes::BORDER_REFLECT);
    }
    SECTION("kHalf kBorder_Reflect_101 to kBGR_HWC") {
        ImageCopyMakeBorder<DataType::kHalf, ImageType::kBGR_HWC, CopyBorderType::kBorder_Reflect_101>(
            src_half, dst_half, height, width, top, bottom, left, right);
        cv::copyMakeBorder(src_f, expect, top, bottom, left, right, cv::BorderTypes::BORDER_REFLECT_101);
    }
    SECTION("kHalf kBorder_Warp to kBGR_HWC") {
        ImageCopyMakeBorder<DataType::kHalf, ImageType::kBGR_HWC, CopyBorderType::kBorder_Warp>(
            src_half, dst_half, height, width, top, bottom, left, right);
        cv::copyMakeBorder(src_f, expect, top, bottom, left, right, cv::BorderTypes::BORDER_WRAP);
    }

    DataTypeConvert<DataType::kHalf, DataType::kFloat32>(dst_half, dst_float, out_h * out_w * actual.channels());
    CUDA_CHECK(cudaMemcpy(actual.data, dst_float, sizeof(float) * actual.cols * actual.rows * actual.channels(),
                          cudaMemcpyDeviceToHost));

    float max_diff = smartmore::CVMatMaxDiff(actual, expect);
    REQUIRE(max_diff < 0.001);
    CUDA_CHECK_AND_FREE(src_float);
    CUDA_CHECK_AND_FREE(src_half);
    CUDA_CHECK_AND_FREE(dst_float);
    CUDA_CHECK_AND_FREE(dst_half);
}