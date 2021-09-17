/*******************************************************************************
 *  FILENAME:      test_image_alphablend.cu
 *
 *  AUTHORS:       Liang Jia    START DATE: Saturday August 14th 2021
 *
 *  LAST MODIFIED: Monday, August 16th 2021, 1:33:28 pm
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

bool AlphaBlending(cv::Mat &bg_mat, const cv::Mat &fg_mat, const cv::Mat &mask_mat, cv::Point roi_ltpt) {
    if (bg_mat.empty() || fg_mat.empty() || mask_mat.empty()) {
        std::cout << "invalid images\n";
        return false;
    }
    if (!roi_ltpt.inside(cv::Rect(0, 0, bg_mat.cols - fg_mat.cols, bg_mat.rows - fg_mat.rows))) {
        std::cout << "blending roi must be inside the background image\n";
        return false;
    }

    if (bg_mat.type() != fg_mat.type() || mask_mat.type() != CV_32FC1) {
        std::cout << "background image and foreground image must be the same type\n";
        std::cout << "mask image must be gray type\n";
        return false;
    }

    cv::Mat mask;
    cv::cvtColor(mask_mat, mask, cv::COLOR_GRAY2BGR);

    cv::Mat work_mat = bg_mat(cv::Rect(roi_ltpt, fg_mat.size())).clone();

    // Find number of pixels.
    int numberOfPixels = fg_mat.rows * fg_mat.cols * fg_mat.channels();

    // Get floating point pointers to the data matrices
    float *fptr = reinterpret_cast<float *>(fg_mat.data);
    float *bptr = reinterpret_cast<float *>(work_mat.data);
    float *aptr = reinterpret_cast<float *>(mask.data);

    for (int i = 0; i < numberOfPixels; i++, fptr++, aptr++, bptr++) {
        *bptr = (*fptr) * (*aptr) + (*bptr) * (1 - *aptr);
    }
    work_mat.copyTo(bg_mat(cv::Rect(roi_ltpt, fg_mat.size())));
    return true;
}

TEST_CASE("ImageAlphaBlend", "[image_alphablend]") {
    // background
    const int bg_h = 1080, bg_w = 1920, bg_c = 3;
    // foreground
    const int fg_h = 540, fg_w = 960, fg_c = 3;
    // mask
    const int mask_h = fg_h, mask_w = fg_w, mask_c = 1;

    float *bg_device, *fg_device, *mask_device;
    std::vector<float> bg_data(bg_h * bg_w * bg_c);
    smartmore::RandomFloatVector(bg_data);
    std::vector<float> fg_data(fg_h * fg_w * fg_c);
    smartmore::RandomFloatVector(fg_data);
    std::vector<float> mask_data(mask_h * mask_w * mask_c);
    smartmore::RandomFloatVector(mask_data);

    cv::Mat bg_f = cv::Mat(bg_h, bg_w, CV_32FC3, bg_data.data());
    cv::Mat fg_f = cv::Mat(fg_h, fg_w, CV_32FC3, fg_data.data());
    cv::Mat mask_f = cv::Mat(mask_h, mask_w, CV_32FC1, mask_data.data());

    cv::Mat actual_mat = bg_f.clone();
    cv::Mat expet_mat = bg_f.clone();
    size_t bg_data_size = sizeof(float) * bg_f.cols * bg_f.rows * bg_f.channels();
    size_t fg_data_size = sizeof(float) * fg_f.cols * fg_f.rows * fg_f.channels();
    size_t mask_data_size = sizeof(float) * mask_f.cols * mask_f.rows * mask_f.channels();

    CUDA_CHECK(cudaMalloc(&bg_device, bg_data_size));
    CUDA_CHECK(cudaMemcpy(bg_device, bg_f.data, bg_data_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&fg_device, fg_data_size));
    CUDA_CHECK(cudaMemcpy(fg_device, fg_f.data, fg_data_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&mask_device, mask_data_size));
    CUDA_CHECK(cudaMemcpy(mask_device, mask_f.data, mask_data_size, cudaMemcpyHostToDevice));

    {
        smartmore::Clock clk("image alpha blend cost time: ");
        ImageAlphaBlend<ImageType::kBGR_HWC, DataType::kFloat32>(fg_device, bg_device, mask_device, Size{bg_w, bg_h},
                                                                 Rect{Point{100, 100}, Size{fg_w, fg_h}});
    }
    CUDA_CHECK(cudaMemcpy(actual_mat.data, bg_device, bg_data_size, cudaMemcpyDeviceToHost));

    if (!AlphaBlending(expet_mat, fg_f, mask_f, cv::Point(100, 100))) {
        REQUIRE(1 < 0.0001);
    }

    float max_diff = smartmore::CVMatMaxDiff(actual_mat, expet_mat);
    REQUIRE(max_diff < 0.0001);

    CUDA_CHECK_AND_FREE(bg_device);
    CUDA_CHECK_AND_FREE(fg_device);
    CUDA_CHECK_AND_FREE(mask_device);
}