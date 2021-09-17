/*******************************************************************************
 *  FILENAME:      test_image_stitching.cpp
 *
 *  AUTHORS:       Liang Jia    START DATE: Friday July 23rd 2021
 *
 *  LAST MODIFIED: Friday, August 6th 2021, 11:51:17 am
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

TEST_CASE("ImageStitching", "[image_stitching]") {
    const int src_h = 1080, src_w = 1920, src_c = 3;
    const int slice_h = 480, slice_w = 640;
    const int overlay_size = 10;
    auto slice_size = Size{slice_w, slice_h};
    int effective_width = slice_w - 2 * overlay_size;
    int effective_height = slice_h - 2 * overlay_size;
    int rows = std::ceil(src_h * 1.0f / effective_height);
    int cols = std::ceil(src_w * 1.0f / effective_width);
    int slices_count = rows * cols;
    void *src_device;
    void *src_device_actual;
    std::vector<void *> dst_devices(slices_count);

    std::vector<float> input_data(src_h * src_w * src_c);
    smartmore::RandomFloatVector(input_data);

    cv::Mat src_f = cv::Mat(src_h, src_w, CV_32FC3, input_data.data());
    std::vector<cv::Mat> actual_mats, expet_mats;
    int src_buffer_size = sizeof(float) * src_f.cols * src_f.rows * src_f.channels();
    int slice_buffer_size = sizeof(float) * slice_w * slice_h * src_f.channels();
    CUDA_CHECK(cudaMalloc(&src_device, src_buffer_size));
    CUDA_CHECK(cudaMemcpy(src_device, src_f.data, src_buffer_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&src_device_actual, src_buffer_size));
    for (size_t i = 0; i < slices_count; i++) {
        CUDA_CHECK(cudaMalloc(&dst_devices[i], slice_buffer_size));
        CUDA_CHECK(cudaMemset(dst_devices[i], 0, slice_buffer_size));
    }

    std::vector<Rect> sliced_rects;
    ImageSlice<ImageType::kBGR_HWC, DataType::kFloat32>(src_device, Size{src_w, src_h}, slice_size, dst_devices,
                                                        sliced_rects, overlay_size);

    ImageStitching<ImageType::kBGR_HWC, DataType::kFloat32>(dst_devices, src_device_actual, Size{src_w, src_h},
                                                            slice_size, sliced_rects, overlay_size);

    cv::Mat actual_mat(src_f.size(), src_f.type());
    CUDA_CHECK(cudaMemcpy(actual_mat.data, src_device_actual, src_buffer_size, cudaMemcpyDeviceToHost));

    float max_diff = smartmore::CVMatMaxDiff(actual_mat, src_f);

    REQUIRE(max_diff < 0.0001);

    CUDA_CHECK_AND_FREE(src_device);
    for (size_t i = 0; i < slices_count; i++) {
        CUDA_CHECK_AND_FREE(dst_devices[i]);
    }
    CUDA_CHECK_AND_FREE(src_device_actual);
}